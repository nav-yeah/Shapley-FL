"""
GTG-Shapley approximation (Algorithm 1, Liu et al. 2022) — PyTorch version,
CORRECTED to match exact_shapley.py's definition of V(S).

V(S) here means: train a global model for n_rounds using ONLY clients in S,
starting from the same fixed init used by exact_shapley.py, then evaluate.
This is exactly what exact_shapley.py computes for ALL 32 coalitions --
GTG-Shapley instead computes it only for a guided, truncated SAMPLE of
coalitions (with caching so repeated coalitions across permutations are
free), which is where the speedup comes from.

Usage:
    python gtg_shapley_approx.py

Expects:
    preprocessed/har_clients.pkl
    exact_shapley_results.pkl   (for comparison; must match GLOBAL_SEED/N_ROUNDS)
"""

import pickle
import time
import numpy as np

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR
)

PKL_PATH = "preprocessed/har_clients.pkl"
EXACT_RESULTS_PATH = "exact_shapley_results.pkl"
RANDOM_SEED = 42
GLOBAL_SEED = 42  # MUST match exact_shapley.py's GLOBAL_SEED

# GTG-Shapley hyperparameters (tunable)
EPSILON_B = 0.02
EPSILON_I = 0.005
M_GUIDED = 3
N_PERMUTATIONS = 500
CONVERGENCE_WINDOW = 20
CONVERGENCE_THRESHOLD = 0.05


def load_client_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def guided_permutation(client_ids, m, k, rng):
    n = len(client_ids)
    m = min(m, n)
    rotated = client_ids[k % n:] + client_ids[:k % n]
    guided_part = rotated[:m]
    remaining = [c for c in client_ids if c not in guided_part]
    rng.shuffle(remaining)
    return guided_part + remaining


def train_subcoalition_fl(client_ids_subset, data, n_rounds,
                           global_seed=GLOBAL_SEED):
    """
    V(S): runs the FULL n_rounds trajectory using only client_ids_subset,
    starting from the same fixed seed as exact_shapley.py. This is the
    "gradient reconstruction" step -- still real training here (since our
    model is a small MLP, not literal stored gradients), but GTG-Shapley's
    speedup comes from only calling this for a small guided/truncated
    SAMPLE of coalitions instead of all 2^n, with caching for repeats.
    """
    if len(client_ids_subset) == 0:
        return None

    global_state = clone_state_dict(get_model(seed=global_seed).state_dict())
    client_sizes = {cid: len(data[cid]["y_train"]) for cid in client_ids_subset}

    for r in range(n_rounds):
        states, weights = [], []
        for cid in client_ids_subset:
            local_model = get_model()
            local_model.load_state_dict(global_state)
            X_i, y_i = data[cid]["X_train"], data[cid]["y_train"]
            new_state = local_train(local_model, X_i, y_i,
                                     epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
                                     seed=global_seed + r * 100 + cid)
            states.append(new_state)
            weights.append(client_sizes[cid])
        global_state = fedavg(states, weights)

    return global_state


def gtg_shapley_full_trajectory(client_ids, data, X_test, y_test,
                                 n_rounds=5, n_permutations=N_PERMUTATIONS,
                                 m=M_GUIDED, eps_b=EPSILON_B, eps_i=EPSILON_I,
                                 global_seed=GLOBAL_SEED, seed=RANDOM_SEED,
                                 verbose=True):
    """
    Estimates Shapley values for the FULL n_rounds FL trajectory, matching
    exact_shapley.py's V(S) exactly, via guided sampling + truncation
    instead of brute-force enumeration of all 2^n coalitions.

    Caches V(S) per coalition -- repeated coalitions across permutations
    are never recomputed. This caching IS the source of speedup, since
    each individual V(S) call here is just as expensive as in exact
    Shapley (both retrain a small MLP for n_rounds); GTG-Shapley wins by
    needing far fewer DISTINCT coalition evaluations, not by making each
    one cheaper.
    """
    rng = np.random.RandomState(seed)
    n = len(client_ids)
    utility_cache = {}

    def V(S):
        key = tuple(sorted(S))
        if key not in utility_cache:
            state = train_subcoalition_fl(list(key), data, n_rounds, global_seed)
            utility_cache[key] = evaluate(state, X_test, y_test)
        return utility_cache[key]

    v0 = V([])
    vN = V(client_ids)

    phi = {cid: 0.0 for cid in client_ids}

    if abs(vN - v0) <= eps_b:
        if verbose:
            print(f"Truncated entirely: |vN-v0|={abs(vN-v0):.4f} <= eps_b={eps_b}")
        return phi, len(utility_cache)

    phi_history = []
    k = 0
    for k in range(1, n_permutations + 1):
        pi_k = guided_permutation(client_ids, m, k, rng)
        v_prev = v0
        evaluated_subset = []

        for j in range(1, n + 1):
            evaluated_subset.append(pi_k[j - 1])

            if abs(vN - v_prev) >= eps_i:
                v_j = V(evaluated_subset)
            else:
                v_j = v_prev  # within-permutation truncation

            marginal = v_j - v_prev
            cid_j = pi_k[j - 1]
            phi[cid_j] = ((k - 1) / k) * phi[cid_j] + (1 / k) * marginal
            v_prev = v_j

        phi_history.append(dict(phi))

        if k >= max(CONVERGENCE_WINDOW + 1, 499):
            old_phi = phi_history[k - 1 - CONVERGENCE_WINDOW]
            diffs = []
            for cid in client_ids:
                denom = abs(phi[cid]) if abs(phi[cid]) > 1e-9 else 1e-9
                diffs.append(abs(phi[cid] - old_phi[cid]) / denom)
            if np.mean(diffs) < CONVERGENCE_THRESHOLD:
                if verbose:
                    print(f"Converged at permutation {k}")
                break

    if verbose:
        print(f"Ran {k} permutations, {len(utility_cache)} unique coalitions "
              f"evaluated (vs {2**n} for exact Shapley)")

    return phi, len(utility_cache)


def compare_to_ground_truth(phi_approx, phi_exact, client_ids):
    phi_star_vec = np.array([phi_exact[c] for c in client_ids])
    phi_vec = np.array([phi_approx[c] for c in client_ids])

    cos_sim = np.dot(phi_star_vec, phi_vec) / (
        np.linalg.norm(phi_star_vec) * np.linalg.norm(phi_vec) + 1e-12
    )
    cosine_distance = 1 - cos_sim
    euclidean_distance = np.linalg.norm(phi_star_vec - phi_vec)
    max_difference = np.max(np.abs(phi_star_vec - phi_vec))

    return {
        "cosine_distance": cosine_distance,
        "euclidean_distance": euclidean_distance,
        "max_difference": max_difference,
    }


if __name__ == "__main__":
    print("Loading preprocessed HAR client data...")
    data = load_client_data(PKL_PATH)

    print("Loading exact Shapley ground truth...")
    with open(EXACT_RESULTS_PATH, "rb") as f:
        exact_results = pickle.load(f)
    client_ids = exact_results["client_subset"]
    phi_exact = exact_results["phi_exact"]
    n_rounds = exact_results.get("n_rounds", 5)
    print(f"Validating against ground truth for clients: {client_ids} "
          f"over {n_rounds} rounds (full trajectory per coalition)")

    X_test_parts, y_test_parts = [], []
    for cid in client_ids:
        X_test_parts.append(data[cid]["X_test"])
        y_test_parts.append(data[cid]["y_test"])
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    print("\nRunning GTG-Shapley approximation (this calls full n_rounds "
          "training per UNIQUE coalition sampled, so still nontrivial "
          "time, but far fewer coalitions than exact Shapley's 32)...")
    start = time.time()
    phi_approx, n_unique_coalitions = gtg_shapley_full_trajectory(
        client_ids, data, X_test, y_test, n_rounds=n_rounds
    )
    elapsed = time.time() - start

    print("\n=== GTG-SHAPLEY APPROXIMATE VALUES ===")
    for cid in client_ids:
        print(f"  Client {cid}: phi_approx = {phi_approx[cid]:.6f}  "
              f"(phi_exact = {phi_exact[cid]:.6f})")

    print(f"\nApproximation time: {elapsed:.2f}s "
          f"(exact Shapley took {exact_results['elapsed_time']:.2f}s)")
    print(f"Unique coalitions evaluated: {n_unique_coalitions} "
          f"(exact Shapley evaluated all {2**len(client_ids)})")

    metrics = compare_to_ground_truth(phi_approx, phi_exact, client_ids)
    print("\n=== ACCURACY VS GROUND TRUTH ===")
    print(f"  Cosine Distance:    {metrics['cosine_distance']:.6f}")
    print(f"  Euclidean Distance: {metrics['euclidean_distance']:.6f}")
    print(f"  Maximum Difference: {metrics['max_difference']:.6f}")

    results = {
        "client_ids": client_ids,
        "phi_approx": phi_approx,
        "phi_exact": phi_exact,
        "metrics": metrics,
        "elapsed_time": elapsed,
        "n_unique_coalitions": n_unique_coalitions,
        "hyperparams": {
            "epsilon_b": EPSILON_B,
            "epsilon_i": EPSILON_I,
            "m_guided": M_GUIDED,
            "n_permutations": N_PERMUTATIONS,
        },
    }
    with open("gtg_shapley_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved to gtg_shapley_results.pkl")
