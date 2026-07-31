"""
GTG-Shapley at FULL SCALE (all 30 HAR clients) — efficiency measurement only.

At n=30, exact Shapley (2^30 coalitions) is computationally impossible,
so there's no ground truth to compare against here. This script's only
purpose is to measure:
  - wall-clock time for the approximation
  - how many unique coalitions get evaluated (vs the theoretical 2^30)
  - the resulting Shapley value distribution across all 30 clients

This becomes Table/Figure 2 in your experimentation section: efficiency
at the scale your actual FL deployment would run at.

Usage:
    python gtg_shapley_full_scale.py

Expects:
    preprocessed/har_clients.pkl
"""

import pickle
import time
import numpy as np

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR
)

PKL_PATH = "preprocessed/har_clients.pkl"
RANDOM_SEED = 42
GLOBAL_SEED = 42
N_ROUNDS = 50

# Same hyperparameters validated in the n=5 experiment
EPSILON_B = 0.02
EPSILON_I = 0.005
M_GUIDED = 3
N_PERMUTATIONS = 300       # may need more permutations at larger n
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


def train_subcoalition_fl(client_ids_subset, data, n_rounds, global_seed=GLOBAL_SEED):
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
                                 n_rounds=N_ROUNDS, n_permutations=N_PERMUTATIONS,
                                 m=M_GUIDED, eps_b=EPSILON_B, eps_i=EPSILON_I,
                                 global_seed=GLOBAL_SEED, seed=RANDOM_SEED,
                                 verbose=True):
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
    print(f"V(empty) = {v0:.4f}, V(full 30 clients) = {vN:.4f}")

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
                v_j = v_prev

            marginal = v_j - v_prev
            cid_j = pi_k[j - 1]
            phi[cid_j] = ((k - 1) / k) * phi[cid_j] + (1 / k) * marginal
            v_prev = v_j

        phi_history.append(dict(phi))

        if k >= max(CONVERGENCE_WINDOW + 1, 50):
            old_phi = phi_history[k - 1 - CONVERGENCE_WINDOW]
            diffs = []
            for cid in client_ids:
                denom = abs(phi[cid]) if abs(phi[cid]) > 1e-9 else 1e-9
                diffs.append(abs(phi[cid] - old_phi[cid]) / denom)
            if np.mean(diffs) < CONVERGENCE_THRESHOLD:
                if verbose:
                    print(f"Converged at permutation {k}")
                break

        if verbose and k % 25 == 0:
            print(f"  ...permutation {k}/{n_permutations}, "
                  f"{len(utility_cache)} unique coalitions so far")

    if verbose:
        print(f"\nRan {k} permutations, {len(utility_cache)} unique coalitions "
              f"evaluated (theoretical max for exact Shapley: 2^{n} = {2**n})")

    return phi, len(utility_cache)


if __name__ == "__main__":
    print("Loading preprocessed HAR client data (all 30 clients)...")
    data = load_client_data(PKL_PATH)
    client_ids = sorted(data.keys())
    print(f"Clients: {client_ids}")

    X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
    y_test = np.concatenate([data[c]["y_test"] for c in client_ids])
    print(f"Shared test set size: {len(y_test)} samples")

    print(f"\nRunning GTG-Shapley at FULL SCALE (n=30)...")
    print("NOTE: this will take noticeably longer than the n=5 run, since "
          "each unique coalition still requires a full 5-round FL training, "
          "and there may be more unique coalitions to evaluate.\n")

    start = time.time()
    phi, n_unique = gtg_shapley_full_trajectory(client_ids, data, X_test, y_test)
    elapsed = time.time() - start

    print("\n=== SHAPLEY VALUES (all 30 clients) ===")
    for cid in client_ids:
        print(f"  Client {cid}: phi = {phi[cid]:.6f}")

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Unique coalitions evaluated: {n_unique} "
          f"(vs theoretical 2^30 = {2**30:,} for exact Shapley)")
    print(f"Efficiency ratio: {n_unique}/{2**30} = {n_unique/2**30:.2e}")

    results = {
        "client_ids": client_ids,
        "phi": phi,
        "elapsed_time": elapsed,
        "n_unique_coalitions": n_unique,
        "n_rounds": N_ROUNDS,
        "hyperparams": {
            "epsilon_b": EPSILON_B,
            "epsilon_i": EPSILON_I,
            "m_guided": M_GUIDED,
            "n_permutations": N_PERMUTATIONS,
        },
    }
    with open("gtg_shapley_full_scale_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved to gtg_shapley_full_scale_results.pkl")
