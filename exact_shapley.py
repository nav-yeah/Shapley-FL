"""
Exact Shapley Value computation — PyTorch MLP / multi-round FL version.

This is your GROUND TRUTH baseline (phi*). Unlike the sklearn version,
V(S) here is defined properly for FL: train a fresh global model from
the SAME initial weights, but only let clients in S participate across
ALL rounds, then evaluate the resulting model.

This matches the FL Shapley value definition (Eq. 5 in GTG-Shapley):
    V(S) = V(M_S) = V(A(M(0), D_S))
where A is the FULL multi-round FL training algorithm, not a single fit.

Still O(2^n) coalitions -> keep N_CLIENTS_FOR_GROUND_TRUTH small (<=5).

Usage:
    python exact_shapley.py
"""

import pickle
import itertools
import time
import numpy as np
from math import comb

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR
)

PKL_PATH = "preprocessed/har_clients.pkl"
N_CLIENTS_FOR_GROUND_TRUTH = 5
N_ROUNDS = 5            # FL rounds per coalition evaluation
GLOBAL_SEED = 42
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_client_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def select_subset_clients(data, n, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    all_client_ids = list(data.keys())
    chosen = rng.choice(all_client_ids, size=n, replace=False)
    return sorted(chosen.tolist())


def train_coalition_fl(client_ids, data, n_rounds=N_ROUNDS,
                        global_seed=GLOBAL_SEED):
    """
    V(S): trains a global model for n_rounds using ONLY the clients in
    client_ids (the coalition S), starting from a FIXED initial seed so
    all coalitions are compared on equal footing. Returns final global
    state_dict.

    If client_ids is empty, returns None (caller handles baseline eval).
    """
    if len(client_ids) == 0:
        return None

    global_model = get_model(seed=global_seed)
    global_state = clone_state_dict(global_model.state_dict())

    client_sizes = {cid: len(data[cid]["y_train"]) for cid in client_ids}

    for r in range(n_rounds):
        states = []
        weights = []
        for cid in client_ids:
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


def exact_shapley_values(client_ids, data, X_test, y_test, verbose=True):
    """
    phi_i = sum over S subset of N minus {i} of
            [V(S U {i}) - V(S)] / C(n-1, |S|)

    V(S) now means "train the FULL FL pipeline for n_rounds using only
    clients in S, then evaluate" -- not a single model fit.
    """
    n = len(client_ids)
    utility_cache = {}

    def V(S):
        key = tuple(sorted(S))
        if key not in utility_cache:
            state = train_coalition_fl(list(key), data)
            utility_cache[key] = evaluate(state, X_test, y_test)
        return utility_cache[key]

    phi = {cid: 0.0 for cid in client_ids}
    start = time.time()

    total_subsets = 2 ** n
    done = 0

    for i in client_ids:
        others = [c for c in client_ids if c != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = list(S)
                v_with = V(S + [i])
                v_without = V(S)
                weight = 1.0 / comb(n - 1, len(S))
                phi[i] += weight * (v_with - v_without)
                done += 1
                if verbose and done % 10 == 0:
                    print(f"  ...evaluated {done} marginal contributions "
                          f"({len(utility_cache)} unique coalitions cached)")

        phi[i] /= n

    elapsed = time.time() - start
    print(f"\nExact Shapley computation done in {elapsed:.2f}s")
    print(f"Total unique coalitions evaluated: {len(utility_cache)} "
          f"(out of {total_subsets} possible)")
    return phi, elapsed, utility_cache


if __name__ == "__main__":
    print("Loading preprocessed HAR client data...")
    data = load_client_data(PKL_PATH)

    print(f"Selecting {N_CLIENTS_FOR_GROUND_TRUTH} clients for exact Shapley...")
    client_subset = select_subset_clients(data, N_CLIENTS_FOR_GROUND_TRUTH)
    print(f"Chosen clients: {client_subset}")

    X_test_parts, y_test_parts = [], []
    for cid in client_subset:
        X_test_parts.append(data[cid]["X_test"])
        y_test_parts.append(data[cid]["y_test"])
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    print(f"Shared test set size: {len(y_test)} samples")
    print(f"Each coalition trains for {N_ROUNDS} FL rounds "
          f"({LOCAL_EPOCHS} local epochs each) — this will take a while "
          f"since exact Shapley needs {2**N_CLIENTS_FOR_GROUND_TRUTH} "
          f"full coalition trainings.")

    print("\nComputing EXACT Shapley values (this is your ground truth)...")
    phi_exact, elapsed, cache = exact_shapley_values(
        client_subset, data, X_test, y_test
    )

    print("\n=== EXACT SHAPLEY VALUES (ground truth phi*) ===")
    for cid, val in phi_exact.items():
        print(f"  Client {cid}: phi* = {val:.6f}")

    results = {
        "client_subset": client_subset,
        "phi_exact": phi_exact,
        "elapsed_time": elapsed,
        "n_clients": N_CLIENTS_FOR_GROUND_TRUTH,
        "n_rounds": N_ROUNDS,
    }
    with open("exact_shapley_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved to exact_shapley_results.pkl")
