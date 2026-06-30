"""
Runs GTG-Shapley approximation across all 5 distribution scenarios and
collects results into a single comparison table -- mirrors the paper's
Figures 6-10 structure (one efficiency/accuracy measurement per scenario).

NOTE: this does NOT compute exact Shapley ground truth per scenario
(that would require 2^10 = 1024 full coalition trainings PER scenario --
too slow). Instead, this measures:
  - wall-clock time per scenario
  - number of unique coalitions evaluated per scenario
  - the resulting Shapley value distribution (spread/variance) per scenario
This is consistent with how you'd report SCALABILITY/efficiency trends
across distributions, while your earlier n=5 experiment remains your one
formal accuracy-vs-ground-truth validation.

Usage:
    python run_scenarios.py

Expects: scenarios/scenario_*.pkl (from build_scenarios.py)
Produces: scenario_results.csv
"""

import os
import csv
import pickle
import time
import numpy as np

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR
)

SCENARIO_DIR = "scenarios"
OUT_CSV = "scenario_results.csv"
GLOBAL_SEED = 42
RANDOM_SEED = 42
N_ROUNDS = 5

EPSILON_B = 0.02
EPSILON_I = 0.005
M_GUIDED = 3
N_PERMUTATIONS = 200
CONVERGENCE_WINDOW = 20
CONVERGENCE_THRESHOLD = 0.05

SCENARIO_FILES = [
    "scenario_1_same_dist_same_size",
    "scenario_2_diff_dist_same_size",
    "scenario_3_same_dist_diff_size",
    "scenario_4_noisy_labels",
    "scenario_5_noisy_features",
]


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
                                 global_seed=GLOBAL_SEED, seed=RANDOM_SEED):
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
        return phi, len(utility_cache), v0, vN

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

        if k >= max(CONVERGENCE_WINDOW + 1, 30):
            old_phi = phi_history[k - 1 - CONVERGENCE_WINDOW]
            diffs = []
            for cid in client_ids:
                denom = abs(phi[cid]) if abs(phi[cid]) > 1e-9 else 1e-9
                diffs.append(abs(phi[cid] - old_phi[cid]) / denom)
            if np.mean(diffs) < CONVERGENCE_THRESHOLD:
                break

    return phi, len(utility_cache), v0, vN


if __name__ == "__main__":
    all_results = []

    for scenario_name in SCENARIO_FILES:
        path = os.path.join(SCENARIO_DIR, f"{scenario_name}.pkl")
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

        with open(path, "rb") as f:
            scenario = pickle.load(f)
        client_ids = scenario["client_ids"]
        data = scenario["data"]

        X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
        y_test = np.concatenate([data[c]["y_test"] for c in client_ids])

        start = time.time()
        phi, n_unique, v0, vN = gtg_shapley_full_trajectory(
            client_ids, data, X_test, y_test
        )
        elapsed = time.time() - start

        n = len(client_ids)
        phi_values = list(phi.values())
        phi_sum = sum(phi_values)
        phi_std = float(np.std(phi_values))
        phi_min = min(phi_values)
        phi_max = max(phi_values)

        print(f"V(empty)={v0:.4f}, V(full)={vN:.4f}, V(full)-V(empty)={vN-v0:.4f}")
        print(f"Sum(phi)={phi_sum:.4f} (efficiency check: should ~= V(full)-V(empty))")
        print(f"Phi std={phi_std:.4f}, range=[{phi_min:.4f}, {phi_max:.4f}]")
        print(f"Time: {elapsed:.2f}s, unique coalitions: {n_unique}/{2**n}")

        all_results.append({
            "scenario": scenario_name,
            "n_clients": n,
            "v_empty": v0,
            "v_full": vN,
            "marginal_total": vN - v0,
            "phi_sum": phi_sum,
            "phi_std": phi_std,
            "phi_min": phi_min,
            "phi_max": phi_max,
            "time_seconds": elapsed,
            "unique_coalitions": n_unique,
            "theoretical_max_coalitions": 2 ** n,
            "efficiency_ratio": n_unique / (2 ** n),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n\nAll scenarios complete. Saved comparison table to {OUT_CSV}")
    print("\n=== SUMMARY TABLE ===")
    for r in all_results:
        print(f"{r['scenario']:35s} time={r['time_seconds']:6.1f}s  "
              f"coalitions={r['unique_coalitions']:4d}/{r['theoretical_max_coalitions']}  "
              f"phi_std={r['phi_std']:.4f}")
