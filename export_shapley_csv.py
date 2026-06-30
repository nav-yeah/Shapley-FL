"""
Per-round GTG-Shapley CSV export — THE STABLE HANDOFF FORMAT for P2/P3/P4.

IMPORTANT DISTINCTION from gtg_shapley_full_scale.py:
That script computed ONE Shapley value per client for the ENTIRE 5-round
trajectory (useful for validating the engine, and for an "end of training"
valuation). This script instead computes a Shapley value PER CLIENT PER
ROUND -- the incremental contribution of each client's update in that
specific round -- because:
  - P2 (adaptive aggregation) needs per-round scores to reweight FedAvg
    each round, not a single end-of-training number.
  - P3 (Byzantine detection via score drift) needs a TIME SERIES of scores
    to detect anomalous drift -- a single end-of-training number has no
    "drift" to observe.
  - P4 (blockchain audit trail) wants a verifiable log entry per round.

This uses Algorithm 1 from the paper directly (one round at a time, using
that round's v0 = V(M^t) and vN = V(M^t+1)), which is cheaper per call
than the full-trajectory replay used for validation, since each round's
sub-coalition evaluation only requires ONE local training step (not a
full 5-round replay).

Usage:
    python export_shapley_csv.py

Expects:
    preprocessed/har_clients.pkl

Produces:
    shapley_scores.csv   <- the file P2/P3/P4 build on
"""

import csv
import pickle
import time
import numpy as np

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR
)

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "shapley_scores.csv"
RANDOM_SEED = 42
GLOBAL_SEED = 42
N_ROUNDS = 5

EPSILON_B = 0.02
EPSILON_I = 0.005
M_GUIDED = 3
N_PERMUTATIONS = 200
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


def gtg_shapley_one_round(client_ids, client_states, client_sizes,
                           global_state_before, global_state_after,
                           X_test, y_test,
                           n_permutations=N_PERMUTATIONS, m=M_GUIDED,
                           eps_b=EPSILON_B, eps_i=EPSILON_I,
                           seed=RANDOM_SEED, verbose=False):
    """
    Algorithm 1 applied to a SINGLE round: estimates each client's
    contribution to THIS round's accuracy gain (v0 -> vN), using their
    already-trained client_states (this round's local updates) and
    reconstructing sub-models via FedAvg over those updates -- the
    actual gradient-reconstruction trick, cheap because no retraining
    is needed, just re-averaging already-computed local updates.
    """
    rng = np.random.RandomState(seed)
    n = len(client_ids)

    v0 = evaluate(global_state_before, X_test, y_test)
    vN = evaluate(global_state_after, X_test, y_test)

    phi = {cid: 0.0 for cid in client_ids}

    if abs(vN - v0) <= eps_b:
        if verbose:
            print(f"    Round truncated: |vN-v0|={abs(vN-v0):.4f} <= eps_b")
        return phi, v0, vN

    phi_history = []
    k = 0
    for k in range(1, n_permutations + 1):
        pi_k = guided_permutation(client_ids, m, k, rng)
        v_prev = v0
        evaluated_subset = []

        for j in range(1, n + 1):
            evaluated_subset.append(pi_k[j - 1])

            if abs(vN - v_prev) >= eps_i:
                sub_states = [client_states[c] for c in evaluated_subset]
                sub_weights = [client_sizes[c] for c in evaluated_subset]
                recon_state = fedavg(sub_states, sub_weights)
                v_j = evaluate(recon_state, X_test, y_test)
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

    if verbose:
        print(f"    Converged at permutation {k}")

    return phi, v0, vN


def run_and_export(client_ids, data, n_rounds=N_ROUNDS, global_seed=GLOBAL_SEED,
                    out_csv=OUT_CSV):
    X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
    y_test = np.concatenate([data[c]["y_test"] for c in client_ids])

    client_sizes = {cid: len(data[cid]["y_train"]) for cid in client_ids}
    global_state = clone_state_dict(get_model(seed=global_seed).state_dict())

    rows = []  # each row: round, client_id, shapley_value, model_accuracy

    for round_idx in range(1, n_rounds + 1):
        print(f"Round {round_idx}/{n_rounds}...")

        # --- Client-side training (this round's "gradient updates") ---
        client_states = {}
        for cid in client_ids:
            local_model = get_model()
            local_model.load_state_dict(global_state)
            X_i, y_i = data[cid]["X_train"], data[cid]["y_train"]
            client_states[cid] = local_train(
                local_model, X_i, y_i, epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
                seed=global_seed + round_idx * 100 + cid
            )

        # --- Server-side FedAvg ---
        weights = [client_sizes[c] for c in client_ids]
        states = [client_states[c] for c in client_ids]
        new_global_state = fedavg(states, weights)

        # --- GTG-Shapley for this round ---
        phi_round, v0, vN = gtg_shapley_one_round(
            client_ids, client_states, client_sizes,
            global_state, new_global_state, X_test, y_test
        )

        round_accuracy = vN  # global model accuracy AFTER this round

        for cid in client_ids:
            rows.append({
                "round": round_idx,
                "client_id": cid,
                "shapley_value": phi_round[cid],
                "model_accuracy": round_accuracy,
            })

        print(f"  v0={v0:.4f} -> vN={vN:.4f}, "
              f"phi range [{min(phi_round.values()):.4f}, "
              f"{max(phi_round.values()):.4f}]")

        global_state = new_global_state

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "client_id",
                                                 "shapley_value", "model_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_csv}")
    return rows


if __name__ == "__main__":
    print("Loading preprocessed HAR client data (all 30 clients)...")
    data = load_client_data(PKL_PATH)
    client_ids = sorted(data.keys())

    start = time.time()
    rows = run_and_export(client_ids, data)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"CSV format: round, client_id, shapley_value, model_accuracy")
    print(f"This is the file to hand off to P2 (adaptive aggregation), "
          f"P3 (Byzantine detection via score drift), and P4 (blockchain "
          f"audit logging).")
