"""
P2 (extension) — FedProx baseline vs Vanilla FedAvg vs Shapley-weighted FedAvg.

WHY THIS EXISTS
---------------
The main P2 deliverable (shapley_weighted_fedavg.py) compares vanilla
size-weighted FedAvg against Shapley-weighted FedAvg. This script adds a
THIRD arm — FedProx (Li et al., MLSys 2020) — on the SAME natural
30-subject HAR partition, SAME initial model (GLOBAL_SEED=42), SAME 5
rounds, and SAME per-client training seeds, so all three curves are
directly comparable and strengthen the baseline comparison for the paper.

WHAT FEDPROX CHANGES (and what it doesn't)
------------------------------------------
FedProx does NOT change server aggregation — it is still size-weighted
averaging, exactly like vanilla FedAvg. What it changes is the CLIENT
objective: each client adds a proximal term (mu/2)*||w - w_global||^2 to
its local loss, so local updates are pulled toward the current global
model. This curbs client drift under non-IID data and stabilises training.
Concretely, the only change vs your fl_utils.local_train is one extra
gradient term  mu * (w - w_global)  applied to every parameter each step.
mu = 0 recovers vanilla FedAvg exactly.

CONSISTENCY WITH YOUR PIPELINE
------------------------------
- Imports your model, FedAvg, evaluate, and constants from fl_utils.
- Reuses gtg_shapley_one_round (P1's engine) and shapley_to_weights /
  train_all_clients from shapley_weighted_fedavg.py for the Shapley arm,
  so the Shapley curve reproduces your committed aggregation_results.csv
  run (BETA=50, size-weight fallback on truncated rounds).
- The vanilla arm reproduces your vanilla curve.
- Only the FedProx arm is new.

FEDPROX LOCAL TRAINING
----------------------
Mirrors fl_utils.local_train (same SGD, same batch size, same epoch/seed
convention) with the added proximal gradient. Kept local to this file so
nothing in the shared pipeline changes.

Usage (from repo root, venv active, after preprocess_har.py):

    python fedprox_compare.py            # default mu = 0.1
    python fedprox_compare.py 0.01       # try a different proximal strength

Runtime: ~2-3 min on CPU (three arms x 5 rounds; Shapley arm dominates).

Outputs:
    fedprox_comparison.csv   round, acc_vanilla, acc_shapley, acc_fedprox
    fedprox_comparison.png   three convergence curves
    fedprox_comparison.txt   summary + final-round table
"""

import csv
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fl_utils import (
    get_model, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR,
)
from export_shapley_csv import gtg_shapley_one_round
from shapley_weighted_fedavg import (
    shapley_to_weights, train_all_clients, GLOBAL_SEED, N_ROUNDS, BETA,
)

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "fedprox_comparison.csv"
OUT_PNG = "fedprox_comparison.png"
OUT_TXT = "fedprox_comparison.txt"

MU = 0.1                 # FedProx proximal strength (mu=0 => vanilla FedAvg)
BATCH_SIZE = 32          # same as fl_utils.local_train default
CONV_TARGETS = [0.50, 0.60, 0.70]


# ----------------------------------------------------------------------
# FedProx local training: fl_utils.local_train + proximal term
# ----------------------------------------------------------------------
def fedprox_local_train(model, global_state, X, y, mu,
                         epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
                         batch_size=BATCH_SIZE, seed=0):
    """
    Identical to fl_utils.local_train, but every SGD step also adds the
    FedProx proximal gradient  mu * (w - w_global)  to each parameter,
    pulling the local model toward the round's global weights. Returns the
    resulting state_dict. mu=0 reproduces fl_utils.local_train exactly.
    """
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    # Snapshot of the global weights this round started from (detached),
    # aligned to model.parameters() order.
    global_params = [global_state[k].detach().clone()
                     for k in model.state_dict().keys()]

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # FedProx: add mu*(w - w_global) to each parameter's gradient.
            if mu > 0:
                for p, g in zip(model.parameters(), global_params):
                    if p.grad is not None:
                        p.grad.add_(mu * (p.data - g))
            optimizer.step()

    return clone_state_dict(model.state_dict())


def train_all_clients_fedprox(client_ids, data, global_state, round_idx,
                               mu, seed=GLOBAL_SEED):
    """One FedProx round of local training for every client, from
    global_state. Same seeding convention as train_all_clients so batch
    orderings match the other two arms."""
    client_states = {}
    for cid in client_ids:
        local_model = get_model()
        local_model.load_state_dict(global_state)
        client_states[cid] = fedprox_local_train(
            local_model, global_state,
            data[cid]["X_train"], data[cid]["y_train"], mu,
            epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
            seed=seed + round_idx * 100 + cid,
        )
    return client_states


def rounds_to_target(acc_history, target):
    for r, a in enumerate(acc_history):
        if a >= target:
            return r
    return "-"


def main():
    mu = MU
    if len(sys.argv) > 1:
        mu = float(sys.argv[1])

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    client_ids = sorted(data.keys())

    X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
    y_test = np.concatenate([data[c]["y_test"] for c in client_ids])
    client_sizes = {c: len(data[c]["y_train"]) for c in client_ids}

    # All three arms start from the IDENTICAL initial model.
    init_state = clone_state_dict(get_model(seed=GLOBAL_SEED).state_dict())
    state_v = clone_state_dict(init_state)   # vanilla
    state_s = clone_state_dict(init_state)   # shapley-weighted
    state_p = clone_state_dict(init_state)   # fedprox

    acc0 = evaluate(init_state, X_test, y_test)
    print(f"Initial (round 0) accuracy: {acc0:.4f}  |  FedProx mu = {mu:g}")

    hist_v, hist_s, hist_p = [acc0], [acc0], [acc0]

    t0 = time.time()
    for r in range(1, N_ROUNDS + 1):
        print(f"\nRound {r}/{N_ROUNDS}")

        # ---- (1) Vanilla FedAvg ----
        cs_v = train_all_clients(client_ids, data, state_v, r)
        state_v = fedavg([cs_v[c] for c in client_ids],
                         [client_sizes[c] for c in client_ids])
        acc_v = evaluate(state_v, X_test, y_test)
        hist_v.append(acc_v)

        # ---- (2) Shapley-weighted FedAvg (reproduces your main run) ----
        cs_s = train_all_clients(client_ids, data, state_s, r)
        provisional = fedavg([cs_s[c] for c in client_ids],
                             [client_sizes[c] for c in client_ids])
        phi, v0, vN = gtg_shapley_one_round(
            client_ids, cs_s, client_sizes,
            state_s, provisional, X_test, y_test)
        weights = shapley_to_weights(phi, client_ids, client_sizes)
        state_s = fedavg([cs_s[c] for c in client_ids],
                         [weights[c] for c in client_ids])
        acc_s = evaluate(state_s, X_test, y_test)
        hist_s.append(acc_s)

        # ---- (3) FedProx (proximal client objective, size-weighted agg) ----
        cs_p = train_all_clients_fedprox(client_ids, data, state_p, r, mu)
        state_p = fedavg([cs_p[c] for c in client_ids],
                         [client_sizes[c] for c in client_ids])
        acc_p = evaluate(state_p, X_test, y_test)
        hist_p.append(acc_p)

        print(f"  vanilla={acc_v:.4f}  shapley={acc_s:.4f}  "
              f"fedprox={acc_p:.4f}")

    elapsed = time.time() - t0

    # ---- write CSV ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "round", "acc_vanilla", "acc_shapley", "acc_fedprox"])
        w.writeheader()
        for r in range(N_ROUNDS + 1):
            w.writerow({"round": r, "acc_vanilla": hist_v[r],
                        "acc_shapley": hist_s[r], "acc_fedprox": hist_p[r]})

    # ---- plot ----
    rounds_axis = list(range(N_ROUNDS + 1))
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(rounds_axis, hist_v, "o-", label="Vanilla FedAvg")
    plt.plot(rounds_axis, hist_s, "s-",
             label=f"Shapley-weighted (softmax, beta={BETA:g})")
    plt.plot(rounds_axis, hist_p, "^-", label=f"FedProx (mu={mu:g})")
    plt.xlabel("Global round")
    plt.ylabel("Global test accuracy")
    plt.title("Aggregation baselines — UCI HAR, 30 clients, seed 42")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)

    # ---- summary ----
    lines = []
    lines.append("FedProx vs Vanilla vs Shapley-weighted FedAvg (UCI HAR)")
    lines.append("=" * 55)
    lines.append(f"clients={len(client_ids)}  rounds={N_ROUNDS}  "
                 f"local_epochs={LOCAL_EPOCHS}  lr={LOCAL_LR}  "
                 f"beta={BETA:g}  mu(FedProx)={mu:g}  seed={GLOBAL_SEED}")
    lines.append("")
    lines.append("round   vanilla   shapley   fedprox")
    for r in range(N_ROUNDS + 1):
        lines.append(f"  {r:<4d} {hist_v[r]:.4f}   {hist_s[r]:.4f}   "
                     f"{hist_p[r]:.4f}")
    lines.append("")
    lines.append(f"Final  vanilla={hist_v[-1]:.4f}  "
                 f"shapley={hist_s[-1]:.4f}  fedprox={hist_p[-1]:.4f}")
    lines.append(f"Shapley - FedProx (final): {hist_s[-1] - hist_p[-1]:+.4f}")
    lines.append(f"Shapley - Vanilla (final): {hist_s[-1] - hist_v[-1]:+.4f}")
    lines.append("")
    lines.append("Convergence speed (first round reaching target accuracy):")
    lines.append("  target   vanilla   shapley   fedprox")
    for t in CONV_TARGETS:
        lines.append(f"  >={t:.2f}    "
                     f"{str(rounds_to_target(hist_v, t)):>5}     "
                     f"{str(rounds_to_target(hist_s, t)):>5}     "
                     f"{str(rounds_to_target(hist_p, t)):>5}")
    lines.append("")
    lines.append("Note: FedProx modifies the CLIENT objective (proximal term "
                 "toward the global model); server aggregation stays size-"
                 "weighted like vanilla. It is a client-drift / stability "
                 "baseline for non-IID data, complementary to contribution-"
                 "based reweighting. On the honest, mildly non-IID 30-subject "
                 "HAR partition (every client holds all 6 classes), FedProx "
                 "is expected to track vanilla closely; the proximal term "
                 "matters more under stronger skew.")
    lines.append("")
    lines.append(f"(run time {elapsed:.1f}s)")
    text = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print()
    print(text)
    print(f"\nWrote {OUT_CSV}, {OUT_PNG}, {OUT_TXT}")


if __name__ == "__main__":
    main()
