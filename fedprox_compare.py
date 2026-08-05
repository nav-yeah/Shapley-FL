"""
P2 (extension) — FedProx baseline vs Vanilla FedAvg vs Shapley-weighted FedAvg.

WHY THIS EXISTS
---------------
The main P2 deliverable (shapley_weighted_fedavg.py) compares vanilla
size-weighted FedAvg against Shapley-weighted FedAvg. This script adds a
THIRD arm — FedProx (Li et al., MLSys 2020) — on the SAME natural
30-subject HAR partition, SAME initial model (GLOBAL_SEED=42), SAME
number of rounds, and SAME per-client training seeds, so all three curves
are directly comparable and strengthen the baseline comparison for the
paper.

WHAT FEDPROX CHANGES (and what it doesn't)
------------------------------------------
FedProx does NOT change server aggregation — it is still size-weighted
averaging, exactly like vanilla FedAvg. What it changes is the CLIENT
objective: each client adds a proximal term (mu/2)*||w - w_global||^2 to
its local loss, so local updates are pulled toward the current global
model. This curbs client drift under non-IID data and stabilises training.
Concretely, the only change vs fl_utils.local_train is one extra gradient
term  mu * (w - w_global)  applied to every parameter each step.
mu = 0 recovers vanilla FedAvg exactly.

50-ROUND UPDATE
---------------
Three things changed when the pipeline moved from 5 rounds to 50:

1. N_ROUNDS is inherited from shapley_weighted_fedavg, so this script
   now runs 50 rounds automatically. That is three training arms over 50
   rounds — budget roughly 15-40 min depending on the machine. Progress
   and ETA print every round and the CSV is flushed every round, so an
   interrupted run keeps its completed rounds.

2. The Shapley arm inherits the GTG truncation problem. Past round ~22
   the per-round accuracy gain drops below eps_b = 0.02 and GTG returns
   all-zero scores, so the Shapley arm silently becomes vanilla FedAvg.
   Run with --adaptive to use the lower threshold and keep the arm
   genuinely distinct; either way the truncated-round count is reported
   so the comparison is not read as stronger than it is.

3. Final-round accuracy is no longer the only summary. After the model
   plateaus a single round is noisy enough to flip the ranking of three
   arms that sit within a fraction of a point of each other, so the
   summary also reports best-round accuracy and the mean over the last
   10 rounds.

WHY FEDPROX IS THE RIGHT THIRD ARM (and what it cannot show)
-------------------------------------------------------------
FedProx and Shapley reweighting attack non-IID from opposite ends: one
constrains the CLIENT update, the other reweights the SERVER aggregate.
They are complementary rather than competing, so on the honest,
mildly non-IID 30-subject HAR partition — where every client holds all 6
activity classes — FedProx is expected to track vanilla closely. That is
a real finding, not a failed experiment: it says the natural partition is
not skewed enough for client drift to be the binding constraint, which is
exactly why dirichlet_alpha_sweep.py exists.

CONSISTENCY WITH THE PIPELINE
-----------------------------
- Imports the model, FedAvg, evaluate, and constants from fl_utils.
- Reuses gtg_shapley_one_round (P1's engine) and shapley_to_weights /
  train_all_clients from shapley_weighted_fedavg.py for the Shapley arm,
  so the Shapley curve reproduces the main run (BETA=50, size-weight
  fallback on truncated rounds).
- The vanilla arm reproduces the main run's vanilla curve.
- Only the FedProx arm is new.

FEDPROX LOCAL TRAINING
----------------------
Mirrors fl_utils.local_train (same SGD, same batch size, same epoch/seed
convention) with the added proximal gradient. Kept local to this file so
nothing in the shared pipeline changes.

Usage (from repo root, venv active, after preprocess_har.py):

    python fedprox_compare.py                    # 50 rounds, mu = 0.1
    python fedprox_compare.py --mu 0.01          # different proximal strength
    python fedprox_compare.py --rounds 3         # quick smoke test
    python fedprox_compare.py --adaptive --suffix _adaptive

Outputs:
    fedprox_comparison.csv   round, acc_vanilla, acc_shapley, acc_fedprox
    fedprox_comparison.png   three convergence curves, truncation shaded
    fedprox_comparison.txt   summary + convergence table
"""

import argparse
import csv
import pickle
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
    shapley_to_weights, train_all_clients, fmt_eta,
    GLOBAL_SEED, N_ROUNDS, BETA,
    EPS_B_FIXED, EPS_I_FIXED, EPS_B_FLOOR, EPS_I_RATIO,
)

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "fedprox_comparison.csv"
OUT_PNG = "fedprox_comparison.png"
OUT_TXT = "fedprox_comparison.txt"

MU = 0.1                 # FedProx proximal strength (mu=0 => vanilla FedAvg)
BATCH_SIZE = 32          # same as fl_utils.local_train default
CONV_TARGETS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
PLATEAU_WINDOW = 10


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
    ap = argparse.ArgumentParser(
        description="FedProx vs vanilla vs Shapley-weighted FedAvg")
    ap.add_argument("--mu", type=float, default=MU,
                    help=f"FedProx proximal strength (default {MU:g})")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS,
                    help=f"number of FL rounds (default {N_ROUNDS})")
    ap.add_argument("--beta", type=float, default=BETA,
                    help=f"softmax inverse temperature (default {BETA:g})")
    ap.add_argument("--adaptive", action="store_true",
                    help=f"use eps_b={EPS_B_FLOOR:g} for the Shapley arm "
                         f"instead of P1's {EPS_B_FIXED:g}")
    ap.add_argument("--suffix", default="", help="suffix for output filenames")
    args = ap.parse_args()

    mu = args.mu
    n_rounds = args.rounds
    beta = args.beta

    if args.adaptive:
        eps_b, eps_i = EPS_B_FLOOR, EPS_B_FLOOR * EPS_I_RATIO
        mode = "adaptive"
    else:
        eps_b, eps_i = EPS_B_FIXED, EPS_I_FIXED
        mode = "fixed (P1 defaults)"

    out_csv = OUT_CSV.replace(".csv", f"{args.suffix}.csv")
    out_png = OUT_PNG.replace(".png", f"{args.suffix}.png")
    out_txt = OUT_TXT.replace(".txt", f"{args.suffix}.txt")

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

    print("=" * 70)
    print("P2 — FedProx vs Vanilla vs Shapley-weighted FedAvg")
    print("=" * 70)
    print(f"clients={len(client_ids)}  rounds={n_rounds}  beta={beta:g}  "
          f"mu={mu:g}  seed={GLOBAL_SEED}")
    print(f"GTG truncation mode: {mode}  ->  eps_b={eps_b:g}, eps_i={eps_i:g}")
    print(f"Initial (round 0) accuracy: {acc0:.4f}")
    print()

    hist_v, hist_s, hist_p = [acc0], [acc0], [acc0]
    truncated_rounds = []
    fields = ["round", "acc_vanilla", "acc_shapley", "acc_fedprox"]

    def flush_csv():
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(len(hist_v)):
                w.writerow({"round": i, "acc_vanilla": hist_v[i],
                            "acc_shapley": hist_s[i], "acc_fedprox": hist_p[i]})

    t0 = time.time()
    for r in range(1, n_rounds + 1):
        t_round = time.time()

        # ---- (1) Vanilla FedAvg ----
        cs_v = train_all_clients(client_ids, data, state_v, r)
        state_v = fedavg([cs_v[c] for c in client_ids],
                         [client_sizes[c] for c in client_ids])
        acc_v = evaluate(state_v, X_test, y_test)
        hist_v.append(acc_v)

        # ---- (2) Shapley-weighted FedAvg (reproduces the main run) ----
        cs_s = train_all_clients(client_ids, data, state_s, r)
        provisional = fedavg([cs_s[c] for c in client_ids],
                             [client_sizes[c] for c in client_ids])
        phi, v0, vN = gtg_shapley_one_round(
            client_ids, cs_s, client_sizes,
            state_s, provisional, X_test, y_test,
            eps_b=eps_b, eps_i=eps_i)
        is_trunc = bool(np.allclose(
            np.array([phi[c] for c in client_ids]), 0.0))
        if is_trunc:
            truncated_rounds.append(r)
        weights = shapley_to_weights(phi, client_ids, client_sizes, beta=beta)
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

        elapsed = time.time() - t0
        eta = (elapsed / r) * (n_rounds - r)
        tag = "  [Shapley arm TRUNCATED -> size weights]" if is_trunc else ""
        print(f"Round {r:>3}/{n_rounds}  vanilla={acc_v:.4f}  "
              f"shapley={acc_s:.4f}  fedprox={acc_p:.4f}{tag}")
        print(f"           ({time.time() - t_round:.1f}s, ETA {fmt_eta(eta)})")

        flush_csv()

    elapsed = time.time() - t0
    n_trunc = len(truncated_rounds)

    # ---- plot ----
    rounds_axis = list(range(n_rounds + 1))
    markevery = max(1, n_rounds // 12)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for r in truncated_rounds:
        ax.axvspan(r - 0.5, r + 0.5, color="#d62728", alpha=0.06, linewidth=0)
    if truncated_rounds:
        ax.plot([], [], "s", color="#d62728", alpha=0.3, markersize=9,
                label=f"Shapley arm truncated ({n_trunc}/{n_rounds})")

    ax.plot(rounds_axis, hist_v, "o-", markevery=markevery, markersize=5,
            linewidth=1.8, color="#888888", label="Vanilla FedAvg")
    ax.plot(rounds_axis, hist_s, "s-", markevery=markevery, markersize=5,
            linewidth=1.8, color="#1f77b4",
            label=f"Shapley-weighted (softmax, beta={beta:g})")
    ax.plot(rounds_axis, hist_p, "^-", markevery=markevery, markersize=5,
            linewidth=1.8, color="#2ca02c", label=f"FedProx (mu={mu:g})")

    ax.set_xlabel("Global round")
    ax.set_ylabel("Global test accuracy")
    ax.set_title(f"Aggregation baselines — UCI HAR, {len(client_ids)} clients, "
                 f"seed {GLOBAL_SEED}\neps_b={eps_b:g} ({mode})", fontsize=11)
    ax.set_xticks(range(0, n_rounds + 1, max(1, n_rounds // 10)))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=8.5)

    if n_rounds >= 20:
        start = int(n_rounds * 2 / 3)
        axins = ax.inset_axes([0.44, 0.28, 0.34, 0.32])
        for h, col in [(hist_v, "#888888"), (hist_s, "#1f77b4"),
                       (hist_p, "#2ca02c")]:
            axins.plot(rounds_axis[start:], h[start:], "-", linewidth=1.4,
                       color=col)
        axins.set_title("plateau detail", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- summary ----
    pw = min(PLATEAU_WINDOW, n_rounds)
    pv, ps, pp = (float(np.mean(h[-pw:])) for h in (hist_v, hist_s, hist_p))

    lines = []
    lines.append("FedProx vs Vanilla vs Shapley-weighted FedAvg (UCI HAR)")
    lines.append("=" * 60)
    lines.append(f"clients={len(client_ids)}  rounds={n_rounds}  "
                 f"local_epochs={LOCAL_EPOCHS}  lr={LOCAL_LR}  "
                 f"beta={beta:g}  mu(FedProx)={mu:g}  seed={GLOBAL_SEED}")
    lines.append(f"GTG truncation: {mode}, eps_b={eps_b:g}, eps_i={eps_i:g}")
    lines.append(f"Runtime: {fmt_eta(elapsed)}")
    lines.append("")
    lines.append("Accuracy summary")
    lines.append("-" * 60)
    lines.append(f"  final round {n_rounds:<3d}     vanilla={hist_v[-1]:.4f}  "
                 f"shapley={hist_s[-1]:.4f}  fedprox={hist_p[-1]:.4f}")
    lines.append(f"  mean last {pw:<2d} rounds  vanilla={pv:.4f}  "
                 f"shapley={ps:.4f}  fedprox={pp:.4f}")
    lines.append(f"  best round         vanilla={max(hist_v):.4f}  "
                 f"shapley={max(hist_s):.4f}  fedprox={max(hist_p):.4f}")
    lines.append("")
    lines.append(f"  Shapley - FedProx (final): {hist_s[-1] - hist_p[-1]:+.4f}"
                 f"   (plateau: {ps - pp:+.4f})")
    lines.append(f"  Shapley - Vanilla (final): {hist_s[-1] - hist_v[-1]:+.4f}"
                 f"   (plateau: {ps - pv:+.4f})")
    lines.append(f"  FedProx - Vanilla (final): {hist_p[-1] - hist_v[-1]:+.4f}"
                 f"   (plateau: {pp - pv:+.4f})")
    lines.append("")
    lines.append("Shapley arm truncation")
    lines.append("-" * 60)
    lines.append(f"  truncated rounds: {n_trunc}/{n_rounds} "
                 f"({100.0 * n_trunc / n_rounds:.0f}%) — in these the Shapley "
                 f"arm ran plain size-weighted FedAvg.")
    lines.append(f"  Shapley-live rounds: {n_rounds - n_trunc}/{n_rounds}")
    if n_trunc > n_rounds / 2:
        lines.append("  WARNING: the Shapley arm was inactive for most of the "
                     "run, so its curve is")
        lines.append("  close to vanilla by construction. Re-run with "
                     "--adaptive for a comparison")
        lines.append("  in which the method is actually operating.")
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
                 "toward the global")
    lines.append("model); server aggregation stays size-weighted like vanilla. "
                 "It is a client-")
    lines.append("drift / stability baseline for non-IID data, complementary "
                 "to contribution-")
    lines.append("based reweighting. On the honest, mildly non-IID 30-subject "
                 "HAR partition")
    lines.append("(every client holds all 6 classes), FedProx is expected to "
                 "track vanilla")
    lines.append("closely; the proximal term matters more under stronger skew "
                 "— see")
    lines.append("dirichlet_alpha_sweep.py.")

    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print()
    print(text)
    print(f"\nWrote {out_csv}, {out_png}, {out_txt}")


if __name__ == "__main__":
    main()
