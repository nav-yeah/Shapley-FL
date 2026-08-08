"""
P2 (extension) — Non-IID severity sweep: Dirichlet alpha experiment.

This implements "Experiment 2" from the original project plan:
    Vary Dirichlet alpha (0.1 = extreme non-IID label skew, 1.0 = mild).
    Compare Shapley-weighted aggregation against vanilla FedAvg as data
    becomes more skewed.

WHY THIS EXISTS SEPARATELY from shapley_weighted_fedavg.py:
The main P2 deliverable uses the repo's natural subject-based partition
(30 clients = 30 HAR subjects), which is non-IID by size/distribution
shift but every client has all 6 classes (see SETUP_AND_HANDOFF.md §5).
This script instead RE-partitions the pooled HAR data synthetically with
a Dirichlet(alpha) label-skew split — the standard protocol from the FL
literature — so non-IID severity can be dialed up and down.

50-ROUND UPDATE
---------------
N_ROUNDS now follows P1's 50-round pipeline. Three consequences:

1. RUNTIME. Three alphas x 50 rounds x 2 arms, with GTG-Shapley every
   round, is roughly 30-60 min on CPU. Results are written to the CSV
   after EACH alpha finishes, and the figure is regenerated from the CSV
   each time, so the sweep can be run one alpha at a time across
   sessions:  python dirichlet_alpha_sweep.py 0.1   then  0.5   then 1.0.

2. TRUNCATION. GTG-Shapley returns all-zero scores once the per-round
   accuracy gain falls below eps_b, at which point the weighted arm
   falls back to plain size weights. On the natural partition this
   silences 76% of rounds at 50 rounds (see shapley_signal_decay.py).
   Under label skew the accuracy trajectory is noisier, so it silences
   fewer — but the count is now tracked and reported per alpha, because
   "weighted matches vanilla" means something completely different when
   the weighting was switched off for most of the run. Use --adaptive to
   lower the threshold.

3. THE α=0.1 STORY GETS LONGER TO TELL. The 5-round run showed a ~10
   point accuracy drop under extreme skew. Over 50 rounds the question
   becomes whether that gap closes, persists, or diverges — which is a
   stronger result either way than a single endpoint. The summary
   therefore reports the final round, the mean over the last 10 rounds,
   and the best round for each arm.

THREE DESIGN DECISIONS SPECIFIC TO EXTREME SKEW (all learned the hard
way — the first version of this script produced training collapses):

1. PARTITION REPAIR. At alpha=0.1 a plain Dirichlet split routinely
   leaves some client with only a handful of samples (re-drawing doesn't
   help; with 30 clients it is nearly impossible for every client to
   clear a minimum). We repair deterministically instead: transfer
   random samples from the largest client to any client below
   MIN_SAMPLES_PER_CLIENT until all clients clear it.

2. SIZE-AWARE SHAPLEY TILT. The main script's pure softmax(beta*phi)
   is fine on the natural partition, where client sizes are nearly
   equal (224-327 samples) and size is therefore an irrelevant factor.
   Under Dirichlet skew, sizes vary by up to ~40x, and pure softmax
   hands a near-empty client a full 1/30-ish weight, poisoning the
   average with an under-trained update. Here the weighted variant uses
        w_i  proportional to  n_i * exp(beta * phi_i)
   i.e. standard FedAvg size weighting multiplicatively tilted by the
   Shapley signal. On near-equal sizes this reduces to the main
   script's softmax, so the two scripts are methodologically
   consistent. beta=0 recovers vanilla FedAvg exactly.

3. GENTLER TEMPERATURE UNDER SKEW (SWEEP_BETA = 10, not 50). Per-round
   phi magnitudes are larger under label skew than on the natural
   partition (roughly +-0.05 vs +-0.02), so the main script's beta=50
   over-concentrates weight, skews the global model toward the dominant
   clients' class mix, and collapses training (observed at alpha=0.5:
   weighted fell 0.71 -> 0.54 over rounds 2-5 with beta=50; stable and
   competitive with beta=10). Making the temperature adaptive to the
   per-round phi scale — or correcting phi for client heterogeneity
   before weighting — is the natural next step, and is exactly the
   motivation for a heterogeneity-corrected Shapley variant.

Usage (from repo root, venv active, after preprocess_har.py):

    python dirichlet_alpha_sweep.py            # full sweep (all alphas)
    python dirichlet_alpha_sweep.py 0.5        # one alpha only
                                               # (merged into the CSV)
    python dirichlet_alpha_sweep.py --rounds 3       # quick smoke test
    python dirichlet_alpha_sweep.py --adaptive       # lower eps_b

Outputs:
    dirichlet_sweep_results.csv  — alpha, round, acc_vanilla, acc_weighted
    dirichlet_sweep_rounds.csv   — alpha, round, truncated, phi range
    dirichlet_sweep.png          — one accuracy curve panel per alpha
                                   (regenerated from the CSV each run,
                                    so per-alpha runs still build the
                                    full figure once all alphas exist)
    dirichlet_sweep.txt          — per-alpha summary table
"""

import argparse
import csv
import os
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fl_utils import (
    get_model, fedavg, evaluate, clone_state_dict,
)
from export_shapley_csv import gtg_shapley_one_round
from shapley_weighted_fedavg import (
    train_all_clients, fmt_eta,
    EPS_B_FIXED, EPS_I_FIXED, EPS_B_FLOOR, EPS_I_RATIO,
)

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "dirichlet_sweep_results.csv"
OUT_ROUND_CSV = "dirichlet_sweep_rounds.csv"
OUT_PNG = "dirichlet_sweep.png"
OUT_TXT = "dirichlet_sweep.txt"

GLOBAL_SEED = 42
N_ROUNDS = 50           # follows P1's 50-round pipeline (was 5)
N_CLIENTS = 30          # keep 30 synthetic clients for comparability
ALPHAS = [0.1, 0.5, 1.0]
N_CLASSES = 6
MIN_SAMPLES_PER_CLIENT = 30
SWEEP_BETA = 10.0       # gentler than the main script's 50 — see
                        # design decision 3 in the module docstring
PLATEAU_WINDOW = 10


def pool_har_data(pkl_path):
    """Pool ALL clients' train data into one big (X, y); keep the global
    test set (concatenated per-client test splits) as the fixed
    evaluation set across every alpha, so accuracy numbers are
    comparable between partitions."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    cids = sorted(data.keys())
    X_pool = np.concatenate([data[c]["X_train"] for c in cids])
    y_pool = np.concatenate([data[c]["y_train"] for c in cids])
    X_test = np.concatenate([data[c]["X_test"] for c in cids])
    y_test = np.concatenate([data[c]["y_test"] for c in cids])
    return X_pool, y_pool, X_test, y_test


def dirichlet_partition(X, y, n_clients, alpha, seed=GLOBAL_SEED,
                        min_size=MIN_SAMPLES_PER_CLIENT):
    """Standard Dirichlet label-skew partition (per-class proportions
    drawn from Dir(alpha) across clients), followed by a deterministic
    REPAIR step: any client below min_size receives random samples
    transferred from the currently largest client, until every client
    clears min_size. Repair preserves the overall skew while ensuring
    every client can actually train."""
    rng = np.random.RandomState(seed)
    idx_by_client = [[] for _ in range(n_clients)]
    for c in range(N_CLASSES):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(cls_idx)).astype(int)[:-1]
        for client_i, part in enumerate(np.split(cls_idx, cuts)):
            idx_by_client[client_i].extend(part.tolist())

    # ---- repair: top up under-sized clients from the largest client ----
    n_transferred = 0
    while True:
        sizes = [len(ix) for ix in idx_by_client]
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_size:
            break
        largest = int(np.argmax(sizes))
        need = min_size - sizes[smallest]
        take = min(need, sizes[largest] - min_size)
        take_pos = rng.choice(len(idx_by_client[largest]), size=take,
                              replace=False)
        take_set = set(int(t) for t in take_pos)
        moved = [s for j, s in enumerate(idx_by_client[largest])
                 if j in take_set]
        idx_by_client[largest] = [s for j, s
                                  in enumerate(idx_by_client[largest])
                                  if j not in take_set]
        idx_by_client[smallest].extend(moved)
        n_transferred += take
    if n_transferred:
        print(f"  partition repair: transferred {n_transferred} samples "
              f"to enforce min {min_size}/client")

    data = {}
    for i, ix in enumerate(idx_by_client, start=1):
        ix = np.array(ix)
        rng.shuffle(ix)
        data[i] = {"X_train": X[ix].astype(np.float32),
                   "y_train": y[ix].astype(np.int64)}
    return data


def size_tilted_shapley_weights(phi_dict, client_ids, client_sizes,
                                beta=SWEEP_BETA):
    """w_i proportional to n_i * exp(beta * phi_i).

    Size-aware version of the main script's softmax reweighting —
    required under Dirichlet skew where client sizes vary widely (see
    module docstring, design decision 2). Falls back to plain size
    weights on GTG-truncated rounds (all phi = 0), matching the main
    script's fallback behavior."""
    phi = np.array([phi_dict[c] for c in client_ids], dtype=float)
    sizes = np.array([client_sizes[c] for c in client_ids], dtype=float)
    if np.allclose(phi, 0.0):
        w = sizes / sizes.sum()
    else:
        z = beta * phi
        z = z - z.max()
        w = sizes * np.exp(z)
        w = w / w.sum()
    return {c: float(w_i) for c, w_i in zip(client_ids, w)}


def run_pair(data, X_test, y_test, n_rounds, beta, eps_b, eps_i, label=""):
    """Vanilla vs Shapley-weighted FedAvg on one partition. Returns
    (hist_vanilla, hist_weighted, round_records). Index 0 of each history
    is the round-0 init accuracy."""
    client_ids = sorted(data.keys())
    client_sizes = {c: len(data[c]["y_train"]) for c in client_ids}

    init_state = clone_state_dict(get_model(seed=GLOBAL_SEED).state_dict())
    state_v = clone_state_dict(init_state)
    state_w = clone_state_dict(init_state)
    acc0 = evaluate(init_state, X_test, y_test)
    hist_v, hist_w = [acc0], [acc0]
    records = []

    t0 = time.time()
    for r in range(1, n_rounds + 1):
        cs_v = train_all_clients(client_ids, data, state_v, r)
        state_v = fedavg([cs_v[c] for c in client_ids],
                         [client_sizes[c] for c in client_ids])
        hist_v.append(evaluate(state_v, X_test, y_test))

        cs_w = train_all_clients(client_ids, data, state_w, r)
        provisional = fedavg([cs_w[c] for c in client_ids],
                             [client_sizes[c] for c in client_ids])
        phi, v0, vN = gtg_shapley_one_round(
            client_ids, cs_w, client_sizes,
            state_w, provisional, X_test, y_test,
            eps_b=eps_b, eps_i=eps_i)
        phi_vals = np.array([phi[c] for c in client_ids])
        is_trunc = bool(np.allclose(phi_vals, 0.0))

        weights = size_tilted_shapley_weights(phi, client_ids, client_sizes,
                                              beta=beta)
        state_w = fedavg([cs_w[c] for c in client_ids],
                         [weights[c] for c in client_ids])
        hist_w.append(evaluate(state_w, X_test, y_test))

        records.append({
            "round": r,
            "truncated": int(is_trunc),
            "gain": vN - v0,
            "phi_min": float(phi_vals.min()),
            "phi_max": float(phi_vals.max()),
        })

        elapsed = time.time() - t0
        eta = (elapsed / r) * (n_rounds - r)
        tag = "  [truncated]" if is_trunc else ""
        print(f"  {label} round {r:>3}/{n_rounds}: vanilla={hist_v[-1]:.4f} "
              f"weighted={hist_w[-1]:.4f}{tag}  (ETA {fmt_eta(eta)})")

    return hist_v, hist_w, records


def load_existing_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [{"alpha": float(r["alpha"]), "round": int(r["round"]),
                 "acc_vanilla": float(r["acc_vanilla"]),
                 "acc_weighted": float(r["acc_weighted"])}
                for r in csv.DictReader(f)]


def save_rows(rows, path):
    rows = sorted(rows, key=lambda r: (r["alpha"], r["round"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "round",
                                          "acc_vanilla", "acc_weighted"])
        w.writeheader()
        w.writerows(rows)


def load_existing_round_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [{"alpha": float(r["alpha"]), "round": int(r["round"]),
                 "truncated": int(r["truncated"]), "gain": float(r["gain"]),
                 "phi_min": float(r["phi_min"]), "phi_max": float(r["phi_max"])}
                for r in csv.DictReader(f)]


def save_round_rows(rows, path):
    rows = sorted(rows, key=lambda r: (r["alpha"], r["round"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "round", "truncated",
                                          "gain", "phi_min", "phi_max"])
        w.writeheader()
        w.writerows(rows)


def plot_from_rows(rows, round_rows, path):
    alphas = sorted({r["alpha"] for r in rows})
    fig, axes = plt.subplots(1, len(alphas),
                             figsize=(5.2 * len(alphas), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, alphas):
        sub = sorted([r for r in rows if r["alpha"] == alpha],
                     key=lambda r: r["round"])
        rounds_axis = [r["round"] for r in sub]
        n_rounds = max(rounds_axis) if rounds_axis else 0
        markevery = max(1, len(rounds_axis) // 10)

        trunc = sorted(r["round"] for r in round_rows
                       if r["alpha"] == alpha and r["truncated"] == 1)
        for r in trunc:
            ax.axvspan(r - 0.5, r + 0.5, color="#d62728", alpha=0.06,
                       linewidth=0)
        if trunc:
            ax.plot([], [], "s", color="#d62728", alpha=0.3, markersize=8,
                    label=f"truncated ({len(trunc)}/{n_rounds})")

        ax.plot(rounds_axis, [r["acc_vanilla"] for r in sub], "o-",
                markevery=markevery, markersize=4, linewidth=1.6,
                color="#888888", label="Vanilla FedAvg")
        ax.plot(rounds_axis, [r["acc_weighted"] for r in sub], "s-",
                markevery=markevery, markersize=4, linewidth=1.6,
                color="#1f77b4", label="Shapley-weighted (size-tilted)")
        ax.set_title(f"Dirichlet alpha = {alpha:g}"
                     + ("  (extreme skew)" if alpha == 0.1 else
                        "  (mild skew)" if alpha == 1.0 else ""))
        ax.set_xlabel("Global round")
        ax.set_xticks(range(0, n_rounds + 1, max(1, n_rounds // 10)))
        ax.grid(alpha=0.3, linestyle="--")
    axes[0].set_ylabel("Global test accuracy")
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot to {path}")


def write_summary(rows, round_rows, path, beta, eps_b, mode):
    alphas = sorted({r["alpha"] for r in rows})
    lines = []
    lines.append("Dirichlet alpha sweep — non-IID severity")
    lines.append("=" * 60)
    lines.append(f"clients={N_CLIENTS}  beta={beta:g}  seed={GLOBAL_SEED}")
    lines.append(f"GTG truncation: {mode}, eps_b={eps_b:g}")
    lines.append("")
    lines.append("  alpha   n_rounds  vanilla_final  weighted_final  gap_final"
                 "   gap_plateau  truncated")
    for alpha in alphas:
        sub = sorted([r for r in rows if r["alpha"] == alpha],
                     key=lambda r: r["round"])
        n_rounds = sub[-1]["round"]
        pw = min(PLATEAU_WINDOW, n_rounds)
        v = [r["acc_vanilla"] for r in sub]
        w = [r["acc_weighted"] for r in sub]
        gap_f = w[-1] - v[-1]
        gap_p = float(np.mean(w[-pw:]) - np.mean(v[-pw:]))
        n_tr = sum(1 for r in round_rows
                   if r["alpha"] == alpha and r["truncated"] == 1)
        lines.append(f"  {alpha:<6g}  {n_rounds:>8d}  {v[-1]:>13.4f}  "
                     f"{w[-1]:>14.4f}  {gap_f:>+9.4f}  {gap_p:>+11.4f}  "
                     f"{n_tr:>4d}/{n_rounds}")
    lines.append("")
    lines.append("gap_plateau = mean(weighted) - mean(vanilla) over the last "
                 f"{PLATEAU_WINDOW} rounds.")
    lines.append("It is the number to quote: after convergence a single final "
                 "round is noisy")
    lines.append("enough to flip the sign of the gap.")
    lines.append("")
    lines.append("Reading the truncated column: in a truncated round the "
                 "weighted arm ran plain")
    lines.append("size-proportional FedAvg, so a small gap at high truncation "
                 "counts means the")
    lines.append("method was mostly switched off, not that it made no "
                 "difference.")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print()
    print(text)


def main():
    ap = argparse.ArgumentParser(
        description="Dirichlet non-IID severity sweep for P2 aggregation")
    ap.add_argument("alpha", nargs="?", type=float, default=None,
                    help="run a single alpha (default: full sweep)")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS,
                    help=f"number of FL rounds (default {N_ROUNDS})")
    ap.add_argument("--beta", type=float, default=SWEEP_BETA,
                    help=f"size-tilted softmax temperature "
                         f"(default {SWEEP_BETA:g})")
    ap.add_argument("--adaptive", action="store_true",
                    help=f"use eps_b={EPS_B_FLOOR:g} instead of "
                         f"{EPS_B_FIXED:g}")
    args = ap.parse_args()

    n_rounds = args.rounds
    beta = args.beta
    alphas_to_run = [args.alpha] if args.alpha is not None else ALPHAS

    if args.adaptive:
        eps_b, eps_i = EPS_B_FLOOR, EPS_B_FLOOR * EPS_I_RATIO
        mode = "adaptive"
    else:
        eps_b, eps_i = EPS_B_FIXED, EPS_I_FIXED
        mode = "fixed (P1 defaults)"

    X_pool, y_pool, X_test, y_test = pool_har_data(PKL_PATH)
    print(f"Pooled train: {X_pool.shape[0]} samples | "
          f"global test: {X_test.shape[0]} samples | beta={beta:g}")
    print(f"GTG truncation: {mode}  ->  eps_b={eps_b:g}, eps_i={eps_i:g}")
    print(f"Running alphas {alphas_to_run} for {n_rounds} rounds each.")
    print()

    rows = [r for r in load_existing_rows(OUT_CSV)
            if r["alpha"] not in alphas_to_run]
    round_rows = [r for r in load_existing_round_rows(OUT_ROUND_CSV)
                  if r["alpha"] not in alphas_to_run]

    t0 = time.time()
    for alpha in alphas_to_run:
        print(f"\n=== Dirichlet alpha = {alpha} ===")
        data = dirichlet_partition(X_pool, y_pool, N_CLIENTS, alpha)
        sizes = sorted(len(d["y_train"]) for d in data.values())
        print(f"  client sizes: min={sizes[0]}, "
              f"median={sizes[len(sizes) // 2]}, max={sizes[-1]}")
        hv, hw, recs = run_pair(data, X_test, y_test, n_rounds, beta,
                                eps_b, eps_i, label=f"a={alpha}")
        for r in range(n_rounds + 1):
            rows.append({"alpha": alpha, "round": r,
                         "acc_vanilla": hv[r], "acc_weighted": hw[r]})
        for rec in recs:
            round_rows.append({"alpha": alpha, **rec})

        # Save after EACH alpha so a long sweep can be interrupted.
        save_rows(rows, OUT_CSV)
        save_round_rows(round_rows, OUT_ROUND_CSV)
        print(f"  alpha {alpha} done — results saved to {OUT_CSV}")

    print(f"\nSaved {len(rows)} rows to {OUT_CSV} "
          f"({fmt_eta(time.time() - t0)} this run)")
    plot_from_rows(rows, round_rows, OUT_PNG)
    write_summary(rows, round_rows, OUT_TXT, beta, eps_b, mode)


if __name__ == "__main__":
    main()
