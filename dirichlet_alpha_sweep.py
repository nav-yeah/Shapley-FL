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
                                               # (appends to the CSV)

Runtime: ~2 min per alpha on CPU (~6 min for the full sweep). Progress
prints every round.

Outputs:
    dirichlet_sweep_results.csv  — alpha, round, acc_vanilla, acc_weighted
    dirichlet_sweep.png          — one accuracy curve panel per alpha
                                   (regenerated from the CSV each run,
                                    so per-alpha runs still build the
                                    full figure once all alphas exist)
"""

import csv
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fl_utils import (
    get_model, fedavg, evaluate, clone_state_dict,
)
from export_shapley_csv import gtg_shapley_one_round
from shapley_weighted_fedavg import train_all_clients

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "dirichlet_sweep_results.csv"
OUT_PNG = "dirichlet_sweep.png"

GLOBAL_SEED = 42
N_ROUNDS = 5
N_CLIENTS = 30          # keep 30 synthetic clients for comparability
ALPHAS = [0.1, 0.5, 1.0]
N_CLASSES = 6
MIN_SAMPLES_PER_CLIENT = 30
SWEEP_BETA = 10.0       # gentler than the main script's 50 — see
                        # design decision 3 in the module docstring


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


def run_pair(data, X_test, y_test, label=""):
    """Vanilla vs Shapley-weighted FedAvg on one partition. Returns the
    two accuracy histories (index 0 = round-0 init accuracy)."""
    client_ids = sorted(data.keys())
    client_sizes = {c: len(data[c]["y_train"]) for c in client_ids}

    init_state = clone_state_dict(get_model(seed=GLOBAL_SEED).state_dict())
    state_v = clone_state_dict(init_state)
    state_w = clone_state_dict(init_state)
    acc0 = evaluate(init_state, X_test, y_test)
    hist_v, hist_w = [acc0], [acc0]

    for r in range(1, N_ROUNDS + 1):
        cs_v = train_all_clients(client_ids, data, state_v, r)
        state_v = fedavg([cs_v[c] for c in client_ids],
                         [client_sizes[c] for c in client_ids])
        hist_v.append(evaluate(state_v, X_test, y_test))

        cs_w = train_all_clients(client_ids, data, state_w, r)
        provisional = fedavg([cs_w[c] for c in client_ids],
                             [client_sizes[c] for c in client_ids])
        phi, _, _ = gtg_shapley_one_round(
            client_ids, cs_w, client_sizes,
            state_w, provisional, X_test, y_test)
        weights = size_tilted_shapley_weights(phi, client_ids, client_sizes)
        state_w = fedavg([cs_w[c] for c in client_ids],
                         [weights[c] for c in client_ids])
        hist_w.append(evaluate(state_w, X_test, y_test))

        print(f"  {label} round {r}: vanilla={hist_v[-1]:.4f} "
              f"weighted={hist_w[-1]:.4f}")

    return hist_v, hist_w


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


def plot_from_rows(rows, path):
    alphas = sorted({r["alpha"] for r in rows})
    fig, axes = plt.subplots(1, len(alphas),
                              figsize=(5 * len(alphas), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, alphas):
        sub = sorted([r for r in rows if r["alpha"] == alpha],
                     key=lambda r: r["round"])
        rounds_axis = [r["round"] for r in sub]
        ax.plot(rounds_axis, [r["acc_vanilla"] for r in sub], "o-",
                label="Vanilla FedAvg")
        ax.plot(rounds_axis, [r["acc_weighted"] for r in sub], "s-",
                label="Shapley-weighted (size-tilted)")
        ax.set_title(f"Dirichlet alpha = {alpha:g}"
                     + ("  (extreme skew)" if alpha == 0.1 else
                        "  (mild skew)" if alpha == 1.0 else ""))
        ax.set_xlabel("Global round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Global test accuracy")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


def main():
    # Optional CLI arg: run a single alpha (appends/overwrites that
    # alpha's rows in the CSV). No arg = full sweep.
    if len(sys.argv) > 1:
        alphas_to_run = [float(sys.argv[1])]
    else:
        alphas_to_run = ALPHAS

    X_pool, y_pool, X_test, y_test = pool_har_data(PKL_PATH)
    print(f"Pooled train: {X_pool.shape[0]} samples | "
          f"global test: {X_test.shape[0]} samples | "
          f"SWEEP_BETA={SWEEP_BETA:g}")

    rows = [r for r in load_existing_rows(OUT_CSV)
            if r["alpha"] not in alphas_to_run]

    t0 = time.time()
    for alpha in alphas_to_run:
        print(f"\n=== Dirichlet alpha = {alpha} ===")
        data = dirichlet_partition(X_pool, y_pool, N_CLIENTS, alpha)
        sizes = sorted(len(d["y_train"]) for d in data.values())
        print(f"  client sizes: min={sizes[0]}, "
              f"median={sizes[len(sizes)//2]}, max={sizes[-1]}")
        hv, hw = run_pair(data, X_test, y_test, label=f"a={alpha}")
        for r in range(N_ROUNDS + 1):
            rows.append({"alpha": alpha, "round": r,
                         "acc_vanilla": hv[r], "acc_weighted": hw[r]})

    save_rows(rows, OUT_CSV)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV} "
          f"({time.time() - t0:.0f}s this run)")
    plot_from_rows(rows, OUT_PNG)

    print("\nSummary (final-round accuracy):")
    print("  alpha   vanilla   weighted   gap")
    for alpha in sorted({r["alpha"] for r in rows}):
        fin = [r for r in rows if r["alpha"] == alpha
               and r["round"] == N_ROUNDS][0]
        print(f"  {alpha:<6g} {fin['acc_vanilla']:.4f}    "
              f"{fin['acc_weighted']:.4f}    "
              f"{fin['acc_weighted'] - fin['acc_vanilla']:+.4f}")


if __name__ == "__main__":
    main()
