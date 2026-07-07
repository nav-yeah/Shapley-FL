"""
P2 — Adaptive Aggregation: Shapley-weighted FedAvg vs vanilla FedAvg.

WHAT THIS DOES
--------------
Runs TWO federated trainings side by side from the SAME initial model
(GLOBAL_SEED=42), same 30 HAR clients, same 5 rounds, same local-training
seeds, so results are directly comparable:

  (1) VANILLA FedAvg  — aggregation weights proportional to client data
      size (the standard McMahan et al. weighting; identical to
      fl_utils.run_federated_rounds).

  (2) SHAPLEY-WEIGHTED FedAvg — each round, per-round GTG-Shapley values
      are computed for that round's client updates (reusing P1's
      gtg_shapley_one_round from export_shapley_csv.py), and the
      aggregation weights are  w_i = softmax(BETA * phi_i).

WARM-UP PERIOD (from the original project plan)
-----------------------------------------------
WARMUP_ROUNDS controls how many initial rounds the weighted run uses
standard size-proportional FedAvg before switching to Shapley weighting
(the original plan's "first rounds use standard FedAvg until Shapley
scores stabilize"). Default is 0 because in this pipeline Shapley is
computed WITHIN each round before aggregating (there is no stale-score
instability to protect against), and with only 5 total rounds a long
warm-up would erase the method's effect. Set WARMUP_ROUNDS=1 (or more)
to reproduce the original plan's behavior; results with warm-up are
reported in the writeup's appendix table.

DESIGN NOTES (read before changing anything)
--------------------------------------------
* Per-round Shapley values are used, NOT end-of-training values — see
  SETUP_AND_HANDOFF.md Section 4. The weighted run computes its own
  per-round scores online each round, because once aggregation weights
  change, the model trajectory diverges from the one that produced the
  committed shapley_scores.csv, so those stored round-2..5 scores would
  no longer describe this run's updates.
* Within each round, Shapley is computed BEFORE the weighted aggregation:
  clients train -> provisional (size-weighted) aggregate defines this
  round's vN for GTG-Shapley -> phi values -> softmax weights -> the
  ACTUAL aggregate used to advance the global model.
* BETA (inverse softmax temperature) = 50.0. Per-round phi values live
  roughly in [-0.02, +0.02] (see experimentation_writeup.md Section 6),
  so BETA*phi spans about ±1: enough spread to matter, not
  winner-take-all. BETA=0 recovers uniform weighting.
* Negative-phi clients are NOT hard-excluded; softmax just gives them
  small weight. Every round has legitimate small negative values
  (transient, not malicious) — hard exclusion would be unstable.
* FALLBACK: rounds truncated by GTG-Shapley (|vN - v0| <= eps_b, all
  phi = 0) carry no contribution signal, so the weighted run falls back
  to standard size-proportional weights for those rounds.

CONVERGENCE-SPEED METRIC (from the original plan's "compare convergence")
-------------------------------------------------------------------------
Reports rounds-to-reach-target-accuracy for each variant at several
accuracy thresholds, alongside final accuracy.

OUTPUTS
-------
  aggregation_results.csv    — round, client_id, aggregation_weight,
                               global_accuracy_vanilla, global_accuracy_weighted
  fairness_metrics.txt       — ShapFed-style Pearson correlations +
                               convergence-speed table
  aggregation_comparison.png — accuracy-vs-round curves

Usage (from repo root, venv active, after preprocess_har.py):

    python shapley_weighted_fedavg.py

Runtime: ~1-2 min on CPU.
"""

import csv
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fl_utils import (
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR,
)
from export_shapley_csv import gtg_shapley_one_round

PKL_PATH = "preprocessed/har_clients.pkl"
OUT_CSV = "aggregation_results.csv"
OUT_TXT = "fairness_metrics.txt"
OUT_PNG = "aggregation_comparison.png"

GLOBAL_SEED = 42
N_ROUNDS = 5
BETA = 50.0        # inverse softmax temperature for Shapley -> weight mapping
WARMUP_ROUNDS = 0  # original plan's warm-up: rounds of plain FedAvg before
                   # Shapley weighting kicks in (0 = weight from round 1)

# Accuracy thresholds for the convergence-speed comparison
CONV_TARGETS = [0.50, 0.60, 0.70]


# ----------------------------------------------------------------------
# Reweighting function: per-round Shapley scores -> aggregation weights
# ----------------------------------------------------------------------
def shapley_to_weights(phi_dict, client_ids, client_sizes, beta=BETA):
    """Softmax over per-round Shapley values. Returns dict {cid: weight},
    weights sum to 1. Numerically stabilized (max-subtraction).

    FALLBACK: if this round was truncated by GTG-Shapley (|vN - v0| <=
    eps_b -> all phi exactly 0), there is no contribution signal, so we
    fall back to standard size-proportional FedAvg weights for this round
    rather than silently going uniform (uniform would over-weight tiny
    clients relative to vanilla FedAvg for no reason)."""
    phi = np.array([phi_dict[c] for c in client_ids], dtype=float)
    if np.allclose(phi, 0.0):
        total = float(sum(client_sizes[c] for c in client_ids))
        return {c: client_sizes[c] / total for c in client_ids}
    z = beta * phi
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    return {c: float(w_i) for c, w_i in zip(client_ids, w)}


def size_weights(client_ids, client_sizes):
    total = float(sum(client_sizes[c] for c in client_ids))
    return {c: client_sizes[c] / total for c in client_ids}


def train_all_clients(client_ids, data, global_state, round_idx, seed=GLOBAL_SEED):
    """One round of local training for every client, starting from
    global_state. Same seeding convention as fl_utils / export_shapley_csv
    so both runs use identical batch orderings."""
    client_states = {}
    for cid in client_ids:
        local_model = get_model()
        local_model.load_state_dict(global_state)
        client_states[cid] = local_train(
            local_model, data[cid]["X_train"], data[cid]["y_train"],
            epochs=LOCAL_EPOCHS, lr=LOCAL_LR,
            seed=seed + round_idx * 100 + cid,
        )
    return client_states


def rounds_to_target(acc_history, target):
    """First round index (1-based) at which accuracy >= target, or '-'."""
    for r, a in enumerate(acc_history):
        if a >= target:
            return r  # acc_history[0] is round 0 (init)
    return "-"


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    client_ids = sorted(data.keys())

    X_test = np.concatenate([data[c]["X_test"] for c in client_ids])
    y_test = np.concatenate([data[c]["y_test"] for c in client_ids])
    client_sizes = {c: len(data[c]["y_train"]) for c in client_ids}

    # Both runs start from the IDENTICAL initial model.
    init_state = clone_state_dict(get_model(seed=GLOBAL_SEED).state_dict())
    state_vanilla = clone_state_dict(init_state)
    state_weighted = clone_state_dict(init_state)

    acc_init = evaluate(init_state, X_test, y_test)
    print(f"Initial (round 0) accuracy: {acc_init:.4f}")
    if WARMUP_ROUNDS > 0:
        print(f"Warm-up enabled: first {WARMUP_ROUNDS} round(s) of the "
              f"weighted run use standard size-weighted FedAvg.")

    acc_vanilla_hist, acc_weighted_hist = [acc_init], [acc_init]
    rows = []
    cum_phi = {c: 0.0 for c in client_ids}
    cum_weight = {c: 0.0 for c in client_ids}

    t0 = time.time()
    for r in range(1, N_ROUNDS + 1):
        print(f"\nRound {r}/{N_ROUNDS}")

        # ---------- (1) VANILLA FedAvg branch ----------
        cs_v = train_all_clients(client_ids, data, state_vanilla, r)
        state_vanilla = fedavg(
            [cs_v[c] for c in client_ids],
            [client_sizes[c] for c in client_ids],
        )
        acc_v = evaluate(state_vanilla, X_test, y_test)
        acc_vanilla_hist.append(acc_v)

        # ---------- (2) SHAPLEY-WEIGHTED branch ----------
        cs_w = train_all_clients(client_ids, data, state_weighted, r)

        # Provisional size-weighted aggregate: defines this round's vN
        # for GTG-Shapley (the "what would this round achieve" reference).
        provisional = fedavg(
            [cs_w[c] for c in client_ids],
            [client_sizes[c] for c in client_ids],
        )
        phi, v0, vN = gtg_shapley_one_round(
            client_ids, cs_w, client_sizes,
            state_weighted, provisional, X_test, y_test,
        )

        if r <= WARMUP_ROUNDS:
            # Original plan's warm-up: plain FedAvg while scores stabilize.
            weights = size_weights(client_ids, client_sizes)
            state_weighted = provisional  # provisional IS the size-weighted aggregate
        else:
            weights = shapley_to_weights(phi, client_ids, client_sizes)
            state_weighted = fedavg(
                [cs_w[c] for c in client_ids],
                [weights[c] for c in client_ids],
            )
        acc_w = evaluate(state_weighted, X_test, y_test)
        acc_weighted_hist.append(acc_w)

        tag = " [warm-up]" if r <= WARMUP_ROUNDS else ""
        print(f"  vanilla acc={acc_v:.4f} | weighted acc={acc_w:.4f}{tag} "
              f"(phi range [{min(phi.values()):+.4f}, {max(phi.values()):+.4f}], "
              f"weight range [{min(weights.values()):.4f}, {max(weights.values()):.4f}])")

        for c in client_ids:
            cum_phi[c] += phi[c]
            cum_weight[c] += weights[c]
            rows.append({
                "round": r,
                "client_id": c,
                "aggregation_weight": weights[c],
                "global_accuracy_vanilla": acc_v,
                "global_accuracy_weighted": acc_w,
            })

    elapsed = time.time() - t0

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "round", "client_id", "aggregation_weight",
            "global_accuracy_vanilla", "global_accuracy_weighted",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}  ({elapsed:.1f}s total)")

    # ------------------------------------------------------------------
    # Convergence-speed comparison (original plan: "compare convergence
    # speed and final accuracy")
    # ------------------------------------------------------------------
    conv_lines = ["", "Convergence speed (first round reaching target accuracy):",
                  "  target   vanilla   weighted"]
    for t in CONV_TARGETS:
        rv = rounds_to_target(acc_vanilla_hist, t)
        rw = rounds_to_target(acc_weighted_hist, t)
        conv_lines.append(f"  >={t:.2f}    {str(rv):>5}     {str(rw):>5}")

    # ------------------------------------------------------------------
    # Optional: ShapFed-style fairness metric.
    # Standalone accuracy = each client trains ALONE from the same init
    # for an equivalent budget (N_ROUNDS * LOCAL_EPOCHS epochs), evaluated
    # on the shared global test set. Fairness = Pearson correlation
    # between standalone accuracy and the credit signal each scheme
    # assigns. Higher correlation = credit assignment tracks genuine data
    # usefulness more faithfully.
    # ------------------------------------------------------------------
    print("\nComputing standalone client accuracies for fairness metric...")
    standalone_acc = {}
    for cid in client_ids:
        m = get_model(seed=GLOBAL_SEED)
        local_train(m, data[cid]["X_train"], data[cid]["y_train"],
                    epochs=N_ROUNDS * LOCAL_EPOCHS, lr=LOCAL_LR,
                    seed=GLOBAL_SEED + cid)
        standalone_acc[cid] = evaluate(m.state_dict(), X_test, y_test)

    sa = np.array([standalone_acc[c] for c in client_ids])
    ph = np.array([cum_phi[c] for c in client_ids])
    wt = np.array([cum_weight[c] for c in client_ids])
    sz = np.array([client_sizes[c] for c in client_ids], dtype=float)

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    lines = [
        "ShapFed-style fairness metric (Pearson correlation between",
        "standalone client accuracy on the global test set and the",
        "credit signal each scheme assigns):",
        "",
        f"  vanilla FedAvg (credit = data size):        r = {pearson(sa, sz):+.4f}",
        f"  per-round Shapley values (cumulative phi):  r = {pearson(sa, ph):+.4f}",
        f"  Shapley-softmax aggregation weights:        r = {pearson(sa, wt):+.4f}",
        "",
        f"Config: BETA={BETA:g}, WARMUP_ROUNDS={WARMUP_ROUNDS}, "
        f"N_ROUNDS={N_ROUNDS}, seed={GLOBAL_SEED}",
        f"Final accuracy — vanilla: {acc_vanilla_hist[-1]:.4f}, "
        f"weighted: {acc_weighted_hist[-1]:.4f}",
        f"Per-round accuracy (vanilla):  "
        + ", ".join(f"{a:.4f}" for a in acc_vanilla_hist),
        f"Per-round accuracy (weighted): "
        + ", ".join(f"{a:.4f}" for a in acc_weighted_hist),
    ] + conv_lines
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    rounds_axis = list(range(N_ROUNDS + 1))
    label_w = f"Shapley-weighted (softmax, beta={BETA:g}"
    label_w += f", warm-up={WARMUP_ROUNDS})" if WARMUP_ROUNDS else ")"
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds_axis, acc_vanilla_hist, "o-", label="Vanilla FedAvg (size-weighted)")
    plt.plot(rounds_axis, acc_weighted_hist, "s-", label=label_w)
    plt.xlabel("Global round")
    plt.ylabel("Global test accuracy")
    plt.title("Vanilla vs Shapley-weighted FedAvg — UCI HAR, 30 clients, seed 42")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
