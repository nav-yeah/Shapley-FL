"""
P2 — Adaptive Aggregation: Shapley-weighted FedAvg vs vanilla FedAvg.

WHAT THIS DOES
--------------
Runs TWO federated trainings side by side from the SAME initial model
(GLOBAL_SEED=42), same 30 HAR clients, same N_ROUNDS rounds, same
local-training seeds, so results are directly comparable:

  (1) VANILLA FedAvg  — aggregation weights proportional to client data
      size (the standard McMahan et al. weighting; identical to
      fl_utils.run_federated_rounds).

  (2) SHAPLEY-WEIGHTED FedAvg — each round, per-round GTG-Shapley values
      are computed for that round's client updates (reusing P1's
      gtg_shapley_one_round from export_shapley_csv.py), and the
      aggregation weights are  w_i = softmax(BETA * phi_i).

50-ROUND UPDATE — THE SIGNAL-DECAY PROBLEM (READ THIS FIRST)
-------------------------------------------------------------
P1's pipeline moved from 5 to 50 rounds. That change breaks a hidden
assumption in GTG-Shapley's truncation rule, and P2 has to handle it
explicitly.

GTG-Shapley skips a round entirely (returns all phi = 0) when

    |vN - v0| <= eps_b        with P1's eps_b = 0.02

i.e. when the round's global accuracy gain is under 2 percentage points,
GTG treats the round as carrying no contribution signal. At 5 rounds the
model was still gaining 2-9 points per round, so this never fired. At 50
rounds the model converges around round 20 and per-round gains fall to
0.1-0.5 points, so the rule fires constantly.

Measured directly on P1's committed shapley_scores.csv (1500 rows):

    38 of 50 rounds are all-zero. Every round from 23 onward is
    truncated. Only 12 rounds (24%) carry any Shapley signal at all.

Consequence for P2: with eps_b = 0.02 the weighted run falls back to
size-proportional weights in 76% of rounds, so "Shapley-weighted FedAvg"
is really "vanilla FedAvg with 12 reweighted rounds" and the two curves
land on top of each other. That is a property of the THRESHOLD, not of
the method — and reporting it as if it were a result about Shapley
weighting would be wrong.

So this script supports two truncation regimes, selected on the command
line, and reports the truncation statistics either way:

  --fixed     (default) eps_b = 0.02, eps_i = 0.005  — P1's values.
              Faithful reproduction of the committed pipeline. Use this
              for the headline "as-specified" numbers and to show the
              signal-decay effect honestly.

  --adaptive  eps_b = EPS_B_FLOOR (0.0035), eps_i = EPS_B_FLOOR / 4.
              Keeps Shapley live deep into training. Justification: the
              threshold exists to separate a real accuracy gain from
              evaluation noise, but v0 and vN are evaluated on the SAME
              fixed test set with two highly correlated models, so the
              relevant noise floor is the paired-disagreement scale, not
              the independent-sample standard error. On the HAR test set
              (~2947 samples) eps_b = 0.0035 corresponds to about 10
              samples changing their prediction — comfortably above
              single-sample resolution (0.00034) and far below the
              late-round gains the fixed threshold was discarding.
              Raises live rounds from 12/50 to about 30/50.

Neither regime touches P1's files: gtg_shapley_one_round already accepts
eps_b and eps_i as arguments, so P2 passes its own values.

WARM-UP PERIOD (from the original project plan)
-----------------------------------------------
WARMUP_ROUNDS controls how many initial rounds the weighted run uses
standard size-proportional FedAvg before switching to Shapley weighting
(the original plan's "first rounds use standard FedAvg until Shapley
scores stabilize"). Default is 0 because in this pipeline Shapley is
computed WITHIN each round before aggregating (there is no stale-score
instability to protect against). At 50 rounds a short warm-up is cheap
to afford if the writeup wants it; set WARMUP_ROUNDS=1 or use --warmup N.

DESIGN NOTES (read before changing anything)
--------------------------------------------
* Per-round Shapley values are used, NOT end-of-training values — see
  SETUP_AND_HANDOFF.md Section 4. The weighted run computes its own
  per-round scores online each round, because once aggregation weights
  change, the model trajectory diverges from the one that produced the
  committed shapley_scores.csv, so those stored scores would no longer
  describe this run's updates. Round 1 is the exception: both runs start
  from the identical init and use the identical size-weighted
  provisional aggregate, so round-1 phi values MUST match P1's CSV.
  That is checked automatically at startup (see verify_round1_against_p1).
* Within each round, Shapley is computed BEFORE the weighted aggregation:
  clients train -> provisional (size-weighted) aggregate defines this
  round's vN for GTG-Shapley -> phi values -> softmax weights -> the
  ACTUAL aggregate used to advance the global model.
* BETA (inverse softmax temperature) = 50.0. Per-round phi values live
  roughly in [-0.02, +0.02] (see experimentation_writeup.md Section 6),
  so BETA*phi spans about +-1: enough spread to matter, not
  winner-take-all. BETA=0 recovers uniform weighting.
* Negative-phi clients are NOT hard-excluded; softmax just gives them
  small weight. Every round has legitimate small negative values
  (transient, not malicious) — hard exclusion would be unstable.
* FALLBACK: rounds truncated by GTG-Shapley (all phi = 0) carry no
  contribution signal, so the weighted run falls back to standard
  size-proportional weights for those rounds. Which rounds those were is
  logged explicitly to aggregation_round_log.csv (P3 needs this: a
  truncated round is a NO-SIGNAL round, not a round in which every
  client contributed nothing).

CSV SCHEMA IS LOCKED
--------------------
aggregation_results.csv keeps exactly its five original columns because
P3 and P4 consume it. All new per-round diagnostics go to the separate
aggregation_round_log.csv instead of adding columns.

CONVERGENCE-SPEED METRIC (from the original plan's "compare convergence")
-------------------------------------------------------------------------
Reports rounds-to-reach-target-accuracy for each variant at several
accuracy thresholds, alongside final accuracy. Targets now extend to
0.90 because 50 rounds reaches ~0.89 (5 rounds only reached ~0.74).
Final-round accuracy alone is noisy once the model has plateaued, so the
summary also reports best-round accuracy and the mean over the last 10
rounds.

OUTPUTS
-------
  aggregation_results.csv      — round, client_id, aggregation_weight,
                                 global_accuracy_vanilla,
                                 global_accuracy_weighted   [LOCKED SCHEMA]
  aggregation_round_log.csv    — per-round diagnostics: v0, vN, gain,
                                 truncated flag, phi range, weight range
  fairness_metrics.txt         — ShapFed-style Pearson correlations,
                                 truncation statistics, convergence table
  aggregation_comparison.png   — accuracy-vs-round curves, truncated
                                 rounds shaded

Usage (from repo root, venv active, after preprocess_har.py):

    python shapley_weighted_fedavg.py                  # 50 rounds, eps_b=0.02
    python shapley_weighted_fedavg.py --rounds 3       # quick smoke test
    python shapley_weighted_fedavg.py --adaptive --suffix _adaptive

Runtime: roughly 10-25 min for 50 rounds on CPU depending on machine
(5 rounds was 1-2 min). Progress and ETA print every round, and both
CSVs are flushed every round, so an interrupted run keeps its completed
rounds.
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
    get_model, local_train, fedavg, evaluate, clone_state_dict,
    LOCAL_EPOCHS, LOCAL_LR,
)
from export_shapley_csv import gtg_shapley_one_round

PKL_PATH = "preprocessed/har_clients.pkl"
P1_SHAPLEY_CSV = "shapley_scores.csv"   # P1's committed 50-round scores

OUT_CSV = "aggregation_results.csv"
OUT_ROUND_LOG = "aggregation_round_log.csv"
OUT_TXT = "fairness_metrics.txt"
OUT_PNG = "aggregation_comparison.png"

GLOBAL_SEED = 42
N_ROUNDS = 50      # matches P1's 50-round pipeline (was 5)
BETA = 50.0        # inverse softmax temperature for Shapley -> weight mapping
WARMUP_ROUNDS = 0  # original plan's warm-up: rounds of plain FedAvg before
                   # Shapley weighting kicks in (0 = weight from round 1)

# --- GTG truncation thresholds (see module docstring) ---
EPS_B_FIXED = 0.02      # P1's value: round skipped if accuracy gain < 2 points
EPS_I_FIXED = 0.005     # P1's inner-loop skip threshold
EPS_B_FLOOR = 0.0035    # adaptive mode: ~10 test samples on HAR's 2947-sample
                        # test set; keeps Shapley live after convergence
EPS_I_RATIO = 0.25      # adaptive eps_i = EPS_I_RATIO * eps_b (P1 used 4:1)

# Accuracy thresholds for the convergence-speed comparison.
# Extended past 0.70 because 50 rounds reaches ~0.89.
CONV_TARGETS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]

# Number of trailing rounds averaged for the "plateau accuracy" summary.
PLATEAU_WINDOW = 10


# ----------------------------------------------------------------------
# Reweighting function: per-round Shapley scores -> aggregation weights
# ----------------------------------------------------------------------
def shapley_to_weights(phi_dict, client_ids, client_sizes, beta=BETA):
    """Softmax over per-round Shapley values. Returns dict {cid: weight},
    weights sum to 1. Numerically stabilized (max-subtraction).

    FALLBACK: if this round was truncated by GTG-Shapley (all phi exactly
    0), there is no contribution signal, so we fall back to standard
    size-proportional FedAvg weights for this round rather than silently
    going uniform (uniform would over-weight tiny clients relative to
    vanilla FedAvg for no reason)."""
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


def fmt_eta(seconds):
    seconds = int(max(seconds, 0))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


# ----------------------------------------------------------------------
# Cross-check against P1's committed scores
# ----------------------------------------------------------------------
def verify_round1_against_p1(phi_round1, client_ids, eps_b_used):
    """Round 1 of the weighted arm is computed from the identical init
    state and the identical size-weighted provisional aggregate that P1
    used, with the same seeds — so the round-1 phi values must match
    P1's shapley_scores.csv exactly when eps_b/eps_i match P1's.

    This is the coupling check between P2 and P1: if P1 regenerates the
    pipeline and this stops matching, P2's assumptions are stale.
    Reports but never raises, so a run is never blocked by it."""
    if not os.path.exists(P1_SHAPLEY_CSV):
        print(f"  [check] {P1_SHAPLEY_CSV} not found — skipping P1 cross-check.")
        return
    p1 = {}
    with open(P1_SHAPLEY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["round"]) == 1:
                p1[int(row["client_id"])] = float(row["shapley_value"])
    if not p1:
        print(f"  [check] no round-1 rows in {P1_SHAPLEY_CSV} — skipping.")
        return

    common = [c for c in client_ids if c in p1]
    if len(common) < 3:
        print("  [check] client ids do not overlap with P1's CSV — skipping.")
        return

    ours = np.array([phi_round1[c] for c in common])
    theirs = np.array([p1[c] for c in common])
    max_abs = float(np.max(np.abs(ours - theirs)))
    if np.std(ours) == 0 or np.std(theirs) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ours, theirs)[0, 1])

    # Tolerance tiers. The model is float32 and fedavg accumulates over 30
    # state dicts, so bitwise-identical agreement is NOT expected even when
    # everything is configured identically: summation order and CSV
    # round-tripping both perturb the low bits. Differences at the 1e-6
    # level against phi values of order 1e-2 are ~1e-4 relative — noise,
    # not disagreement. A genuine configuration mismatch (different seed,
    # partition, or model) shows up as a difference of the same order as
    # the phi values themselves, along with a collapsed correlation.
    exact_expected = abs(eps_b_used - EPS_B_FIXED) < 1e-12
    phi_scale = float(np.max(np.abs(theirs))) or 1.0
    rel = max_abs / phi_scale

    if max_abs < 1e-12:
        print(f"  [check] round-1 phi is bitwise identical to P1's "
              f"shapley_scores.csv. P1/P2 coupling verified.")
    elif rel < 1e-3 and (np.isnan(corr) or corr > 0.999):
        print(f"  [check] round-1 phi matches P1's shapley_scores.csv "
              f"(max diff {max_abs:.2e} = {rel:.1e} relative, r={corr:+.4f}) "
              f"— float32 accumulation noise. P1/P2 coupling verified.")
    elif exact_expected:
        print(f"  [check] WARNING: round-1 phi differs from P1's CSV "
              f"(max diff {max_abs:.2e} = {rel:.1e} relative, r={corr:+.4f}) "
              f"even though eps_b matches P1's. This is larger than float32 "
              f"noise. Check that P1 has not changed seeds, the partition, "
              f"or the model since exporting that CSV.")
    else:
        print(f"  [check] round-1 phi differs from P1's CSV (max diff "
              f"{max_abs:.2e}, r={corr:+.4f}) — expected: adaptive mode uses "
              f"eps_b={eps_b_used:g}, not {EPS_B_FIXED:g}.")


def main():
    ap = argparse.ArgumentParser(
        description="P2 adaptive aggregation: vanilla vs Shapley-weighted FedAvg")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS,
                    help=f"number of FL rounds (default {N_ROUNDS})")
    ap.add_argument("--beta", type=float, default=BETA,
                    help=f"softmax inverse temperature (default {BETA:g})")
    ap.add_argument("--warmup", type=int, default=WARMUP_ROUNDS,
                    help="rounds of plain FedAvg before Shapley weighting")
    ap.add_argument("--adaptive", action="store_true",
                    help=f"use eps_b={EPS_B_FLOOR:g} instead of P1's "
                         f"{EPS_B_FIXED:g}, keeping Shapley live after "
                         f"convergence")
    ap.add_argument("--suffix", type=str, default="",
                    help="suffix for output filenames, e.g. _adaptive")
    ap.add_argument("--skip-standalone", action="store_true",
                    help="skip the standalone-accuracy fairness metric")
    args = ap.parse_args()

    n_rounds = args.rounds
    beta = args.beta
    warmup = args.warmup

    if args.adaptive:
        eps_b, eps_i = EPS_B_FLOOR, EPS_B_FLOOR * EPS_I_RATIO
        mode = "adaptive"
    else:
        eps_b, eps_i = EPS_B_FIXED, EPS_I_FIXED
        mode = "fixed (P1 defaults)"

    sfx = args.suffix
    out_csv = OUT_CSV.replace(".csv", f"{sfx}.csv")
    out_log = OUT_ROUND_LOG.replace(".csv", f"{sfx}.csv")
    out_txt = OUT_TXT.replace(".txt", f"{sfx}.txt")
    out_png = OUT_PNG.replace(".png", f"{sfx}.png")

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

    print("=" * 70)
    print("P2 — Shapley-weighted FedAvg vs vanilla FedAvg")
    print("=" * 70)
    print(f"clients={len(client_ids)}  rounds={n_rounds}  beta={beta:g}  "
          f"warmup={warmup}  seed={GLOBAL_SEED}")
    print(f"GTG truncation mode: {mode}  ->  eps_b={eps_b:g}, eps_i={eps_i:g}")
    print(f"test set: {len(y_test)} samples "
          f"(1 sample = {1.0 / len(y_test):.5f} accuracy; "
          f"eps_b = {eps_b * len(y_test):.0f} samples)")
    print(f"Initial (round 0) accuracy: {acc_init:.4f}")
    if warmup > 0:
        print(f"Warm-up enabled: first {warmup} round(s) of the weighted run "
              f"use standard size-weighted FedAvg.")
    print()

    acc_vanilla_hist, acc_weighted_hist = [acc_init], [acc_init]
    rows = []
    round_log = []
    cum_phi = {c: 0.0 for c in client_ids}
    cum_weight = {c: 0.0 for c in client_ids}
    truncated_rounds = []

    csv_fields = ["round", "client_id", "aggregation_weight",
                  "global_accuracy_vanilla", "global_accuracy_weighted"]
    log_fields = ["round", "v0", "vN", "gain", "truncated", "warmup",
                  "phi_min", "phi_max", "phi_abs_mean",
                  "weight_min", "weight_max", "weight_ratio"]

    t0 = time.time()
    for r in range(1, n_rounds + 1):
        t_round = time.time()

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
            eps_b=eps_b, eps_i=eps_i,
        )

        is_truncated = bool(np.allclose(
            np.array([phi[c] for c in client_ids]), 0.0))
        if is_truncated:
            truncated_rounds.append(r)

        if r == 1:
            verify_round1_against_p1(phi, client_ids, eps_b)

        in_warmup = r <= warmup
        if in_warmup:
            # Original plan's warm-up: plain FedAvg while scores stabilize.
            weights = size_weights(client_ids, client_sizes)
            state_weighted = provisional  # provisional IS the size-weighted aggregate
        else:
            weights = shapley_to_weights(phi, client_ids, client_sizes, beta=beta)
            state_weighted = fedavg(
                [cs_w[c] for c in client_ids],
                [weights[c] for c in client_ids],
            )
        acc_w = evaluate(state_weighted, X_test, y_test)
        acc_weighted_hist.append(acc_w)

        phi_vals = np.array([phi[c] for c in client_ids])
        w_vals = np.array([weights[c] for c in client_ids])

        tag = ""
        if in_warmup:
            tag = " [warm-up]"
        elif is_truncated:
            tag = " [TRUNCATED -> size weights]"

        elapsed = time.time() - t0
        eta = (elapsed / r) * (n_rounds - r)
        print(f"Round {r:>3}/{n_rounds}  "
              f"vanilla={acc_v:.4f}  weighted={acc_w:.4f}{tag}")
        print(f"           gain={vN - v0:+.4f}  "
              f"phi=[{phi_vals.min():+.4f}, {phi_vals.max():+.4f}]  "
              f"w=[{w_vals.min():.4f}, {w_vals.max():.4f}]  "
              f"({time.time() - t_round:.1f}s, ETA {fmt_eta(eta)})")

        round_log.append({
            "round": r,
            "v0": v0,
            "vN": vN,
            "gain": vN - v0,
            "truncated": int(is_truncated),
            "warmup": int(in_warmup),
            "phi_min": float(phi_vals.min()),
            "phi_max": float(phi_vals.max()),
            "phi_abs_mean": float(np.abs(phi_vals).mean()),
            "weight_min": float(w_vals.min()),
            "weight_max": float(w_vals.max()),
            "weight_ratio": float(w_vals.max() / w_vals.min()),
        })

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

        # Flush both CSVs every round so an interrupted 50-round run
        # keeps everything it has already computed.
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            w.writerows(rows)
        with open(out_log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            w.writeheader()
            w.writerows(round_log)

    elapsed = time.time() - t0
    print(f"\nSaved {len(rows)} rows to {out_csv}  ({fmt_eta(elapsed)} total)")
    print(f"Saved per-round diagnostics to {out_log}")

    # ------------------------------------------------------------------
    # Truncation statistics — the headline 50-round finding
    # ------------------------------------------------------------------
    n_trunc = len(truncated_rounds)
    live_rounds = [r for r in range(1, n_rounds + 1)
                   if r not in truncated_rounds]
    first_trunc_run = None
    trunc_set = set(truncated_rounds)
    for r in truncated_rounds:
        if all(x in trunc_set for x in range(r, n_rounds + 1)):
            first_trunc_run = r
            break

    trunc_lines = [
        "",
        "GTG-Shapley truncation statistics",
        "---------------------------------",
        f"  mode: {mode}   eps_b={eps_b:g}  eps_i={eps_i:g}",
        f"  truncated (all-phi-zero, size-weight fallback) rounds: "
        f"{n_trunc}/{n_rounds} ({100.0 * n_trunc / n_rounds:.0f}%)",
        f"  Shapley-live rounds: {len(live_rounds)}/{n_rounds} "
        f"({100.0 * len(live_rounds) / n_rounds:.0f}%)",
    ]
    if first_trunc_run is not None:
        trunc_lines.append(
            f"  signal collapses permanently from round {first_trunc_run} "
            f"onward (every round from there is truncated)")
    trunc_lines.append(
        "  truncated round list: "
        + (str(truncated_rounds) if n_trunc <= 25
           else str(truncated_rounds[:25]) + " ..."))
    trunc_lines.append(
        "  NOTE for P3: a truncated round is a NO-SIGNAL round, not a round "
        "in which every client contributed nothing. Exclude these rounds from "
        "eligible-round counts before applying the sustained-deviation ratio.")

    # ------------------------------------------------------------------
    # Convergence-speed comparison
    # ------------------------------------------------------------------
    conv_lines = ["", "Convergence speed (first round reaching target accuracy):",
                  "  target   vanilla   weighted"]
    for t in CONV_TARGETS:
        rv = rounds_to_target(acc_vanilla_hist, t)
        rw = rounds_to_target(acc_weighted_hist, t)
        conv_lines.append(f"  >={t:.2f}    {str(rv):>5}     {str(rw):>5}")

    # Plateau / best-round summary: at 50 rounds the final round alone is
    # noisy, so report best and trailing-window mean too.
    pw = min(PLATEAU_WINDOW, n_rounds)
    plateau_v = float(np.mean(acc_vanilla_hist[-pw:]))
    plateau_w = float(np.mean(acc_weighted_hist[-pw:]))
    best_v, best_w = max(acc_vanilla_hist), max(acc_weighted_hist)
    best_vr = int(np.argmax(acc_vanilla_hist))
    best_wr = int(np.argmax(acc_weighted_hist))

    summary_lines = [
        "",
        "Accuracy summary (final-round alone is noisy after convergence):",
        f"  final round {n_rounds}      vanilla={acc_vanilla_hist[-1]:.4f}  "
        f"weighted={acc_weighted_hist[-1]:.4f}  "
        f"gap={acc_weighted_hist[-1] - acc_vanilla_hist[-1]:+.4f}",
        f"  mean of last {pw:<2d} rounds vanilla={plateau_v:.4f}  "
        f"weighted={plateau_w:.4f}  gap={plateau_w - plateau_v:+.4f}",
        f"  best round          vanilla={best_v:.4f} (r{best_vr})  "
        f"weighted={best_w:.4f} (r{best_wr})",
    ]

    # ------------------------------------------------------------------
    # ShapFed-style fairness metric.
    # Standalone accuracy = each client trains ALONE from the same init
    # for an equivalent budget (n_rounds * LOCAL_EPOCHS epochs), evaluated
    # on the shared global test set. Fairness = Pearson correlation
    # between standalone accuracy and the credit signal each scheme
    # assigns. Higher correlation = credit assignment tracks genuine data
    # usefulness more faithfully.
    # ------------------------------------------------------------------
    if args.skip_standalone:
        print("\nSkipping standalone-accuracy fairness metric (--skip-standalone).")
        fairness_lines = ["", "(standalone fairness metric skipped)"]
    else:
        print(f"\nComputing standalone client accuracies for fairness metric "
              f"({n_rounds * LOCAL_EPOCHS} epochs x {len(client_ids)} clients)...")
        t_sa = time.time()
        standalone_acc = {}
        for cid in client_ids:
            m = get_model(seed=GLOBAL_SEED)
            local_train(m, data[cid]["X_train"], data[cid]["y_train"],
                        epochs=n_rounds * LOCAL_EPOCHS, lr=LOCAL_LR,
                        seed=GLOBAL_SEED + cid)
            standalone_acc[cid] = evaluate(m.state_dict(), X_test, y_test)
        print(f"  done in {fmt_eta(time.time() - t_sa)}")

        sa = np.array([standalone_acc[c] for c in client_ids])
        ph = np.array([cum_phi[c] for c in client_ids])
        wt = np.array([cum_weight[c] for c in client_ids])
        sz = np.array([client_sizes[c] for c in client_ids], dtype=float)

        def pearson(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        fairness_lines = [
            "",
            "ShapFed-style fairness metric (Pearson correlation between",
            "standalone client accuracy on the global test set and the",
            "credit signal each scheme assigns):",
            "",
            f"  vanilla FedAvg (credit = data size):        r = {pearson(sa, sz):+.4f}",
            f"  per-round Shapley values (cumulative phi):  r = {pearson(sa, ph):+.4f}",
            f"  Shapley-softmax aggregation weights:        r = {pearson(sa, wt):+.4f}",
            "",
            "  Caveat: with most rounds truncated, cumulative phi and the",
            "  aggregation weights are both dominated by the few live rounds,",
            "  so these correlations rest on fewer effective observations than",
            "  the round count suggests.",
        ]

    lines = ([
        "P2 — Shapley-weighted FedAvg vs vanilla FedAvg",
        "=" * 55,
        f"Config: BETA={beta:g}, WARMUP_ROUNDS={warmup}, N_ROUNDS={n_rounds}, "
        f"seed={GLOBAL_SEED}",
        f"GTG truncation: {mode}, eps_b={eps_b:g}, eps_i={eps_i:g}",
        f"Dataset: UCI HAR, {len(client_ids)} clients, "
        f"{len(y_test)} global test samples",
        f"Runtime: {fmt_eta(elapsed)}",
    ] + summary_lines + trunc_lines + conv_lines + fairness_lines + [
        "",
        "Per-round accuracy (vanilla):  "
        + ", ".join(f"{a:.4f}" for a in acc_vanilla_hist),
        "Per-round accuracy (weighted): "
        + ", ".join(f"{a:.4f}" for a in acc_weighted_hist),
    ])
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))

    # ------------------------------------------------------------------
    # Plot — 50-round friendly: sparse markers, shaded truncated rounds
    # ------------------------------------------------------------------
    rounds_axis = list(range(n_rounds + 1))
    label_w = f"Shapley-weighted (softmax, beta={beta:g}"
    label_w += f", warm-up={warmup})" if warmup else ")"

    markevery = max(1, n_rounds // 12)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Shade truncated rounds so the reader sees where the signal died.
    for r in truncated_rounds:
        ax.axvspan(r - 0.5, r + 0.5, color="#d62728", alpha=0.07, linewidth=0)
    if truncated_rounds:
        ax.plot([], [], "s", color="#d62728", alpha=0.35, markersize=9,
                label=f"GTG-truncated round ({n_trunc}/{n_rounds})")

    ax.plot(rounds_axis, acc_vanilla_hist, "o-", markevery=markevery,
            linewidth=1.8, markersize=5, color="#888888",
            label="Vanilla FedAvg (size-weighted)")
    ax.plot(rounds_axis, acc_weighted_hist, "s-", markevery=markevery,
            linewidth=1.8, markersize=5, color="#1f77b4", label=label_w)

    ax.set_xlabel("Global round")
    ax.set_ylabel("Global test accuracy")
    ax.set_title(f"Vanilla vs Shapley-weighted FedAvg — UCI HAR, "
                 f"{len(client_ids)} clients, seed {GLOBAL_SEED}\n"
                 f"eps_b={eps_b:g} ({mode})", fontsize=11)
    ax.set_xticks(range(0, n_rounds + 1, max(1, n_rounds // 10)))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
