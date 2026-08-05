"""
P2 (new, 50-round) — Shapley signal decay and truncation-threshold sensitivity.

WHY THIS EXISTS
---------------
When P1's pipeline moved from 5 rounds to 50, a rule that had never fired
started firing constantly. GTG-Shapley declares a round to carry no
contribution signal — and returns all phi = 0 — when

    |vN - v0| <= eps_b

with eps_b = 0.02, i.e. when the round's global accuracy gain is under
2 percentage points. At 5 rounds the model was still gaining 2-9 points
per round, so the rule was invisible. At 50 rounds the model converges
and per-round gains fall to 0.1-0.5 points, so most rounds are skipped.

This script measures the effect directly on P1's committed
shapley_scores.csv. It runs in about a second and needs no FL training,
no preprocessed data, and no GPU — only the CSV.

It answers three questions the writeup needs:

  1. HOW MANY rounds actually carry Shapley signal at 50 rounds?
  2. WHEN does the signal die, and is the collapse permanent?
  3. HOW SENSITIVE is that to eps_b? What threshold would keep Shapley
     live through convergence?

WHY THIS MATTERS FOR EACH SUB-TEAM
----------------------------------
P2 (aggregation): every truncated round falls back to size-proportional
weights, so a naive 50-round run is mostly vanilla FedAvg wearing a
Shapley label. The two curves land on top of each other and the result
says nothing about the method.

P3 (Byzantine detection): the sustained-deviation rule needs a count of
ELIGIBLE rounds. Truncated rounds are no-signal rounds, not rounds in
which everyone contributed zero. If 38 of 50 rounds are all-zero, the
"flagged in >=30% of eligible rounds" rule is operating on 12 rounds,
not 50 — a very different statistical footing, and worth stating
explicitly in the paper rather than leaving implicit.

P4 (blockchain): rounds with all-zero scores still get logged on chain.
Worth knowing that most on-chain entries carry no valuation content.

METHOD
------
The exact v0 for round r is the accuracy of the global model entering
that round, which equals model_accuracy for round r-1 in P1's CSV (P1
writes vN as model_accuracy). Round 1's v0 is the untrained init model
and is not in the CSV, so round 1 is reported from its phi values
directly rather than from a reconstructed gain.

The threshold sweep re-applies the |vN - v0| <= eps test at other eps
values against those same measured gains. That tells us how many rounds
each threshold would have kept live WITHOUT rerunning training, which is
the cheap way to pick a candidate threshold before spending 20 minutes
on a full run.

Usage (from repo root; no venv packages beyond pandas/matplotlib needed):

    python shapley_signal_decay.py
    python shapley_signal_decay.py --csv shapley_scores.csv

Outputs:
    shapley_signal_decay.csv   round, accuracy, gain, phi_abs_mean,
                               phi_min, phi_max, all_zero
    shapley_signal_decay.png   two panels: per-round gain vs eps_b
                               thresholds, and live-round count vs eps_b
    shapley_signal_decay.txt   summary + threshold sensitivity table
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_IN = "shapley_scores.csv"
CSV_OUT = "shapley_signal_decay.csv"
PNG_OUT = "shapley_signal_decay.png"
TXT_OUT = "shapley_signal_decay.txt"

# P1's current truncation threshold, for reference lines in the plot.
EPS_B_P1 = 0.02

# Candidate thresholds for the sensitivity sweep.
EPS_CANDIDATES = [0.02, 0.015, 0.01, 0.0075, 0.005, 0.0035, 0.002, 0.001]

# HAR global test set size, used to express thresholds in "test samples".
# Detected from preprocessed/har_clients.pkl when that file is present;
# this constant is only the fallback when it is not. Note the pipeline
# re-splits merged train+test 80/20 per subject, so the global test set is
# NOT the 2947 samples of the original UCI HAR test split — detecting it is
# the difference between a correct and a misleading "test samples" column.
N_TEST_DEFAULT = 2074
PKL_PATH = "preprocessed/har_clients.pkl"


def detect_n_test(pkl_path=PKL_PATH):
    """Return the global test set size from the preprocessed pickle, or
    None if it is unavailable (the pickle is gitignored, so anyone reading
    a fresh clone will not have it)."""
    if not os.path.exists(pkl_path):
        return None
    try:
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return int(sum(len(d["y_test"]) for d in data.values()))
    except Exception:
        return None

ZERO_TOL = 1e-12


def main():
    ap = argparse.ArgumentParser(
        description="Measure GTG-Shapley signal decay across rounds")
    ap.add_argument("--csv", default=CSV_IN,
                    help=f"P1 Shapley scores CSV (default {CSV_IN})")
    ap.add_argument("--n-test", type=int, default=None,
                    help="global test set size, for sample-count reporting "
                         "(default: detected from the preprocessed pickle)")
    args = ap.parse_args()

    if args.n_test is None:
        detected = detect_n_test()
        if detected:
            args.n_test = detected
            n_test_src = f"detected from {PKL_PATH}"
        else:
            args.n_test = N_TEST_DEFAULT
            n_test_src = (f"assumed — {PKL_PATH} not found; pass --n-test if "
                          f"this is wrong")
    else:
        n_test_src = "supplied on the command line"

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"Could not find {args.csv}. Run this from the Shapley-FL repo "
            f"root, after P1 has committed the scores CSV.")

    df = pd.read_csv(args.csv)
    required = {"round", "client_id", "shapley_value", "model_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing expected columns: {missing}")

    n_clients = df["client_id"].nunique()
    rounds = sorted(df["round"].unique())
    n_rounds = len(rounds)

    # ---- per-round table ----
    recs = []
    prev_acc = None
    for r in rounds:
        grp = df[df["round"] == r]
        phi = grp["shapley_value"].to_numpy()
        acc = float(grp["model_accuracy"].iloc[0])
        gain = float("nan") if prev_acc is None else acc - prev_acc
        recs.append({
            "round": int(r),
            "accuracy": acc,
            "gain": gain,
            "abs_gain": abs(gain) if prev_acc is not None else float("nan"),
            "phi_min": float(phi.min()),
            "phi_max": float(phi.max()),
            "phi_abs_mean": float(np.abs(phi).mean()),
            "all_zero": int(bool(np.all(np.abs(phi) < ZERO_TOL))),
        })
        prev_acc = acc

    out = pd.DataFrame(recs)
    out.to_csv(CSV_OUT, index=False)

    zero_rounds = out.loc[out["all_zero"] == 1, "round"].tolist()
    live_rounds = out.loc[out["all_zero"] == 0, "round"].tolist()
    n_zero = len(zero_rounds)

    # Round from which truncation is permanent (every later round zero too).
    zero_set = set(zero_rounds)
    collapse_at = None
    for r in zero_rounds:
        if all(x in zero_set for x in range(r, rounds[-1] + 1)):
            collapse_at = r
            break

    # ---- threshold sensitivity sweep ----
    # Round 1's gain is unknown (v0 = init accuracy, not in the CSV), but
    # round 1 is live at every candidate threshold in practice, so it is
    # counted as live and noted as such.
    gains = out["abs_gain"].to_numpy()[1:]  # rounds 2..N
    sweep = []
    for eps in EPS_CANDIDATES:
        live = int((gains > eps).sum()) + 1  # +1 for round 1
        sweep.append({
            "eps_b": eps,
            "live_rounds": live,
            "truncated_rounds": n_rounds - live,
            "live_pct": 100.0 * live / n_rounds,
            "test_samples": eps * args.n_test,
        })
    sweep_df = pd.DataFrame(sweep)

    # Sanity check: the sweep at P1's own eps_b must reproduce the
    # observed all-zero count. If it does not, the CSV's model_accuracy
    # column is not vN, or the run used different thresholds.
    at_p1 = sweep_df.loc[sweep_df["eps_b"] == EPS_B_P1]
    check_msg = ""
    if not at_p1.empty:
        predicted_zero = int(at_p1["truncated_rounds"].iloc[0])
        if predicted_zero == n_zero:
            check_msg = (f"  [check] sweep reproduces the observed truncation "
                         f"count at eps_b={EPS_B_P1:g} ({n_zero} rounds) — "
                         f"gain reconstruction verified.")
        else:
            check_msg = (f"  [check] WARNING: sweep predicts {predicted_zero} "
                         f"truncated rounds at eps_b={EPS_B_P1:g} but the CSV "
                         f"has {n_zero}. The model_accuracy column may not be "
                         f"vN, or P1 changed eps_b/eps_i.")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    r_axis = out["round"].to_numpy()
    ax1.semilogy(r_axis[1:], np.maximum(gains, 1e-5), "o-", markersize=3.5,
                 linewidth=1.4, color="#1f77b4",
                 label="|per-round accuracy gain|")
    for eps, style, col in [(EPS_B_P1, "-", "#d62728"),
                            (0.0035, "--", "#2ca02c")]:
        ax1.axhline(eps, linestyle=style, color=col, linewidth=1.5,
                    label=f"eps_b = {eps:g}"
                          + (" (P1 current)" if eps == EPS_B_P1
                             else " (proposed floor)"))
    if collapse_at is not None:
        ax1.axvline(collapse_at, color="#7f7f7f", linestyle=":", linewidth=1.5)
        ax1.annotate(f"permanent collapse\nfrom round {collapse_at}",
                     xy=(collapse_at / max(rounds), 0.06),
                     xycoords="axes fraction", xytext=(6, 0),
                     textcoords="offset points",
                     fontsize=8, color="#555555", va="bottom")
    ax1.set_xlabel("Global round")
    ax1.set_ylabel("|vN - v0|  (log scale)")
    ax1.set_title("Per-round accuracy gain vs truncation threshold")
    ax1.grid(alpha=0.3, linestyle="--", which="both")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.95)

    ax2.plot(sweep_df["eps_b"], sweep_df["live_rounds"], "s-",
             color="#1f77b4", linewidth=1.8, markersize=6)
    ax2.axvline(EPS_B_P1, color="#d62728", linestyle="-", linewidth=1.5,
                label=f"P1 current ({EPS_B_P1:g})")
    ax2.axvline(0.0035, color="#2ca02c", linestyle="--", linewidth=1.5,
                label="proposed floor (0.0035)")
    ax2.set_xscale("log")
    ax2.set_xlabel("eps_b (truncation threshold)")
    ax2.set_ylabel(f"Shapley-live rounds (of {n_rounds})")
    ax2.set_title("Threshold sensitivity")
    ax2.set_ylim(0, n_rounds + 2)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"GTG-Shapley signal decay — UCI HAR, {n_clients} clients, "
                 f"{n_rounds} rounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)

    # ---- text summary ----
    lines = []
    lines.append("GTG-Shapley signal decay across rounds")
    lines.append("=" * 60)
    lines.append(f"source: {args.csv}   ({len(df)} rows, {n_clients} clients, "
                 f"{n_rounds} rounds)")
    lines.append("")
    lines.append("Truncation at P1's current threshold (eps_b = "
                 f"{EPS_B_P1:g}):")
    lines.append(f"  all-zero (truncated) rounds : {n_zero}/{n_rounds} "
                 f"({100.0 * n_zero / n_rounds:.0f}%)")
    lines.append(f"  Shapley-live rounds         : {len(live_rounds)}/{n_rounds} "
                 f"({100.0 * len(live_rounds) / n_rounds:.0f}%)")
    lines.append(f"  live round numbers          : {live_rounds}")
    if collapse_at is not None:
        lines.append(f"  permanent collapse from round {collapse_at}: every "
                     f"round from {collapse_at} to {rounds[-1]} is truncated")
    if check_msg:
        lines.append(check_msg)
    lines.append("")

    early = out.iloc[1:10]["abs_gain"].mean()
    late = out.iloc[max(1, n_rounds // 2):]["abs_gain"].mean()
    lines.append("Why the signal dies:")
    lines.append(f"  mean |gain|, rounds 2-10          : {early:.5f}")
    lines.append(f"  mean |gain|, second half of run   : {late:.5f}")
    lines.append(f"  ratio                             : "
                 f"{early / late if late else float('nan'):.1f}x")
    lines.append("  The threshold is an ABSOLUTE accuracy delta. It was")
    lines.append("  calibrated on a 5-round run where every round gained")
    lines.append("  several points. After convergence the same rule discards")
    lines.append("  every round, so the valuation signal is thrown away")
    lines.append("  exactly where per-client differences become subtle.")
    lines.append("")

    lines.append("Threshold sensitivity (how many rounds each eps_b keeps live):")
    lines.append("  eps_b     live   truncated   live%   ~test samples")
    for _, row in sweep_df.iterrows():
        lines.append(f"  {row['eps_b']:<8.4f}  {int(row['live_rounds']):>3}   "
                     f"{int(row['truncated_rounds']):>7}   "
                     f"{row['live_pct']:>5.0f}%   {row['test_samples']:>6.0f}")
    lines.append("")
    lines.append(f"  For reference: on a {args.n_test}-sample test set "
                 f"({n_test_src}), one sample = {1.0 / args.n_test:.5f} "
                 f"accuracy.")
    lines.append("  Because v0 and vN are evaluated on the SAME fixed test set")
    lines.append("  with two highly correlated models, the meaningful noise")
    lines.append("  floor is the paired-disagreement scale (a handful of")
    lines.append("  samples), not the independent-sample standard error.")
    lines.append(f"  eps_b = 0.0035 (~{0.0035 * args.n_test:.0f} samples) is "
                 f"the proposed floor.")
    lines.append("")

    lines.append("Handoff notes:")
    lines.append("  P2: truncated rounds fall back to size-proportional")
    lines.append("      weights, so Shapley weighting is inactive in them.")
    lines.append("      Report the live-round count alongside any accuracy")
    lines.append("      comparison, or the comparison is misleading.")
    lines.append("  P3: exclude truncated rounds from eligible-round counts")
    lines.append("      before applying the sustained-deviation ratio. A")
    lines.append("      client cannot deviate from its history in a round")
    lines.append("      where nobody has a score.")
    lines.append("  P4: all-zero rounds still produce on-chain entries that")
    lines.append("      carry no valuation content; worth noting in the audit")
    lines.append("      trail description.")

    text = "\n".join(lines)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)
    print()
    print(f"Wrote {CSV_OUT}, {PNG_OUT}, {TXT_OUT}")


if __name__ == "__main__":
    main()
