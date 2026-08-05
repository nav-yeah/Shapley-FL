"""
P2 - Jain's Fairness Index on Shapley aggregation weights
==========================================================
Computes Jain's fairness index J = (sum(w_i))^2 / (n * sum(w_i^2))
for the per-round aggregation weights stored in aggregation_results.csv.

J ranges from 1/n (maximally unfair: one client gets all weight) to 1.0
(perfectly fair: every client weighted equally). Higher = more evenly the
Shapley-derived aggregation weight is spread across clients.

50-ROUND UPDATE — WHY THE POOLED NUMBER IS NOW MISLEADING
----------------------------------------------------------
At 5 rounds every round carried Shapley signal, so a single mean J over
all rounds described the method. At 50 rounds most rounds are truncated
by GTG-Shapley (all phi = 0) and fall back to SIZE-PROPORTIONAL weights.
Those fallback rounds are not measuring Shapley fairness at all — they
are measuring how evenly HAR client sizes happen to be distributed, and
because the 30 subjects hold roughly 225-330 samples each, they score a
near-perfect J of about 0.99.

Averaging those in drags the reported fairness up toward vanilla FedAvg's
value and hides what the Shapley weighting actually does. So this script
now splits the rounds into two groups and reports them separately:

  LIVE rounds      — Shapley weighting was active (phi not all zero)
  TRUNCATED rounds — fell back to size weights (no Shapley signal)

A round is classified as truncated using aggregation_round_log.csv if
that file is present (written by the current shapley_weighted_fedavg.py).
If it is absent — e.g. reading an older results CSV — the script falls
back to detecting rounds whose weight vector matches size-proportional
weights, and says so in the output.

The headline number for the paper is the mean J over LIVE rounds. Report
the truncated-round J too, as the size-weighting reference point.

Reads : aggregation_results.csv  (locked schema:
        round, client_id, aggregation_weight,
        global_accuracy_vanilla, global_accuracy_weighted)
        aggregation_round_log.csv  (optional; round, truncated, ...)
Writes: jains_fairness.txt   (human-readable summary)
        jains_fairness.csv   (round, n_clients, jain_index, truncated)
        jains_fairness.png   (J vs round, truncated rounds shaded)

Run:  python jains_fairness.py
      python jains_fairness.py --csv aggregation_results_adaptive.csv \
                               --log aggregation_round_log_adaptive.csv \
                               --suffix _adaptive
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_IN = "aggregation_results.csv"
LOG_IN = "aggregation_round_log.csv"
TXT_OUT = "jains_fairness.txt"
CSV_OUT = "jains_fairness.csv"
PNG_OUT = "jains_fairness.png"

# Tolerance for "these weights are indistinguishable from size-proportional".
SIZE_MATCH_TOL = 1e-9

# Rounds listed individually in the text report; beyond this, the report
# summarises and points at the CSV (50 rounds of raw lines is unreadable).
MAX_LISTED_ROUNDS = 12


def jain_index(weights):
    """Jain's fairness index for a 1-D array-like of weights."""
    weights = np.asarray(weights, dtype=float)
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    n = len(weights)
    if s2 == 0:
        # All-zero round = no signal; fairness is undefined, return NaN.
        return float("nan")
    return float((s1 ** 2) / (n * s2))


def load_truncated_flags(log_path, rounds):
    """Return (set_of_truncated_rounds, source_description).

    Prefers the explicit round log written by shapley_weighted_fedavg.py.
    Returns (None, reason) if unavailable, so the caller can fall back."""
    if not os.path.exists(log_path):
        return None, (f"{log_path} not found — truncated rounds inferred "
                      f"from the weight vectors instead")
    log = pd.read_csv(log_path)
    if "truncated" not in log.columns or "round" not in log.columns:
        return None, (f"{log_path} lacks a 'truncated' column — truncated "
                      f"rounds inferred from the weight vectors instead")
    flags = set(log.loc[log["truncated"] == 1, "round"].astype(int).tolist())
    return flags, f"truncation flags read from {log_path}"


def infer_truncated_from_weights(df):
    """Fallback classifier: a round is treated as truncated (size-weight
    fallback) if its weight vector is proportional to the SAME vector in
    every other candidate round — i.e. it equals the size-proportional
    weights. We recover the size weights as the modal weight vector
    across rounds, since fallback rounds all produce exactly it."""
    by_round = {}
    for rnd, grp in df.groupby("round"):
        g = grp.sort_values("client_id")
        by_round[int(rnd)] = g["aggregation_weight"].to_numpy()

    # The size-weight vector is whichever vector repeats most often.
    keys = {}
    for rnd, w in by_round.items():
        k = tuple(np.round(w, 12))
        keys.setdefault(k, []).append(rnd)
    modal_key, modal_rounds = max(keys.items(), key=lambda kv: len(kv[1]))
    if len(modal_rounds) < 2:
        # Nothing repeats: no fallback rounds detectable.
        return set()
    modal = np.array(modal_key)
    return {rnd for rnd, w in by_round.items()
            if np.allclose(w, modal, atol=SIZE_MATCH_TOL)}


def main():
    ap = argparse.ArgumentParser(
        description="Jain's fairness index on Shapley aggregation weights")
    ap.add_argument("--csv", default=CSV_IN, help=f"results CSV (default {CSV_IN})")
    ap.add_argument("--log", default=LOG_IN, help=f"round log (default {LOG_IN})")
    ap.add_argument("--suffix", default="", help="suffix for output filenames")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"Could not find {args.csv} in the current folder. "
            "Run this from your Shapley-FL project folder, after "
            "shapley_weighted_fedavg.py has produced results.")

    df = pd.read_csv(args.csv)
    required = {"round", "client_id", "aggregation_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing expected columns: {missing}")

    rounds = sorted(df["round"].unique())
    truncated, source = load_truncated_flags(args.log, rounds)
    if truncated is None:
        inferred_note = source
        truncated = infer_truncated_from_weights(df)
        source = inferred_note
    n_rounds = len(rounds)

    results = []
    for rnd, grp in df.groupby("round"):
        w = grp["aggregation_weight"].to_numpy()
        results.append({
            "round": int(rnd),
            "n_clients": len(w),
            "jain_index": jain_index(w),
            "truncated": int(int(rnd) in truncated),
        })

    out = pd.DataFrame(results).sort_values("round").reset_index(drop=True)

    txt_out = TXT_OUT.replace(".txt", f"{args.suffix}.txt")
    csv_out = CSV_OUT.replace(".csv", f"{args.suffix}.csv")
    png_out = PNG_OUT.replace(".png", f"{args.suffix}.png")
    out.to_csv(csv_out, index=False)

    live = out[out["truncated"] == 0]
    trunc = out[out["truncated"] == 1]
    n_clients = int(out["n_clients"].iloc[0])
    floor = 1.0 / n_clients

    # Pooled J over all client-rounds (kept for continuity with the
    # 5-round report, but no longer the headline).
    overall = jain_index(df["aggregation_weight"].to_numpy())

    lines = []
    lines.append("Jain's Fairness Index on Shapley aggregation weights")
    lines.append("=" * 60)
    lines.append("J = (sum w_i)^2 / (n * sum w_i^2)")
    lines.append(f"Range: {floor:.4f} (one client dominates) .. 1.0 "
                 f"(perfectly equal)")
    lines.append(f"Source: {args.csv}  ({n_rounds} rounds, {n_clients} clients)")
    lines.append(f"Truncation: {source}")
    lines.append("")

    lines.append("HEADLINE — Shapley-live rounds only")
    lines.append("-" * 60)
    if len(live):
        lines.append(f"  live rounds        : {len(live)}/{n_rounds} "
                     f"({100.0 * len(live) / n_rounds:.0f}%)")
        lines.append(f"  mean J (live)      : {live['jain_index'].mean():.4f}")
        lines.append(f"  min / max J (live) : {live['jain_index'].min():.4f} "
                     f"/ {live['jain_index'].max():.4f}")
        lines.append(f"  std  J (live)      : {live['jain_index'].std():.4f}")
        lines.append(f"  live round numbers : {live['round'].tolist()}")
    else:
        lines.append("  NO live rounds — Shapley weighting never activated. "
                     "Every round fell back to size weights.")
    lines.append("")

    lines.append("Reference — GTG-truncated rounds (size-weight fallback)")
    lines.append("-" * 60)
    if len(trunc):
        lines.append(f"  truncated rounds   : {len(trunc)}/{n_rounds} "
                     f"({100.0 * len(trunc) / n_rounds:.0f}%)")
        lines.append(f"  mean J (truncated) : {trunc['jain_index'].mean():.4f}")
        lines.append("  These rounds use size-proportional FedAvg weights, so "
                     "their J measures")
        lines.append("  how evenly HAR client sizes are distributed — not "
                     "anything about Shapley.")
        lines.append("  Treat this as the vanilla-FedAvg reference point.")
    else:
        lines.append("  none — Shapley signal was live in every round.")
    lines.append("")

    lines.append("Pooled / legacy figures")
    lines.append("-" * 60)
    lines.append(f"  mean J across ALL rounds : {out['jain_index'].mean():.4f}")
    lines.append(f"  pooled J (all rows)      : {overall:.4f}")
    lines.append("  Note: the all-rounds mean mixes Shapley-weighted and "
                 "size-weighted rounds and")
    lines.append("  is dominated by whichever group is larger. Quote the "
                 "live-round mean instead.")
    lines.append("")

    lines.append("Per-round detail")
    lines.append("-" * 60)
    if n_rounds <= MAX_LISTED_ROUNDS:
        for _, row in out.iterrows():
            tag = "  [truncated]" if row["truncated"] else ""
            lines.append(f"  Round {int(row['round']):>3}: "
                         f"J = {row['jain_index']:.4f}{tag}")
    else:
        lines.append(f"  ({n_rounds} rounds — full table in {csv_out}; "
                     f"live rounds listed below)")
        for _, row in live.iterrows():
            lines.append(f"  Round {int(row['round']):>3}: "
                         f"J = {row['jain_index']:.4f}")
    lines.append("")
    lines.append("Interpretation: within the rounds where Shapley weighting is "
                 "actually active, J")
    lines.append("near 1.0 means the contribution-derived weights stay spread "
                 "across the 30 HAR")
    lines.append("clients rather than collapsing onto a few dominant ones — "
                 "i.e. the method")
    lines.append("reweights by contribution without starving anyone out of "
                 "the aggregate.")

    text = "\n".join(lines)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # ---- plot: J vs round, truncated rounds shaded ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for rnd in trunc["round"]:
        ax.axvspan(rnd - 0.5, rnd + 0.5, color="#d62728", alpha=0.07,
                   linewidth=0)
    if len(trunc):
        ax.plot([], [], "s", color="#d62728", alpha=0.35, markersize=9,
                label=f"GTG-truncated ({len(trunc)}/{n_rounds}, size weights)")

    ax.plot(out["round"], out["jain_index"], "o-", markersize=4,
            linewidth=1.6, color="#1f77b4", label="Jain's index J")
    if len(live):
        ax.axhline(live["jain_index"].mean(), color="#2ca02c",
                   linestyle="--", linewidth=1.4,
                   label=f"mean J, live rounds = {live['jain_index'].mean():.4f}")
    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Global round")
    ax.set_ylabel("Jain's fairness index")
    ax.set_title(f"Fairness of Shapley aggregation weights — "
                 f"{n_clients} clients, {n_rounds} rounds")
    ax.set_ylim(min(0.5, out["jain_index"].min() - 0.05), 1.02)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(png_out, dpi=150)
    plt.close(fig)

    print(text)
    print()
    print(f"Wrote {txt_out}, {csv_out}, {png_out}")


if __name__ == "__main__":
    main()
