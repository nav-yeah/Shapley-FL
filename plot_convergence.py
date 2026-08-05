"""
P2 - Convergence curve: accuracy vs round (vanilla vs Shapley-weighted)
=======================================================================
Plots global accuracy against round number for both aggregation variants
so the round-by-round trend is visible at a glance (better than a table).

The accuracy columns are per-round constants repeated across all client
rows, so we collapse to one value per round before plotting.

50-ROUND UPDATE
---------------
Three changes were needed once the pipeline went from 5 to 50 rounds:

1. A marker on every round and a tick on every round is unreadable at 50
   points, so markers and ticks are now spaced automatically.

2. GTG-truncated rounds are shaded. In a truncated round the weighted arm
   falls back to size-proportional weights, so it is running plain
   FedAvg — the reader needs to see where the two methods are and are not
   actually different. Truncation flags come from aggregation_round_log.csv
   when present; otherwise the shading is skipped and the plot says so.

3. An inset zooms into the plateau (the last third of training), because
   after convergence the two curves differ by fractions of a point and
   the full-range axis flattens that difference into invisibility.

The final-round gap is no longer the headline annotation. After the model
plateaus, a single round is noisy — one lucky round can flip the sign of
the gap. The annotation now reports the mean gap over the last 10 rounds
alongside the final-round value.

Reads : aggregation_results.csv  (locked schema:
        round, client_id, aggregation_weight,
        global_accuracy_vanilla, global_accuracy_weighted)
        aggregation_round_log.csv  (optional; for truncation shading)
Writes: convergence_curve.png

Run:  python plot_convergence.py
      python plot_convergence.py --csv aggregation_results_adaptive.csv \
                                 --log aggregation_round_log_adaptive.csv \
                                 --suffix _adaptive
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to file
import matplotlib.pyplot as plt

CSV_IN = "aggregation_results.csv"
LOG_IN = "aggregation_round_log.csv"
PNG_OUT = "convergence_curve.png"

# Trailing rounds averaged for the plateau-gap annotation.
PLATEAU_WINDOW = 10

# Only draw the plateau inset when there are enough rounds for it to mean
# something.
INSET_MIN_ROUNDS = 20


def load_truncated(log_path):
    """Return a set of truncated round numbers, or None if unavailable."""
    if not os.path.exists(log_path):
        return None
    log = pd.read_csv(log_path)
    if "truncated" not in log.columns or "round" not in log.columns:
        return None
    return set(log.loc[log["truncated"] == 1, "round"].astype(int).tolist())


def main():
    ap = argparse.ArgumentParser(
        description="Convergence curve: vanilla vs Shapley-weighted FedAvg")
    ap.add_argument("--csv", default=CSV_IN, help=f"results CSV (default {CSV_IN})")
    ap.add_argument("--log", default=LOG_IN, help=f"round log (default {LOG_IN})")
    ap.add_argument("--suffix", default="", help="suffix for output filename")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"Could not find {args.csv} in the current folder. "
            "Run this from your Shapley-FL project folder.")

    df = pd.read_csv(args.csv)
    required = {"round", "global_accuracy_vanilla", "global_accuracy_weighted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing expected columns: {missing}")

    # One accuracy value per round (columns are constant within a round).
    per_round = (
        df.groupby("round")[["global_accuracy_vanilla",
                             "global_accuracy_weighted"]]
        .mean()
        .reset_index()
        .sort_values("round")
    )

    rounds = per_round["round"].to_numpy()
    vanilla = per_round["global_accuracy_vanilla"].to_numpy()
    weighted = per_round["global_accuracy_weighted"].to_numpy()
    n_rounds = len(rounds)

    truncated = load_truncated(args.log)

    png_out = PNG_OUT.replace(".png", f"{args.suffix}.png")

    markevery = max(1, n_rounds // 12)
    tickstep = max(1, int(np.ceil(n_rounds / 10)))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if truncated:
        for r in sorted(truncated):
            ax.axvspan(r - 0.5, r + 0.5, color="#d62728", alpha=0.07,
                       linewidth=0)
        ax.plot([], [], "s", color="#d62728", alpha=0.35, markersize=9,
                label=f"GTG-truncated ({len(truncated)}/{n_rounds}) — "
                      f"weighted arm falls back to size weights")

    ax.plot(rounds, vanilla, marker="o", markevery=markevery, markersize=5,
            linewidth=1.8, color="#888888",
            label="Vanilla FedAvg (size-weighted)")
    ax.plot(rounds, weighted, marker="s", markevery=markevery, markersize=5,
            linewidth=1.8, color="#1f77b4", label="Shapley-weighted FedAvg")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global accuracy")
    ax.set_title("Convergence: Vanilla vs Shapley-weighted FedAvg "
                 "(UCI HAR, 30 clients)")
    ax.set_xticks(range(int(rounds.min()), int(rounds.max()) + 1, tickstep))
    ax.grid(True, linestyle="--", alpha=0.4)

    # ---- gap reporting: final round AND plateau mean ----
    pw = min(PLATEAU_WINDOW, n_rounds)
    final_gap = weighted[-1] - vanilla[-1]
    plateau_gap = float(np.mean(weighted[-pw:]) - np.mean(vanilla[-pw:]))
    ax.annotate(
        f"final-round gap: {final_gap:+.4f}\n"
        f"mean gap, last {pw} rounds: {plateau_gap:+.4f}",
        xy=(0.985, 0.06), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    if truncated is None:
        ax.annotate("(no aggregation_round_log.csv — truncated rounds "
                    "not shaded)",
                    xy=(0.015, 0.965), xycoords="axes fraction",
                    ha="left", va="top", fontsize=7.5, color="#999999")

    ax.legend(loc="lower right", fontsize=8.5)

    # ---- plateau inset ----
    if n_rounds >= INSET_MIN_ROUNDS:
        start = int(n_rounds * 2 / 3)
        axins = ax.inset_axes([0.44, 0.30, 0.34, 0.32])
        axins.plot(rounds[start:], vanilla[start:], "-", linewidth=1.4,
                   color="#888888")
        axins.plot(rounds[start:], weighted[start:], "-", linewidth=1.4,
                   color="#1f77b4")
        if truncated:
            for r in sorted(truncated):
                if r >= rounds[start]:
                    axins.axvspan(r - 0.5, r + 0.5, color="#d62728",
                                  alpha=0.07, linewidth=0)
        axins.set_title("plateau detail", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(png_out, dpi=150)
    plt.close(fig)

    # ---- console summary ----
    print("Per-round accuracy:")
    if n_rounds <= 15:
        print(per_round.to_string(index=False))
    else:
        print(per_round.head(5).to_string(index=False))
        print(f"  ... ({n_rounds - 10} rounds omitted) ...")
        print(per_round.tail(5).to_string(index=False))
    print()
    print(f"Final round {int(rounds[-1])}: vanilla={vanilla[-1]:.4f}  "
          f"weighted={weighted[-1]:.4f}  gap={final_gap:+.4f}")
    print(f"Mean over last {pw} rounds: vanilla={np.mean(vanilla[-pw:]):.4f}  "
          f"weighted={np.mean(weighted[-pw:]):.4f}  gap={plateau_gap:+.4f}")
    print(f"Best: vanilla={vanilla.max():.4f} (r{int(rounds[vanilla.argmax()])})"
          f"  weighted={weighted.max():.4f} "
          f"(r{int(rounds[weighted.argmax()])})")
    if truncated:
        print(f"Truncated rounds: {len(truncated)}/{n_rounds} — in these the "
              f"weighted arm is running plain size-weighted FedAvg.")
    print()
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
