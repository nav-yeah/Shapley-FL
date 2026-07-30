"""
P2 - Convergence curve: accuracy vs round (vanilla vs Shapley-weighted)
=======================================================================
Plots global accuracy against round number for both aggregation variants
so the round-by-round trend is visible at a glance (better than a table).

The accuracy columns are per-round constants repeated across all 30 client
rows, so we collapse to one value per round before plotting.

Reads : aggregation_results.csv  (locked schema:
        round, client_id, aggregation_weight,
        global_accuracy_vanilla, global_accuracy_weighted)
Writes: convergence_curve.png

Designed to extend cleanly to 50 rounds later: nothing is hard-coded to 5.

Run:  python plot_convergence.py
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to file
import matplotlib.pyplot as plt

CSV_IN = "aggregation_results.csv"
PNG_OUT = "convergence_curve.png"


def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(
            f"Could not find {CSV_IN} in the current folder. "
            "Run this from your Shapley-FL project folder."
        )

    df = pd.read_csv(CSV_IN)

    required = {
        "round",
        "global_accuracy_vanilla",
        "global_accuracy_weighted",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{CSV_IN} is missing expected columns: {missing}")

    # One accuracy value per round (columns are constant within a round).
    per_round = (
        df.groupby("round")[
            ["global_accuracy_vanilla", "global_accuracy_weighted"]
        ]
        .mean()
        .reset_index()
        .sort_values("round")
    )

    rounds = per_round["round"].to_numpy()
    vanilla = per_round["global_accuracy_vanilla"].to_numpy()
    weighted = per_round["global_accuracy_weighted"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        rounds,
        vanilla,
        marker="o",
        linewidth=2,
        color="#888888",
        label="Vanilla FedAvg (size-weighted)",
    )
    ax.plot(
        rounds,
        weighted,
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Shapley-weighted FedAvg",
    )

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global accuracy")
    ax.set_title("Convergence: Vanilla vs Shapley-weighted FedAvg (UCI HAR, 30 clients)")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Annotate the final-round gap.
    final_gap = weighted[-1] - vanilla[-1]
    ax.annotate(
        f"Final gap: {final_gap:+.4f}",
        xy=(rounds[-1], weighted[-1]),
        xytext=(-10, 12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#1f77b4",
    )

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)

    print("Per-round accuracy:")
    print(per_round.to_string(index=False))
    print()
    print(f"Wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
