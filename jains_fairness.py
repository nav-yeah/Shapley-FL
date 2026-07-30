"""
P2 - Jain's Fairness Index on Shapley aggregation weights
==========================================================
Computes Jain's fairness index J = (sum(w_i))^2 / (n * sum(w_i^2))
for the per-round aggregation weights already stored in aggregation_results.csv.

J ranges from 1/n (maximally unfair: one client gets all weight) to 1.0
(perfectly fair: every client weighted equally). Higher = more evenly the
Shapley-derived aggregation weight is spread across clients.

Reads : aggregation_results.csv  (locked schema:
        round, client_id, aggregation_weight,
        global_accuracy_vanilla, global_accuracy_weighted)
Writes: jains_fairness.txt   (human-readable summary)
        jains_fairness.csv   (round, n_clients, jain_index)

Run:  python jains_fairness.py
"""

import os
import pandas as pd

CSV_IN = "aggregation_results.csv"
TXT_OUT = "jains_fairness.txt"
CSV_OUT = "jains_fairness.csv"


def jain_index(weights):
    """Jain's fairness index for a 1-D array-like of weights."""
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    n = len(weights)
    if s2 == 0:
        # All-zero round = no signal; fairness is undefined, return NaN.
        return float("nan")
    return float((s1 ** 2) / (n * s2))


def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(
            f"Could not find {CSV_IN} in the current folder. "
            "Run this from your Shapley-FL project folder."
        )

    df = pd.read_csv(CSV_IN)

    required = {"round", "client_id", "aggregation_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{CSV_IN} is missing expected columns: {missing}")

    results = []
    for rnd, grp in df.groupby("round"):
        w = grp["aggregation_weight"].to_numpy()
        j = jain_index(w)
        results.append(
            {"round": int(rnd), "n_clients": len(w), "jain_index": j}
        )

    out = pd.DataFrame(results).sort_values("round").reset_index(drop=True)

    # Overall across all client-rounds (treats every weight equally).
    overall = jain_index(df["aggregation_weight"].to_numpy())

    out.to_csv(CSV_OUT, index=False)

    lines = []
    lines.append("Jain's Fairness Index on Shapley aggregation weights")
    lines.append("=" * 55)
    lines.append("J = (sum w_i)^2 / (n * sum w_i^2)")
    lines.append("Range: 1/n (one client dominates) .. 1.0 (perfectly equal)")
    lines.append("")
    for _, row in out.iterrows():
        lower = 1.0 / row["n_clients"]
        lines.append(
            f"Round {int(row['round'])}: "
            f"J = {row['jain_index']:.4f}  "
            f"(n = {int(row['n_clients'])}, floor = {lower:.4f})"
        )
    lines.append("")
    lines.append(f"Mean J across rounds : {out['jain_index'].mean():.4f}")
    lines.append(f"Pooled J (all rows)  : {overall:.4f}")
    lines.append("")
    lines.append(
        "Interpretation: J near 1.0 means the Shapley-derived weights are "
        "spread evenly across the 30 HAR clients; a lower J means a few "
        "clients dominate the aggregate. Track this alongside accuracy to "
        "show fairness is preserved while weighting by contribution."
    )
    text = "\n".join(lines)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)
    print()
    print(f"Wrote {TXT_OUT} and {CSV_OUT}")


if __name__ == "__main__":
    main()
