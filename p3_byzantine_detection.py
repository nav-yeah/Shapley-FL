"""
P3 — Byzantine Detection via Score Drift.

This script stays self-contained so the existing repo files do not need
to be modified. It does three things:
  1. Loads the clean per-round Shapley baseline from shapley_scores.csv.
  2. Recomputes per-round Shapley values for scenario_4_noisy_labels.pkl
     using the same GTG-Shapley logic as export_shapley_csv.py.
  3. Applies a drift detector that uses rolling variance, normalized trend
      slope, and a temporal z-score built from standardized residuals
      against the clean per-round baseline, then requires sustained
      anomalies across consecutive eligible rounds before flagging a client.

Outputs:
  - byzantine_detection_results.csv
  - byzantine_detection_metrics.txt
  - byzantine_scenario4_flagged.csv          (P4 handoff: scenario_4 run
    with real injected attacks, 10 clients / 50 rows)
  - byzantine_detection_results_merged.csv   (P4 handoff: full 30-client /
    150-row file, with the 10 scenario_4 clients' real evaluated results
    overlaid on the clean baseline for the other 20. See the
    `evaluated_for_attack` column — only rows with 1 were actually tested
    against an injected attack; rows with 0 are untested, not "tested
    and clean".)

The detector is tuned on the real clean per-round ranges in
shapley_scores.csv, and the evaluation uses the genuine scenario_4 noisy
label injection built from the repository's own HAR preprocessing.
"""

from __future__ import annotations

import csv
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fl_utils import (
    get_model,
    local_train,
    fedavg,
    evaluate,
    clone_state_dict,
    LOCAL_EPOCHS,
    LOCAL_LR,
)


BASELINE_CSV = Path("shapley_scores.csv")
DETECTION_INPUT_CSV = Path("shapley_scores.csv")
SCENARIO_PATH = Path("scenarios") / "scenario_4_noisy_labels.pkl"
OUT_CSV = Path("byzantine_detection_results.csv")
OUT_TXT = Path("byzantine_detection_metrics.txt")
OUT_PNG = Path("byzantine_detection_summary.png")
OUT_SCENARIO_CSV = Path("byzantine_scenario4_flagged.csv")
OUT_MERGED_CSV = Path("byzantine_detection_results_merged.csv")

GLOBAL_SEED = 42
RANDOM_SEED = 42
N_ROUNDS = 5

N_PERMUTATIONS = 200
EPSILON_B = 0.02
EPSILON_I = 0.005
M_GUIDED = 3
CONVERGENCE_WINDOW = 20
CONVERGENCE_THRESHOLD = 0.05

WINDOW_SIZE = 3
MIN_SUSTAINED_RATIO = 0.30
TEMPORAL_Z_WINDOW = 2
TEMPORAL_Z_THRESHOLD = 2.0
SLOPE_DENOM_FLOOR = 0.01

STD_MULTIPLIER = 1.0

SEVERITY_BY_POSITION = [0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20]


@dataclass(frozen=True)
class Thresholds:
    round_means: dict
    round_stds: dict


def load_client_data(pkl_path: Path):
    with pkl_path.open("rb") as handle:
        return pickle.load(handle)


def guided_permutation(client_ids, m, k, rng):
    n = len(client_ids)
    m = min(m, n)
    rotated = client_ids[k % n :] + client_ids[: k % n]
    guided_part = rotated[:m]
    remaining = [cid for cid in client_ids if cid not in guided_part]
    rng.shuffle(remaining)
    return guided_part + remaining


def gtg_shapley_one_round(
    client_ids,
    client_states,
    client_sizes,
    global_state_before,
    global_state_after,
    X_test,
    y_test,
    n_permutations=N_PERMUTATIONS,
    m=M_GUIDED,
    eps_b=EPSILON_B,
    eps_i=EPSILON_I,
    seed=RANDOM_SEED,
):
    """Single-round GTG-Shapley estimate for the local contribution signal."""
    rng = np.random.RandomState(seed)
    n = len(client_ids)

    v0 = evaluate(global_state_before, X_test, y_test)
    vN = evaluate(global_state_after, X_test, y_test)

    phi = {cid: 0.0 for cid in client_ids}

    if abs(vN - v0) <= eps_b:
        return phi, v0, vN

    phi_history = []
    for k in range(1, n_permutations + 1):
        pi_k = guided_permutation(client_ids, m, k, rng)
        v_prev = v0
        evaluated_subset = []

        for j in range(1, n + 1):
            evaluated_subset.append(pi_k[j - 1])

            if abs(vN - v_prev) >= eps_i:
                sub_states = [client_states[cid] for cid in evaluated_subset]
                sub_weights = [client_sizes[cid] for cid in evaluated_subset]
                recon_state = fedavg(sub_states, sub_weights)
                v_j = evaluate(recon_state, X_test, y_test)
            else:
                v_j = v_prev

            marginal = v_j - v_prev
            cid_j = pi_k[j - 1]
            phi[cid_j] = ((k - 1) / k) * phi[cid_j] + (1 / k) * marginal
            v_prev = v_j

        phi_history.append(dict(phi))

        if k >= max(CONVERGENCE_WINDOW + 1, 30):
            old_phi = phi_history[k - 1 - CONVERGENCE_WINDOW]
            diffs = []
            for cid in client_ids:
                denom = abs(phi[cid]) if abs(phi[cid]) > 1e-9 else 1e-9
                diffs.append(abs(phi[cid] - old_phi[cid]) / denom)
            if np.mean(diffs) < CONVERGENCE_THRESHOLD:
                break

    return phi, v0, vN


def train_all_clients(client_ids, data, global_state, round_idx, seed=GLOBAL_SEED):
    client_states = {}
    for cid in client_ids:
        local_model = get_model()
        local_model.load_state_dict(global_state)
        client_states[cid] = local_train(
            local_model,
            data[cid]["X_train"],
            data[cid]["y_train"],
            epochs=LOCAL_EPOCHS,
            lr=LOCAL_LR,
            seed=seed + round_idx * 100 + cid,
        )
    return client_states


def run_scenario_rounds(scenario_path: Path):
    scenario = load_client_data(scenario_path)
    client_ids = list(scenario["client_ids"])
    data = scenario["data"]

    X_test = np.concatenate([data[cid]["X_test"] for cid in client_ids])
    y_test = np.concatenate([data[cid]["y_test"] for cid in client_ids])
    client_sizes = {cid: len(data[cid]["y_train"]) for cid in client_ids}

    global_state = clone_state_dict(get_model(seed=GLOBAL_SEED).state_dict())
    rows = []

    for round_idx in range(1, N_ROUNDS + 1):
        client_states = train_all_clients(client_ids, data, global_state, round_idx)
        states = [client_states[cid] for cid in client_ids]
        weights = [client_sizes[cid] for cid in client_ids]
        new_global_state = fedavg(states, weights)

        phi_round, v0, vN = gtg_shapley_one_round(
            client_ids,
            client_states,
            client_sizes,
            global_state,
            new_global_state,
            X_test,
            y_test,
        )

        for cid in client_ids:
            rows.append(
                {
                    "round": round_idx,
                    "client_id": cid,
                    "shapley_value": float(phi_round[cid]),
                    "model_accuracy": float(vN),
                }
            )

        global_state = new_global_state

    return pd.DataFrame(rows), client_ids


def load_baseline_series(baseline_csv: Path):
    df = pd.read_csv(baseline_csv)
    df = df.sort_values(["client_id", "round"]).reset_index(drop=True)
    return df


def load_detection_series(detection_csv: Path):
    df = pd.read_csv(detection_csv)
    required = {"round", "client_id", "shapley_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {detection_csv}: {sorted(missing)}"
        )
    df = df.sort_values(["client_id", "round"]).reset_index(drop=True)
    return df


def rolling_window_features(
    values: np.ndarray,
    window_size: int,
    round_numbers=None,
    round_means=None,
    round_stds=None,
):
    records = []
    for index in range(len(values)):
        round_number = index + 1
        current = float(values[index])

        if index + 1 < window_size:
            if (
                round_numbers is not None
                and round_means is not None
                and round_stds is not None
            ):
                round_number = int(round_numbers[index])
                z_score = (current - round_means[round_number]) / (
                    round_stds[round_number] + 1e-6
                )
            else:
                z_score = 0.0
            records.append(
                {
                    "round": round_number,
                    "rolling_variance": np.nan,
                    "trend_slope": np.nan,
                    "z_score": z_score,
                    "raw_anomaly": False,
                }
            )
            continue

        window = values[index - window_size + 1 : index + 1]
        window_mean = float(np.mean(window))
        rolling_variance = float(np.var(window, ddof=0))

        x = np.arange(window_size, dtype=float)
        slope = float(np.polyfit(x, window, 1)[0])
        # Avoid unstable slope inflation when the window mean is near zero.
        normalized_slope = slope / max(abs(window_mean), SLOPE_DENOM_FLOOR)

        if (
            round_numbers is not None
            and round_means is not None
            and round_stds is not None
        ):
            round_number = int(round_numbers[index])
            z_score = (current - round_means[round_number]) / (
                round_stds[round_number] + 1e-6
            )
        else:
            z_score = 0.0

        records.append(
            {
                "round": round_number,
                "rolling_variance": rolling_variance,
                "trend_slope": normalized_slope,
                "z_score": z_score,
                "raw_anomaly": False,
            }
        )

    return records


def calibrate_thresholds(baseline_df: pd.DataFrame, window_size: int):
    round_means = {}
    round_stds = {}

    for round_number, round_frame in baseline_df.groupby("round"):
        values = round_frame["shapley_value"].to_numpy(dtype=float)
        round_means[int(round_number)] = float(np.mean(values))
        round_stds[int(round_number)] = float(np.std(values, ddof=0))

    return Thresholds(round_means=round_means, round_stds=round_stds)


def apply_detector(
    df: pd.DataFrame, thresholds: Thresholds, window_size: int, sustain_ratio: float
):
    output_rows = []

    for client_id, client_frame in df.groupby("client_id"):
        client_frame = client_frame.sort_values("round").reset_index(drop=True)
        values = client_frame["shapley_value"].to_numpy(dtype=float)
        round_numbers = client_frame["round"].to_numpy(dtype=int)
        feature_rows = rolling_window_features(
            values,
            window_size,
            round_numbers=round_numbers,
            round_means=thresholds.round_means,
            round_stds=thresholds.round_stds,
        )

        temporal_z_scores = []
        for index, feature_row in enumerate(feature_rows):
            current_z = feature_row["z_score"]
            if index + 1 < TEMPORAL_Z_WINDOW:
                temporal_z_scores.append(current_z)
            else:
                window_scores = [
                    row["z_score"]
                    for row in feature_rows[
                        max(0, index - TEMPORAL_Z_WINDOW + 1) : index + 1
                    ]
                ]
                temporal_z_scores.append(float(np.mean(window_scores)))

        method_flags = []
        naive_flags = []
        for index, feature_row in enumerate(feature_rows):
            single_round_z = feature_row["z_score"]
            naive_flags.append(abs(single_round_z) >= TEMPORAL_Z_THRESHOLD)
            method_flags.append(abs(temporal_z_scores[index]) >= TEMPORAL_Z_THRESHOLD)

        eligible_rounds = len(feature_rows)
        sustain_count = max(1, int(np.ceil(sustain_ratio * eligible_rounds)))

        sustained_flags = [False] * len(method_flags)
        cumulative_anomalies = 0
        is_client_flagged = False
        for index, is_anomalous in enumerate(method_flags):
            if is_anomalous:
                cumulative_anomalies += 1
            if cumulative_anomalies >= sustain_count:
                is_client_flagged = True
            sustained_flags[index] = bool(is_client_flagged)

        for index, feature_row in enumerate(feature_rows):
            output_rows.append(
                {
                    "round": int(client_frame.loc[index, "round"]),
                    "client_id": int(client_id),
                    "flagged_status": int(sustained_flags[index]),
                    "rolling_variance": float(feature_row["rolling_variance"])
                    if not np.isnan(feature_row["rolling_variance"])
                    else np.nan,
                    "trend_slope": float(feature_row["trend_slope"])
                    if not np.isnan(feature_row["trend_slope"])
                    else np.nan,
                    "z_score": float(temporal_z_scores[index]),
                    "raw_anomaly": int(naive_flags[index]),
                }
            )

    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(["round", "client_id"]).reset_index(drop=True)
    return output_df


def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int(np.sum(y_true == 1)),
    }


def row_level_metrics(result_df: pd.DataFrame, client_ids, window_size: int):
    client_to_position = {cid: position for position, cid in enumerate(client_ids)}
    eligible = result_df[result_df["round"] >= window_size].copy()
    eligible["ground_truth"] = eligible["client_id"].map(
        lambda cid: int(SEVERITY_BY_POSITION[client_to_position[int(cid)]] > 0.0)
    )
    method = precision_recall_f1(eligible["ground_truth"], eligible["flagged_status"])
    naive = precision_recall_f1(eligible["ground_truth"], eligible["raw_anomaly"])
    return method, naive, eligible


def client_level_metrics(result_df: pd.DataFrame, client_ids):
    client_to_position = {cid: position for position, cid in enumerate(client_ids)}
    rows = []
    for cid in client_ids:
        client_rows = result_df[result_df["client_id"] == cid]
        detected = int(client_rows["flagged_status"].any())
        naive_detected = int(client_rows["raw_anomaly"].any())
        truth = int(SEVERITY_BY_POSITION[client_to_position[int(cid)]] > 0.0)
        rows.append(
            {
                "client_id": int(cid),
                "severity": SEVERITY_BY_POSITION[client_to_position[int(cid)]],
                "truth": truth,
                "method_detected": detected,
                "naive_detected": naive_detected,
            }
        )

    summary = pd.DataFrame(rows)
    return summary


def severity_summary(client_summary: pd.DataFrame):
    rows = []
    for severity, subset in client_summary.groupby("severity"):
        truth = subset["truth"].to_numpy(dtype=int)
        method_pred = subset["method_detected"].to_numpy(dtype=int)
        naive_pred = subset["naive_detected"].to_numpy(dtype=int)
        method_metrics = precision_recall_f1(truth, method_pred)
        naive_metrics = precision_recall_f1(truth, naive_pred)
        rows.append(
            {
                "severity": severity,
                "method_precision": method_metrics["precision"],
                "method_recall": method_metrics["recall"],
                "method_f1": method_metrics["f1"],
                "naive_precision": naive_metrics["precision"],
                "naive_recall": naive_metrics["recall"],
                "naive_f1": naive_metrics["f1"],
                "n_clients": len(subset),
            }
        )
    return pd.DataFrame(rows).sort_values("severity")


def plot_severity_summary(severity_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(severity_df))
    width = 0.36

    ax.bar(
        x - width / 2,
        severity_df["method_recall"],
        width,
        label="Sustained drift detector",
    )
    ax.bar(
        x + width / 2,
        severity_df["naive_recall"],
        width,
        label="Naive single-round threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sev:.2f}" for sev in severity_df["severity"]])
    ax.set_xlabel("Injected label-flip severity")
    ax.set_ylabel("Client-level recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("P3 Byzantine detection: recall by attack severity")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)


def format_metric_line(name: str, metrics: dict):
    return (
        f"{name:<10s} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"support={metrics['support']} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}"
    )


def main():
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_CSV}. Run export_shapley_csv.py first or keep the repo's committed CSV."
        )
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SCENARIO_PATH}. Run build_scenarios.py after preprocessing the HAR dataset."
        )
    if not DETECTION_INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {DETECTION_INPUT_CSV}. Run export_shapley_csv.py first or keep the committed CSV."
        )

    start = time.time()
    baseline_df = load_baseline_series(BASELINE_CSV)
    thresholds = calibrate_thresholds(baseline_df, WINDOW_SIZE)

    # Deliverable CSV for P4: run detector over the full 30-client handoff file.
    detection_df = load_detection_series(DETECTION_INPUT_CSV)
    deliverable_df = apply_detector(
        detection_df, thresholds, WINDOW_SIZE, MIN_SUSTAINED_RATIO
    )
    output_df = deliverable_df.drop(columns=["raw_anomaly"]).sort_values(
        ["round", "client_id"]
    )
    output_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    # Evaluation with known labels: run on scenario_4_noisy_labels (10 clients).
    scenario_df, client_ids = run_scenario_rounds(SCENARIO_PATH)
    metrics_df = apply_detector(
        scenario_df, thresholds, WINDOW_SIZE, MIN_SUSTAINED_RATIO
    )
    method_metrics, naive_metrics, eligible_df = row_level_metrics(
        metrics_df, client_ids, WINDOW_SIZE
    )
    client_summary = client_level_metrics(metrics_df, client_ids)
    severity_df = severity_summary(client_summary)
    plot_severity_summary(severity_df)

    # ------------------------------------------------------------------
    # NEW (added for P4): export the scenario_4 run — which has real
    # injected attacks and real flagged_status=1 rows — as its own CSV.
    # metrics_df doesn't carry shapley_value (apply_detector only outputs
    # detector features), so merge it back in from scenario_df first.
    # ------------------------------------------------------------------
    scenario_flagged_df = metrics_df.drop(columns=["raw_anomaly"]).merge(
        scenario_df[["round", "client_id", "shapley_value"]],
        on=["round", "client_id"],
        how="left",
    )
    scenario_flagged_df = (
        scenario_flagged_df[
            [
                "round",
                "client_id",
                "shapley_value",
                "flagged_status",
                "rolling_variance",
                "trend_slope",
                "z_score",
            ]
        ]
        .sort_values(["round", "client_id"])
        .reset_index(drop=True)
    )
    scenario_flagged_df.to_csv(OUT_SCENARIO_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    # ------------------------------------------------------------------
    # NEW (added for P4): a 150-row (30-client) file for P4's full-scale
    # ledger, with the 10 scenario_4 clients' REAL evaluated detection
    # results overlaid on top of the clean 30-client baseline.
    #
    # IMPORTANT — this does NOT mean all 30 clients were attack-tested.
    # Only the 10 scenario_4 clients (real client IDs: see
    # scenario_client_ids below) were actually run against injected
    # attacks. The other 20 clients' flagged_status=0 reflects that they
    # were never evaluated for attacks in this run, same as in
    # byzantine_detection_results.csv — it is not a claim they were
    # tested and found clean. An `evaluated_for_attack` column marks
    # this distinction explicitly so downstream consumers (P4, the
    # paper) don't misread coverage.
    # ------------------------------------------------------------------
    scenario_client_ids = set(
        int(c) for c in client_ids
    )  # the 10 real scenario_4 clients

    merged_df = deliverable_df.copy()  # the 150-row clean-baseline detector output
    merged_df["evaluated_for_attack"] = (
        merged_df["client_id"].isin(scenario_client_ids).astype(int)
    )

    # Merge shapley_value onto the baseline rows so the merged file has it too.
    merged_df = merged_df.merge(
        detection_df[["round", "client_id", "shapley_value"]],
        on=["round", "client_id"],
        how="left",
    )

    # Overlay the real scenario_4 detector results for the 10 evaluated clients.
    overlay = scenario_flagged_df.copy()
    overlay["evaluated_for_attack"] = 1
    overlay_indexed = overlay.set_index(["round", "client_id"])

    merged_indexed = merged_df.set_index(["round", "client_id"])
    overlay_cols = [
        "shapley_value",
        "flagged_status",
        "rolling_variance",
        "trend_slope",
        "z_score",
    ]
    for col in overlay_cols:
        merged_indexed.loc[overlay_indexed.index, col] = overlay_indexed[col]
    merged_indexed.loc[overlay_indexed.index, "evaluated_for_attack"] = 1

    merged_df = merged_indexed.reset_index()
    merged_df = (
        merged_df[
            [
                "round",
                "client_id",
                "shapley_value",
                "flagged_status",
                "rolling_variance",
                "trend_slope",
                "z_score",
                "evaluated_for_attack",
            ]
        ]
        .sort_values(["round", "client_id"])
        .reset_index(drop=True)
    )
    merged_df.to_csv(OUT_MERGED_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    # ------------------------------------------------------------------
    # END NEW
    # ------------------------------------------------------------------

    lines = [
        "P3 Byzantine detection via score drift",
        "",
        f"Baseline calibration source: {BASELINE_CSV}",
        f"Scenario evaluated: {SCENARIO_PATH}",
        f"Window size: {WINDOW_SIZE}, sustained ratio: {MIN_SUSTAINED_RATIO:.2f}",
        f"Thresholds calibrated from clean data: temporal z threshold={TEMPORAL_Z_THRESHOLD:.1f}, sustain ratio={MIN_SUSTAINED_RATIO:.2f} ({int(np.ceil(MIN_SUSTAINED_RATIO * N_ROUNDS))} anomalous rounds required)",
        f"Slope normalization floor: {SLOPE_DENOM_FLOOR:.2f}",
        f"Window behavior: rolling_variance/trend_slope are NaN for rounds < {WINDOW_SIZE} by design.",
        f"Eligible rows for row-level metrics: {len(eligible_df)}",
        "",
        format_metric_line("Method", method_metrics),
        format_metric_line("Naive", naive_metrics),
        "",
        "Severity-bucket client precision/recall/F1 (method vs naive):",
        "severity | method_p method_r method_f1 | naive_p naive_r naive_f1 | n_clients",
    ]

    for _, row in severity_df.iterrows():
        lines.append(
            f"  {row['severity']:.2f} | {row['method_precision']:.4f} {row['method_recall']:.4f} {row['method_f1']:.4f} | "
            f"{row['naive_precision']:.4f} {row['naive_recall']:.4f} {row['naive_f1']:.4f} | {int(row['n_clients'])}"
        )

    lines.extend(
        [
            "",
            f"Rows written to {OUT_CSV} ({len(output_df)} rows from full 30-client detection input)",
            f"Scenario-4 flagged rows written to {OUT_SCENARIO_CSV} ({len(scenario_flagged_df)} rows, "
            f"{int(scenario_flagged_df['flagged_status'].sum())} flagged)",
            f"Merged 30-client rows written to {OUT_MERGED_CSV} ({len(merged_df)} rows, "
            f"{int(merged_df['flagged_status'].sum())} flagged, "
            f"{int(merged_df['evaluated_for_attack'].sum())} rows attack-evaluated)",
            f"Summary plot written to {OUT_PNG}",
            f"Total runtime: {time.time() - start:.2f}s",
            "",
            "Client-level summary:",
        ]
    )

    for _, row in client_summary.iterrows():
        lines.append(
            f"  client={int(row['client_id'])} severity={row['severity']:.2f} "
            f"truth={int(row['truth'])} method_detected={int(row['method_detected'])} "
            f"naive_detected={int(row['naive_detected'])}"
        )

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
