"""
P3 — Byzantine Detection via Score Drift.

This script stays self-contained so the existing repo files do not need
to be modified. It does three things:
  1. Loads the clean per-round Shapley baseline from shapley_scores.csv.
  2. Recomputes per-round Shapley values for scenario_4_noisy_labels.pkl
     using the same GTG-Shapley logic as export_shapley_csv.py.
  3. Applies a drift detector that uses rolling variance, normalized trend
     slope, and a personal z-score, then requires sustained anomalies
     across consecutive eligible rounds before flagging a client.

Outputs:
  - byzantine_detection_results.csv
  - byzantine_detection_metrics.txt

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
SCENARIO_PATH = Path("scenarios") / "scenario_4_noisy_labels.pkl"
OUT_CSV = Path("byzantine_detection_results.csv")
OUT_TXT = Path("byzantine_detection_metrics.txt")
OUT_PNG = Path("byzantine_detection_summary.png")

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
MIN_SUSTAINED_WINDOWS = 2

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


def rolling_window_features(values: np.ndarray, window_size: int):
    records = []
    for index in range(len(values)):
        round_number = index + 1
        current = float(values[index])

        if index + 1 < window_size:
            records.append(
                {
                    "round": round_number,
                    "rolling_variance": np.nan,
                    "trend_slope": np.nan,
                    "z_score": np.nan,
                    "raw_anomaly": False,
                }
            )
            continue

        window = values[index - window_size + 1 : index + 1]
        window_mean = float(np.mean(window))
        rolling_variance = float(np.var(window, ddof=0))

        x = np.arange(window_size, dtype=float)
        slope = float(np.polyfit(x, window, 1)[0])
        normalized_slope = slope / (abs(window_mean) + 1e-3)

        history = values[:index]
        if len(history) > 1:
            history_mean = float(np.mean(history))
            history_std = float(np.std(history, ddof=0))
            z_score = (current - history_mean) / (history_std + 1e-6)
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


def apply_detector(df: pd.DataFrame, thresholds: Thresholds, window_size: int, sustain: int):
    output_rows = []

    for client_id, client_frame in df.groupby("client_id"):
        client_frame = client_frame.sort_values("round").reset_index(drop=True)
        values = client_frame["shapley_value"].to_numpy(dtype=float)
        feature_rows = rolling_window_features(values, window_size)

        raw_flags = []
        for index, feature_row in enumerate(feature_rows):
            round_number = int(client_frame.loc[index, "round"])
            if np.isnan(feature_row["rolling_variance"]):
                raw_flags.append(False)
                continue

            mean = thresholds.round_means[round_number]
            std = thresholds.round_stds[round_number]
            deviation = abs(values[index] - mean)
            is_anomalous = deviation > (STD_MULTIPLIER * std)
            raw_flags.append(is_anomalous)

        sustained_flags = [False] * len(raw_flags)
        for index, is_anomalous in enumerate(raw_flags):
            if not is_anomalous or index + 1 < sustain:
                continue
            if sum(raw_flags[max(0, index - 2): index + 1]) >= sustain:
                sustained_flags[index] = True

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
                    "z_score": float(feature_row["z_score"])
                    if not np.isnan(feature_row["z_score"])
                    else np.nan,
                    "raw_anomaly": int(raw_flags[index]),
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

    ax.bar(x - width / 2, severity_df["method_recall"], width, label="Sustained drift detector")
    ax.bar(x + width / 2, severity_df["naive_recall"], width, label="Naive single-round threshold")

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

    start = time.time()
    baseline_df = load_baseline_series(BASELINE_CSV)
    thresholds = calibrate_thresholds(baseline_df, WINDOW_SIZE)

    scenario_df, client_ids = run_scenario_rounds(SCENARIO_PATH)
    metrics_df = apply_detector(scenario_df, thresholds, WINDOW_SIZE, MIN_SUSTAINED_WINDOWS)
    output_df = metrics_df.drop(columns=["raw_anomaly"]).sort_values(["round", "client_id"])
    output_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    method_metrics, naive_metrics, eligible_df = row_level_metrics(metrics_df, client_ids, WINDOW_SIZE)
    client_summary = client_level_metrics(metrics_df, client_ids)
    severity_df = severity_summary(client_summary)
    plot_severity_summary(severity_df)

    lines = [
        "P3 Byzantine detection via score drift",
        "",
        f"Baseline calibration source: {BASELINE_CSV}",
        f"Scenario evaluated: {SCENARIO_PATH}",
        f"Window size: {WINDOW_SIZE}, sustained windows: {MIN_SUSTAINED_WINDOWS}",
        f"Thresholds calibrated from clean data: roundwise mean +/- {STD_MULTIPLIER:.1f} std",
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
            f"Rows written to {OUT_CSV} ({len(output_df)} rows)",
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