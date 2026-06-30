# ShapleyFL — Setup & Handoff Guide (P1: Shapley Computation Engine)

This document is the starting point for anyone running or building on the
Shapley computation engine. Read this before `experimentation_writeup.md`
(the results) or before building on `shapley_scores.csv`.

## 1. Environment Setup

Requires Python 3.12, a venv, and the following packages:

```cmd
python -m venv venv --copies
venv\Scripts\activate
pip install numpy==1.26.4 flwr==1.9.0 pandas matplotlib scikit-learn torch
```

If `pip install` fails on a managed/system Python (not relevant inside a
venv, but relevant if running outside one): add `--break-system-packages`.

## 2. Dataset

Download UCI HAR from:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract the zip, then copy the `UCI HAR Dataset` folder into:
`<project_root>/data/UCI HAR Dataset/`

## 3. Run Order (IMPORTANT — scripts depend on each other's outputs)

```cmd
python preprocess_har.py        # produces preprocessed/har_clients.pkl
python exact_shapley.py         # produces exact_shapley_results.pkl (ground truth, n=5, SLOW ~15s)
python gtg_shapley_approx.py    # validates approximation against ground truth (n=5)
python build_scenarios.py       # produces scenarios/*.pkl (5 distribution scenarios, n=10)
python run_scenarios.py         # runs GTG-Shapley on all 5 scenarios (SLOW, ~15-20 min total)
python export_shapley_csv.py    # produces shapley_scores.csv — THE FILE FOR P2/P3/P4 (fast, ~2 min)
```

`gtg_shapley_full_scale.py` (n=30, full-trajectory) is optional — it's a
one-time efficiency benchmark, took ~63 minutes to run, and is NOT needed
to produce `shapley_scores.csv`.

All scripts use `GLOBAL_SEED = 42` / `RANDOM_SEED = 42` for
reproducibility. Exact bit-identical results across machines are not
guaranteed (PyTorch numerics can vary slightly by platform/hardware) —
small floating-point differences (e.g. 0.1259 vs 0.1258) are normal, not
a bug.

## 4. CRITICAL: Two Different Shapley Computations Exist — Don't Confuse Them

This codebase produces Shapley values in TWO different ways, answering
TWO different questions. Mixing them up will cause confusion:

**(A) End-of-training Shapley** (`exact_shapley.py`, `gtg_shapley_approx.py`,
`gtg_shapley_full_scale.py`, `run_scenarios.py`)
- Question answered: "What is each client's total contribution across
  the ENTIRE 5-round training process?"
- One Shapley value per client (not per round).
- Expensive: each coalition requires a full 5-round retraining.
- This is what was used for the n=5 ground-truth validation and the
  n=30 / 5-scenario efficiency benchmarks reported in the writeup.

**(B) Per-round Shapley** (`export_shapley_csv.py` → `shapley_scores.csv`)
- Question answered: "What did each client contribute IN THIS SPECIFIC
  ROUND, given the model state at the start of that round?"
- One Shapley value per client PER ROUND (150 rows total: 30 clients x
  5 rounds).
- Cheap: reuses each round's already-trained local updates, no
  retraining needed.
- **This is the file P2/P3/P4 should build on.**

**Why this matters for P3 specifically (Byzantine detection):**
Per-round Shapley values legitimately include negative numbers in every
round (a client's update can transiently reduce accuracy on that round's
eval, even if their overall contribution across all rounds is positive).
This is normal, expected behavior — NOT evidence of malicious behavior.
Your drift-detection threshold needs to be tuned to catch SUSTAINED
anomalous patterns over multiple rounds, not flag any single negative
value. See `experimentation_writeup.md` Section 6 for the actual
per-round value ranges observed.

**Why this matters for P2 specifically (adaptive aggregation):**
Use `shapley_scores.csv`'s per-round values for reweighting FedAvg each
round. Do NOT use the end-of-training values from the n=30 benchmark —
those represent total contribution over all 5 rounds combined and
aren't meaningful as a per-round reweighting signal.

## 5. Known Data Limitations (not bugs)

- **HAR partitioning is non-IID by client SIZE and natural distribution
  shift, not by forced label skew.** Every client has all 6 activity
  classes present (see `preprocessed/label_distribution.png`). If you
  need harder label skew for your experiments (e.g. P3's Byzantine
  injection), use `scenarios/scenario_4_noisy_labels.pkl` or
  `scenario_2_diff_dist_same_size.pkl` rather than the raw 30-client data.

- **Scenario 2 (label skew) clients have smaller-than-target sizes**
  (121 and 88 samples instead of the 230 target for the other 8
  clients). This is because HAR doesn't have enough natural samples per
  activity class to support an 80%-skewed coalition at full size — it's
  a genuine data constraint, not a bug in `build_scenarios.py`.

## 6. Files in This Repo

| File | Purpose |
|---|---|
| `preprocess_har.py` | Raw UCI HAR -> per-client train/test splits |
| `fl_utils.py` | Shared model (MLP), FedAvg, local training, eval -- imported by everything else |
| `exact_shapley.py` | Brute-force exact Shapley ground truth (n<=6 only) |
| `gtg_shapley_approx.py` | GTG-Shapley approximation, validated against exact_shapley.py output (n=5) |
| `gtg_shapley_full_scale.py` | End-of-training Shapley at n=30 (efficiency benchmark only) |
| `build_scenarios.py` | Builds the 5 paper-replication distribution scenarios (n=10) |
| `run_scenarios.py` | Runs GTG-Shapley across all 5 scenarios, outputs `scenario_results.csv` |
| `export_shapley_csv.py` | **Produces `shapley_scores.csv` -- the actual P2/P3/P4 handoff file** |
| `experimentation_writeup.md` | Full results write-up (read after this doc) |

## 7. Questions / Issues

If something doesn't run, check (in order): venv activated, all packages
installed, `data/UCI HAR Dataset/` exists with the correct folder
structure, and scripts were run in the order listed in Section 3 (later
scripts depend on earlier scripts' `.pkl` outputs).
