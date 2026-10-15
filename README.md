# Shapley-FL

Contribution evaluation, aggregation, Byzantine detection and on-chain audit for
federated learning on non-IID IoT sensor data (UCI HAR, 30 clients).

This repository contains the code, data pipeline and result files behind the
paper. All results use a single seed (42) and 50 federated rounds.

## Pipeline

| Stage | What it does | Main scripts | Main output |
|---|---|---|---|
| 1 | Data layer, FedAvg, GTG-Shapley engine | `preprocess_har.py`, `export_shapley_csv.py` | `shapley_scores.csv` |
| 2 | Shapley-weighted aggregation, Dirichlet sweep, FedProx, fairness | `shapley_weighted_fedavg.py`, `dirichlet_alpha_sweep.py`, `fedprox_compare.py`, `jains_fairness.py` | `aggregation_results.csv` |
| 3 | Cross-round score-drift Byzantine detection | `p3_byzantine_detection.py` | `byzantine_detection_results.csv` |
| 4 | Solidity audit ledger (Hardhat, Sepolia) | `p4_blockchain-audit/` | `p4_blockchain-audit/*.json` |

Stages communicate only through the CSV files above.

## Setup

Python 3.12 and Node.js (for stage 4) are required.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download UCI HAR from
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
and place the extracted `UCI HAR Dataset` folder at `data/UCI HAR Dataset/`.

## Running the pipeline

Run everything from the repository root. Scripts depend on each other's outputs,
so keep this order.

```bash
# Stage 1: data and Shapley engine
python preprocess_har.py              # -> preprocessed/har_clients.pkl
python exact_shapley.py               # exact ground truth, 5 clients (~2 min)
python gtg_shapley_approx.py          # validation against ground truth (~2 min)
python build_scenarios.py             # -> scenarios/*.pkl (5 scenarios, 10 clients)
python run_scenarios.py               # scenario study (~1.7 h)
python export_shapley_csv.py          # -> shapley_scores.csv (per-round scores)
python shapley_signal_decay.py        # signal-decay figure and threshold sweep

# 30-client efficiency run (about 3.4 h on 4 worker processes)
python -u gtg_shapley_full_scale_parallel.py --workers 4 > gtg_shapley_full_scale_50round.txt 2>&1

# Stage 2: aggregation
python shapley_weighted_fedavg.py             # fixed threshold (~4 min)
python shapley_weighted_fedavg.py --adaptive --suffix _adaptive
python dirichlet_alpha_sweep.py               # alpha in {1.0, 0.5, 0.1}
python fedprox_compare.py                     # FedProx comparison (~5 min)
python jains_fairness.py
python plot_convergence.py

# Stage 3: detection
python p3_byzantine_detection.py      # -> byzantine_detection_*.csv, metrics
python -m pytest tests                # detector unit tests
```

The sequential `gtg_shapley_full_scale.py` is the same algorithm without
multiprocessing; the parallel version gives the same estimates (up to
floating-point differences). Exact bit-identical results across machines are
not guaranteed because PyTorch numerics vary slightly by platform.

## Stage 4: blockchain audit

```bash
cd p4_blockchain-audit
npm install
npx hardhat compile
npm run node                    # terminal 1: local Hardhat network
npm run deploy                  # terminal 2
npm run log                     # logs entries from data/*.csv
npm run gas-scale               # local gas scaling (150, 600, 1500 entries)
npm run dispute-demo
```

For Sepolia, set `SEPOLIA_RPC_URL` and `SEPOLIA_PRIVATE_KEY` in a `.env` file
(use a throwaway test wallet) and use the `:sepolia` scripts in `package.json`.
The Sepolia run logs the 1,500-entry clean baseline (30 clients x 50 rounds) and
`scripts/verifyCorrected.js` reads every entry back and compares it with
`data/shapley_scores.csv` and `data/byzantine_detection_results.csv`.
See `p4_blockchain-audit/p4.md` for details.

## Where the paper's results come from

| Paper item | File |
|---|---|
| Shapley validation (five clients) | `exact_shapley_50round.txt`, `gtg_shapley_approx_50round.txt` |
| 30-client efficiency run | `gtg_shapley_full_scale_50round.txt` |
| Table II (scenario study) | `scenario_results.csv` |
| Table III, Fig. 1 (weighted vs. vanilla) | `aggregation_results.csv`, `aggregation_round_log.csv`, `fairness_metrics.txt` |
| Fig. 2 (signal decay) | `shapley_signal_decay.txt`, `shapley_signal_decay.csv` |
| Table IV, Fig. 3 (Dirichlet sweep) | `dirichlet_sweep_results.csv`, `dirichlet_sweep.txt` |
| Jain's fairness index | `jains_fairness.txt` |
| FedProx comparison | `fedprox_comparison.txt` |
| Table V (detector) | `byzantine_detection_metrics.txt` |
| Synthetic score-shift test (Krum, FLTrust) | `p3_attack_sweep.csv` |
| Local gas scaling | `p4_blockchain-audit/gas_scale_report.csv` |
| Sepolia verification and gas | `p4_blockchain-audit/verification_summary_corrected.json`, `p4_blockchain-audit/gas_report_corrected_sepolia.json` |

## Repository layout

- `*.py` in the root: pipeline scripts (stages 1 to 3)
- `scenarios/`: the five ten-client distribution scenarios
- `tests/`: unit tests for the detector
- `p4_blockchain-audit/`: Solidity contract, Hardhat scripts, reports
- `*.csv`, `*.txt`, `*.png` in the root: generated results

## Notes

- A skipped GTG-Shapley round yields all-zero scores. This means "no signal",
  not "every client contributed nothing".
- Coalition utility is measured on the same test set used to report accuracy,
  so Shapley-weighted accuracies are not independent held-out estimates.
- All results are single-seed.
