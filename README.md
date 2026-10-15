# Shapley-FL

Contribution evaluation, aggregation, Byzantine detection and on-chain audit for
federated learning on non-IID IoT sensor data (UCI HAR, 30 clients).

This repository contains the code, data pipeline and result files behind the
paper. All results use a single seed (42) and 50 federated rounds.

## Pipeline

| Stage | What it does | Main scripts (`src/`) | Main output (`results/`) |
|---|---|---|---|
| 1 | Data layer, FedAvg, GTG-Shapley engine | `preprocess_har.py`, `export_shapley_csv.py` | `p1_shapley/shapley_scores.csv` |
| 2 | Shapley-weighted aggregation, Dirichlet sweep, FedProx, fairness | `shapley_weighted_fedavg.py`, `dirichlet_alpha_sweep.py`, `fedprox_compare.py`, `jains_fairness.py` | `p2_aggregation/aggregation_results.csv` |
| 3 | Cross-round score-drift Byzantine detection | `p3_byzantine_detection.py` | `p3_detection/byzantine_detection_results.csv` |
| 4 | Solidity audit ledger (Hardhat, Sepolia) | `src/p4_blockchain-audit/` | `src/p4_blockchain-audit/*.json` |

Stages communicate only through the CSV files above. Paths are defined in
`src/paths.py`, so every script resolves them from the repository root.

## Setup

Python 3.12 and Node.js (for stage 4 only) are required.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download UCI HAR from
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
and place the extracted `UCI HAR Dataset` folder at `data/UCI HAR Dataset/`.

## Running the pipeline

Run every command from the repository root. Scripts depend on each other's
outputs, so keep this order.

```bash
# Stage 1: data and Shapley engine
python src/preprocess_har.py              # -> preprocessed/har_clients.pkl
python src/exact_shapley.py               # exact ground truth, 5 clients (~2 min)
python src/gtg_shapley_approx.py          # validation against ground truth (~2 min)
python src/build_scenarios.py             # -> scenarios/*.pkl (5 scenarios, 10 clients)
python src/run_scenarios.py               # scenario study (~1.7 h)
python src/export_shapley_csv.py          # -> results/p1_shapley/shapley_scores.csv
python src/shapley_signal_decay.py        # signal-decay figure and threshold sweep

# 30-client efficiency run (about 3.4 h on 4 worker processes)
python -u src/gtg_shapley_full_scale_parallel.py --workers 4 > results/p1_shapley/gtg_shapley_full_scale_50round.txt 2>&1

# Stage 2: aggregation
python src/shapley_weighted_fedavg.py                       # fixed threshold (~4 min)
python src/shapley_weighted_fedavg.py --adaptive --suffix _adaptive
python src/dirichlet_alpha_sweep.py                         # alpha in {1.0, 0.5, 0.1}
python src/fedprox_compare.py                               # FedProx comparison (~5 min)
python src/jains_fairness.py
python src/plot_convergence.py

# Stage 3: detection
python src/p3_byzantine_detection.py      # -> results/p3_detection/
python -m pytest tests                    # detector unit tests
```

`src/gtg_shapley_full_scale.py` is the same algorithm without multiprocessing.
The parallel version gives the same estimates up to floating-point differences.
Bit-identical results across machines are not guaranteed, because PyTorch
numerics vary slightly by platform.

## Stage 4: blockchain audit

```bash
cd src/p4_blockchain-audit
npm install
npx hardhat compile
npm run node                    # terminal 1: local Hardhat network
npm run deploy                  # terminal 2
npm run log                     # logs entries from data/*.csv
npm run gas-scale               # local gas scaling (150, 600, 1500 entries)
npm run dispute-demo
```

For Sepolia, create a `.env` file in `src/p4_blockchain-audit/` with
`SEPOLIA_RPC_URL` and `SEPOLIA_PRIVATE_KEY` (use a throwaway test wallet) and use
the `:sepolia` scripts in `package.json`. The Sepolia run logs the 1,500-entry
clean baseline (30 clients x 50 rounds). `scripts/verifyCorrected.js` reads every
entry back and compares it with `data/shapley_scores.csv` and
`data/byzantine_detection_results.csv`. See `src/p4_blockchain-audit/p4.md`.

## Where the paper's results come from

| Paper item | File |
|---|---|
| Shapley validation (five clients) | `results/p1_shapley/exact_shapley_50round.txt`, `gtg_shapley_approx_50round.txt` |
| 30-client efficiency run | `results/p1_shapley/gtg_shapley_full_scale_50round.txt` |
| Table II (scenario study) | `results/p1_shapley/scenario_results.csv` |
| Table III, Fig. 1 (weighted vs. vanilla) | `results/p2_aggregation/aggregation_results.csv`, `aggregation_round_log.csv`, `fairness_metrics.txt` |
| Fig. 2 (signal decay) | `results/p1_shapley/shapley_signal_decay.txt`, `.csv`, `.png` |
| Table IV, Fig. 3 (Dirichlet sweep) | `results/p2_aggregation/dirichlet_sweep_results.csv`, `dirichlet_sweep.txt` |
| Jain's fairness index | `results/p2_aggregation/jains_fairness.txt` |
| FedProx comparison | `results/p2_aggregation/fedprox_comparison.txt` |
| Table V (detector) | `results/p3_detection/byzantine_detection_metrics.txt` |
| Synthetic score-shift test (Krum, FLTrust) | `results/p3_detection/p3_attack_sweep.csv` |
| Local gas scaling | `src/p4_blockchain-audit/gas_scale_report.csv` |
| Sepolia verification and gas | `src/p4_blockchain-audit/verification_summary_corrected.json`, `gas_report_corrected_sepolia.json` |

## Repository layout

- `src/`: pipeline scripts (stages 1 to 3), `paths.py`, and `p4_blockchain-audit/`
  (Solidity contract, Hardhat scripts and audit reports)
- `results/`: generated CSV, text and figure outputs, grouped by stage
- `scenarios/`: the five ten-client distribution scenarios (regenerate with
  `src/build_scenarios.py`)
- `tests/`: unit tests for the detector
- `preprocessed/` and `data/`: created locally, not tracked

## Notes

- A skipped GTG-Shapley round yields all-zero scores. This means "no signal",
  not "every client contributed nothing".
- Coalition utility is measured on the same test set used to report accuracy, so
  Shapley-weighted accuracies are not independent held-out estimates.
- All results are single-seed.

## License

MIT. See `LICENSE`.
