# P3 PR Description

P3 adds a self-contained Byzantine drift detector for the Shapley-FL pipeline. It calibrates on the real clean per-round baseline in `shapley_scores.csv`, evaluates on the genuine noisy-label scenario generated from the repo's own HAR preprocessing, and writes the required detection output to `byzantine_detection_results.csv`.

The final detector uses a temporal z-score built from clean-baseline standardized residuals plus a ratio-based sustain rule. On the current scenario-4 noisy-label run, it outperforms the naive single-round baseline at the client level: method `precision=0.8333`, `recall=0.4167`, `F1=0.5556` versus naive `precision=0.7273`, `recall=0.3333`, `F1=0.4571`. The implementation also produces `byzantine_detection_metrics.txt` and the summary plot at `byzantine_detection_summary.png`.

# P4 Handoff Note

P3 is ready and uses the repo's real per-round Shapley time series, not synthetic demo data. The detector output is in `byzantine_detection_results.csv` with columns `round, client_id, flagged_status, rolling_variance, trend_slope, z_score`, where `z_score` is the temporal signal used for flagging. The metrics summary in `byzantine_detection_metrics.txt` shows the sustained drift rule versus the naive single-round baseline across multiple injected severities.

P4 can now consume the `flagged_status` field directly for ledger logging and enforcement logic, with the summary plot in `byzantine_detection_summary.png` available as a quick visual reference for the final report.
