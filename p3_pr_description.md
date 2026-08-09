# P3 PR Description

P3 adds a self-contained Byzantine drift detector for the Shapley-FL pipeline. It calibrates on the real clean per-round baseline in `shapley_scores.csv`, evaluates on the repository's own scenario-4 noisy-label data, and writes the required detection output to `byzantine_detection_results.csv`.

The detector uses a temporal z-score built from clean-baseline standardized residuals plus a sustain rule that now excludes truncated rounds before applying the `>=30%` eligible-round threshold. The current implementation writes the full 30-client P3 output (1500 rows), the scenario-4 flagged export, the merged handoff CSV, and the attack-sweep metrics used for the paper-style comparison.

On the current reproducible run, the row-level metrics are:
- method: precision `0.8136`, recall `0.1250`, F1 `0.2167`
- naive single-round baseline: precision `0.7619`, recall `0.0417`, F1 `0.0790`

The attack-sweep comparison now uses two distinct baselines: a Krum-style distance baseline and an FLTrust-style cosine-trust baseline. The results are exported to `p3_attack_sweep.csv` and can be compared directly against the sustained-drift detector.

# P4 Handoff Note

P3 is ready and uses the repo's real per-round Shapley time series, not synthetic demo data. The detector output is in `byzantine_detection_results.csv` with columns `round, client_id, flagged_status, rolling_variance, trend_slope, z_score`, where `z_score` is the temporal signal used for flagging and `flagged_status` is persistence-enforced (no single-round-only flags).

As documented in the metrics note, rounds `< window_size` (here rounds 1-2) intentionally have `NaN` for `rolling_variance` and `trend_slope` because those features require a full rolling window. This is expected behavior for downstream consumers.

P4 can now consume the `flagged_status` field directly for ledger logging and enforcement logic, with the summary plot in `byzantine_detection_summary.png` available as a quick visual reference for the final report.
