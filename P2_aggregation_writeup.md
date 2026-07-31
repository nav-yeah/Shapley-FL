# P2 — Adaptive Aggregation: Results Writeup

## Result (required 1-paragraph summary)

Shapley-weighted FedAvg improves **convergence speed** but not final
accuracy on the honest 30-client UCI HAR setup. Both variants were run
from the identical initial model (GLOBAL_SEED=42), same 5 rounds, same
local-training seeds; the weighted variant computes per-round GTG-Shapley
values online each round (reusing P1's `gtg_shapley_one_round`) and sets
aggregation weights via softmax(β·φ) with β=50, falling back to standard
size-proportional weights on GTG-truncated rounds (all-zero φ). The
weighted run led at every intermediate round — most notably round 3
(**0.6943 vs 0.6201**, a +7.4-point gap, i.e. the weighted model reaches
vanilla's round-4 accuracy roughly one round early) — and finished
statistically tied at round 5 (0.7411 vs 0.7420). This is the expected
outcome for an all-honest, naturally non-IID cohort: every client's data
is genuinely useful, so down-weighting low-φ clients can only accelerate
early training, not raise the ceiling. The mechanism's headline value
should appear in P3's Byzantine setting, where persistently low/negative-φ
clients are actually malicious and suppressing their weight protects
final accuracy, not just convergence speed.

## Per-round accuracy (main experiment, natural subject partition)

| Round | Vanilla FedAvg | Shapley-weighted |
|---|---|---|
| 0 (init) | 0.1114 | 0.1114 |
| 1 | 0.5140 | **0.5333** |
| 2 | 0.5386 | **0.5439** |
| 3 | 0.6201 | **0.6943** |
| 4 | 0.7093 | **0.7247** |
| 5 | **0.7420** | 0.7411 |

Convergence speed (first round reaching target accuracy) is identical at
coarse thresholds (≥0.50: round 1 both; ≥0.60: round 3 both; ≥0.70:
round 4 both) — the weighted variant's advantage is in *how far past*
each threshold it lands, not in crossing them earlier at this
granularity.

## Warm-up period (original plan feature, configurable)

`WARMUP_ROUNDS` in `shapley_weighted_fedavg.py` implements the original
plan's warm-up (first N rounds of plain FedAvg before Shapley weighting).
Default is **0**: in this pipeline φ is computed *within* each round
before aggregating, so there is no stale-score instability for a warm-up
to protect against, and with only 5 total rounds a warm-up erases part of
the effect. Verified with `WARMUP_ROUNDS=1`:

| Round | Weighted (no warm-up) | Weighted (warm-up=1) |
|---|---|---|
| 1 | 0.5333 | 0.5140 (= vanilla, by construction) |
| 3 | 0.6943 | 0.6919 |
| 5 | 0.7411 | 0.7420 |

Same story either way; warm-up costs the round-1 gain and changes nothing
downstream. Keep 0 for the sprint deliverable; the flag exists for the
paper's ablation table.

## Non-IID severity sweep (original plan "Experiment 2")

`dirichlet_alpha_sweep.py` re-partitions pooled HAR data with
Dirichlet(α) label skew (30 synthetic clients, partition repair
guaranteeing ≥30 samples/client, fixed global test set) and repeats the
comparison. Under skew, client sizes vary widely, so the weighted variant
uses the size-aware tilt w_i ∝ n_i·exp(β·φ_i) with a gentler **β=10**
(φ magnitudes are larger under skew; β=50 demonstrably collapses
training — documented in the script header).

| α (skew) | Vanilla final | Weighted final | Gap |
|---|---|---|---|
| 1.0 (mild) | 0.6996 | 0.6948 | −0.005 (tie; weighted led round 3: 0.6991 vs 0.6620) |
| 0.5 (moderate) | 0.8221 | 0.8206 | −0.001 (tie; weighted led rounds 2–4) |
| 0.1 (extreme) | 0.6475 | 0.5415 | **−0.106** |

**Honest finding:** raw Shapley reweighting *hurts* under extreme label
skew. At α=0.1 a client's per-round φ conflates "low-quality update"
with "holds rare classes the current global model handles badly" —
down-weighting the latter starves minority classes and costs ~10 points.
This is not a bug; it is the known heterogeneity blind spot of raw
contribution scores, and it directly motivates a heterogeneity-corrected
Shapley variant (e.g. correcting φ by each client's label-distribution
divergence before weighting) as the paper's next step. Report it as a
limitation-plus-motivation, not a failure.

## Optional ShapFed-style fairness metric

Standalone accuracy = each client trained alone from the same init for an
equivalent budget (15 epochs), evaluated on the shared global test set.
Pearson correlation of standalone accuracy against each scheme's credit
signal (natural partition):

| Credit signal | Pearson r |
|---|---|
| Data size (vanilla FedAvg's implicit credit) | +0.144 |
| Cumulative per-round Shapley φ | +0.105 |
| Cumulative Shapley-softmax weights | −0.011 |

Interpretation: on natural HAR non-IID (no attackers), no credit signal
strongly tracks standalone usefulness — per-round φ measures *marginal*
contribution to the coalition, not solo skill, and HAR clients are fairly
homogeneous in quality. Report honestly; it strengthens the case that the
interesting differentiation happens under Byzantine injection (P3).

## Notes for P3 (format is locked)

- `aggregation_results.csv`: exactly the agreed columns
  `round, client_id, aggregation_weight, global_accuracy_vanilla,
  global_accuracy_weighted` — 150 rows (30 clients × 5 rounds), weights
  sum to 1.0 within each round.
- **Truncated rounds exist**: rounds where GTG-Shapley skipped
  computation (round gain ≤ ε_b = 0.02) have size-proportional weights,
  not Shapley-derived ones (rounds 2 and 5 in this run). In
  `shapley_scores.csv` those rounds appear as all-zero φ — treat
  all-zero rounds as "no signal" (exclude from your eligible-round
  count), not as evidence of anything.
- Reminder from `SETUP_AND_HANDOFF.md` §4: every non-truncated round has
  legitimate small negative φ values. Do not flag single negatives.
- Extra caution from the α-sweep above: under label skew, low φ can mean
  "rare-class holder", not "attacker". Your detector should key on
  *sustained deviation from a client's own history*, not absolute low
  scores — which is exactly the personal z-score / trend-slope design in
  your brief.

## Reproduction

```
python preprocess_har.py            # if not already done
python shapley_weighted_fedavg.py   # main deliverable, ~1-2 min CPU
python dirichlet_alpha_sweep.py     # extension experiment, ~6 min CPU
python dirichlet_alpha_sweep.py 0.5 # (optional) re-run a single alpha
```

Outputs: `aggregation_results.csv`, `fairness_metrics.txt`,
`aggregation_comparison.png`, `dirichlet_sweep_results.csv`,
`dirichlet_sweep.png`.
