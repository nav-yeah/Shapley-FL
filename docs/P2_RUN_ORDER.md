# P2 — Re-run guide against P1's 50-round pipeline

Everything below is run from the repo root in PowerShell with the venv active.

---

## 0. Sync your branch with the updated `main`

P1 merged the 50-round pipeline into `main`. Your branch is behind by
several commits (P3 and P4 merged too), and the earlier `git pull` failed
with `curl 18 / early EOF`. That error is a network buffer problem, not a
repo problem — bump the buffer first.

```powershell
cd C:\path\to\Shapley-FL
.\venv\Scripts\Activate.ps1

# fix the curl 18 / early EOF failure
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

git checkout p2-adaptive-aggregation
git fetch origin
git merge origin/main
```

**Expect one conflict, in `.gitignore`.** Main's most recent commit is
literally "Resolved .gitignore merge conflict", so both sides touched it.
Open the file, keep every line from both sides, delete the `<<<<<<<`,
`=======`, `>>>>>>>` markers, then:

```powershell
git add .gitignore
git commit -m "Merge main (P1 50-round pipeline) into p2-adaptive-aggregation"
```

Verify the merge brought in the new pipeline before going further:

```powershell
python -c "import pandas as pd; d=pd.read_csv('results/p1_shapley/shapley_scores.csv'); print(len(d), 'rows,', d['round'].max(), 'rounds')"
```

You want `1500 rows, 50 rounds`. If you still see 150 rows, the merge
didn't take.

---

## 1. Drop in the six updated files

Replace these five, and add the one new one:

| File | Status |
|---|---|
| `shapley_weighted_fedavg.py` | rewritten |
| `dirichlet_alpha_sweep.py` | rewritten |
| `jains_fairness.py` | rewritten |
| `plot_convergence.py` | rewritten |
| `fedprox_compare.py` | rewritten |
| `shapley_signal_decay.py` | **new** |

---

## 2. Run the free result first (2 seconds, no training)

```powershell
python src/shapley_signal_decay.py
```

This reads P1's committed `results/p1_shapley/shapley_scores.csv` and needs no FL training,
no preprocessed data, nothing. It produces the finding that shapes
everything else, and it has a built-in self-check that confirms the
reconstruction is sound. Read `results/p1_shapley/shapley_signal_decay.txt` before running
anything expensive.

---

## 3. Main deliverable — 50-round aggregation run

```powershell
# clear the 5-round outputs so nothing stale survives
Remove-Item results/p2_aggregation/aggregation_results.csv, results/p2_aggregation/fairness_metrics.txt, results/p2_aggregation/aggregation_comparison.png -ErrorAction SilentlyContinue

python src/shapley_weighted_fedavg.py
```

Budget **15–30 min**. ETA prints every round, and both CSVs flush every
round, so if you Ctrl-C or the machine sleeps, the completed rounds
survive.

Watch for this line in the first few seconds:

```
[check] round-1 phi matches P1's results/p1_shapley/shapley_scores.csv exactly ...
```

That confirms your code and P1's engine agree. If it warns instead, stop
and ask P1 whether the seeds, the partition, or the model changed after
they exported the CSV — everything downstream depends on that agreement.

Then the second configuration:

```powershell
python src/shapley_weighted_fedavg.py --adaptive --suffix _adaptive
```

Another 15–30 min. This one writes to `*_adaptive.*` filenames, so the
two runs coexist and you can put them side by side in the writeup.

Smoke-test either with `--rounds 3 --skip-standalone` first if you want
to confirm it runs on your machine before committing half an hour.

---

## 4. Analysis scripts (seconds each)

```powershell
python src/jains_fairness.py
python src/plot_convergence.py

python src/jains_fairness.py --csv results/p2_aggregation/aggregation_results_adaptive.csv --log results/p2_aggregation/aggregation_round_log_adaptive.csv --suffix _adaptive
python src/plot_convergence.py --csv results/p2_aggregation/aggregation_results_adaptive.csv --log results/p2_aggregation/aggregation_round_log_adaptive.csv --suffix _adaptive
```

---

## 5. FedProx baseline

```powershell
python src/fedprox_compare.py
```

Budget **20–40 min** — three arms this time. Optionally repeat with
`--adaptive --suffix _adaptive`.

---

## 6. Dirichlet sweep

**Delete the old CSV first.** The script merges per-alpha results into an
existing file so you can run one alpha per session, but that means
5-round rows from the old run will sit alongside your 50-round rows if
you don't clear it:

```powershell
Remove-Item results/p2_aggregation/dirichlet_sweep_results.csv, results/p2_aggregation/dirichlet_sweep_rounds.csv -ErrorAction SilentlyContinue

python src/dirichlet_alpha_sweep.py
```

Budget **40–90 min** for all three alphas. To split it across sessions:

```powershell
python src/dirichlet_alpha_sweep.py 0.1
python src/dirichlet_alpha_sweep.py 0.5
python src/dirichlet_alpha_sweep.py 1.0
```

The figure and summary regenerate from the CSV each time, so they're
complete once the last alpha finishes.

---

## 7. Commit and open the PR

```powershell
git add shapley_weighted_fedavg.py dirichlet_alpha_sweep.py jains_fairness.py `
        plot_convergence.py fedprox_compare.py shapley_signal_decay.py
git add results/p2_aggregation/aggregation_results.csv results/p2_aggregation/aggregation_round_log.csv results/p2_aggregation/fairness_metrics.txt `
        results/p2_aggregation/aggregation_comparison.png results/p2_aggregation/convergence_curve.png `
        results/p2_aggregation/jains_fairness.csv results/p2_aggregation/jains_fairness.txt results/p2_aggregation/jains_fairness.png `
        results/p2_aggregation/fedprox_comparison.csv results/p2_aggregation/fedprox_comparison.txt results/p2_aggregation/fedprox_comparison.png `
        results/p2_aggregation/dirichlet_sweep_results.csv results/p2_aggregation/dirichlet_sweep_rounds.csv `
        results/p2_aggregation/dirichlet_sweep.png results/p2_aggregation/dirichlet_sweep.txt `
        results/p1_shapley/shapley_signal_decay.csv results/p1_shapley/shapley_signal_decay.txt results/p1_shapley/shapley_signal_decay.png
git add results/p2_aggregation/aggregation_results_adaptive.csv results/p2_aggregation/aggregation_round_log_adaptive.csv `
        results/p2_aggregation/fairness_metrics_adaptive.txt results/p2_aggregation/aggregation_comparison_adaptive.png `
        results/p2_aggregation/convergence_curve_adaptive.png results/p2_aggregation/jains_fairness_adaptive.csv `
        results/p2_aggregation/jains_fairness_adaptive.txt results/p2_aggregation/jains_fairness_adaptive.png

git commit -m "P2: 50-round adaptive aggregation; GTG truncation analysis and adaptive threshold"
git push origin p2-adaptive-aggregation
```

---

## Total time budget

| Step | Time |
|---|---|
| Signal decay analysis | seconds |
| Main run (fixed) | 15–30 min |
| Main run (adaptive) | 15–30 min |
| Analysis scripts | seconds |
| FedProx | 20–40 min |
| Dirichlet sweep | 40–90 min |
| **Total** | **~1.5–3 hours**, mostly unattended |

If you're short on time, steps 2–4 alone are a complete, defensible P2
deliverable. FedProx and Dirichlet are the extensions.

---

## What to tell P3 and P4

Both are affected by the truncation finding and neither knows yet.

**P3 (Byzantine detection).** Their detector requires a client to be
flagged in ≥30% of *eligible* rounds. At 50 rounds with P1's current
threshold, only 12 rounds carry any Shapley signal — so the rule is
operating on 12 observations, not 50, and every round from 23 onward is
blank. Their `N_ROUNDS = 5` constant in `p3_byzantine_detection.py` is
also still un-updated. `results/p1_shapley/shapley_signal_decay.csv` gives them the exact
per-round eligibility flags.

**P4 (blockchain).** 38 of 50 logged rounds will carry all-zero
valuations. Worth a sentence in the audit-trail description, and it also
means their gas-cost-vs-scale analysis should note that most entries are
structurally empty.
