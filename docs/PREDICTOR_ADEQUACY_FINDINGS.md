# Predictor adequacy findings â€” October 5, 2026

**Optimization diagnostic PASS; prespecified practical-advantage gate FAIL.** Training was insufficient at200 updates. Longer training improves actual entropy-planned utility on all three initializations, including the original192-task source. This supplies a concrete positive development result, not a new architecture or JMLR/NeurIPS readiness.

Frozen protocol: `docs/PREDICTOR_ADEQUACY_PROTOCOL.md`, pre-fit commit43e0562. Same8,161-parameter predictor,200/2000 updates,192/3072 source tasks;12 checkpoints from six training trajectories. All checkpoints and two ridge models were fixed before scoring96 new source-distribution tasks. The same six Boolean families recur; no unseen-family transfer, real-data evaluation, old closed test access or fresh-confirmation claim. Source teachers are privileged population conditionals; inference uses64 incomplete support labels and observed query features.

| Source tasks | Updates | Mean risk + cost | Range over 3 seeds |
|---|---:|---:|---:|
| 192 | 200 | 0.623607 | 0.618854â€“0.626730 |
| 192 | 2000 | 0.530208 | 0.529143â€“0.530892 |
| 3072 | 200 | 0.620365 | 0.616783â€“0.624001 |
| 3072 | 2000 | 0.521829 | 0.521095â€“0.522584 |

Controls: 192-source ridge 0.532667;3072-source ridge 0.528990;support-count EM 0.562446;privileged population Bayes 0.514497. Lower is better. These are exact expected query losses over eight binary worlds, averaged over96 tasks and12 dependent availability/cost scenarios, not independent dataset estimates.

The fixed3072-task/2000-update candidate averages0.521829: 15.88% below its200-update counterpart, 7.22% below counts and 1.35% below ridge. All three seed gains over ridge are positive (.007181,.007895,.006406), but **all miss the predeclared.01 margin**. Do not lower the margin or claim statistical significance. Count control lacks source pretraining; ridge shares the same privileged source teachers but has1,485 coefficients and a different optimization objective/compute cost. This comparison does not isolate a novel architectural ingredient.

All three old small-source/200-update weight sets reproduce exactly. Every saved prediction table,36,864 policy-risk cells, gate and mean is independently replayed; maximum policy-score discrepancy3.93e-08. Raw support summaries and validation truth reconstructed. Audit seconds15.578; fit/evaluation seconds26.922;12,000 updates; cloud0USD; no training retries. The audit uses separate recursive policy evaluation; shared network class/data constants remain common dependencies.

**Close this finite batch.** Prior Bellman failures do not establish architectural impossibility: optimization was a material confound. Conversely, longer optimization of an established MLP cannot be presented as a novel model. No Q retraining or external scoring is authorized by this failed practical margin. Next research needs a separately justified trained mechanism, strong equally trained controls and a new frozen protocol before further fitting. Real-data transfer and distinct novelty remain required; this output does not mean only cosmetic work remains before a venue-level submission.

Reproduce from the pre-fit commit in a clean checkout (prepared arrays are versioned): run `python experiments/predictor_adequacy.py`. Copy `experiments/predictor_adequacy_audit.py` from the final result commit into that checkout, then run it; the audit script was added after fitting. Existing result directories are intentionally refused to prevent overwrites; use a clean checkout for fitting. In this checkout run only the audit. CPU environment:Python3.11.9,Torch2.8.0+cpu,NumPy2.3.5; one thread; no GPU claim. All results and checkpoints are retained.
