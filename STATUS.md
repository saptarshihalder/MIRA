# MIRA status — October 1, 2026

Target: NeurIPS-quality draft by October 20; $26 compute cap; strict token rationing. **Research evidence gates through Day 8 are complete for accessible checkpoints.** The manuscript is a working draft, with a confirmed synthetic representation effect and a completed, narrower real-covariate evaluation. No neural-adapter contribution is established.

## Completed evidence

- CPU pilot reproduced: 31,680 raw rows, 24 summaries; maximum summary discrepancy 2.22e-16. Original hardware/BLAS remain unknown.
- **1,296 development cells**, then **360 frozen confirmation cells** on twenty fresh Gaussian pairwise tasks with 256 support labels. Twelve engineering smoke cells are excluded. Protocol commit `7ffb13d` preceded scores. The two TabPFN v2 primary gains are .103894 [.079695,.128093] native-to-indicators and .094994 [.061620,.128368] shuffled-to-indicators; prespecified 97.5% task intervals, nominal Bonferroni familywise 95% coverage under t assumptions. Seeds 60000–60019 are now used.
- **240 post-confirmation CPU baseline predictions** on the same forty saved tasks. Support-CV L1 interaction logistic loss .193364, oracle gap .008930; descriptive advantage over TabPFN indicators .015945 [.006618,.025272]. The engineered basis is disclosed. Known-beta mixture is a parameter-informed reference, excluded from the operational pool. Gamma-zero harms are retained.
- **1,440 frozen real-covariate cells**: six fixed UCI datasets, 6,955 cleaned rows, five grouped folds/dataset, 128 support labels, four models, three views, two rates/two signals. Data/protocol/source freeze commit `dbda963` preceded any scores; SHA e2058edaf9cb246a691f98659157e573ad94b74c237a53c33b505c711c45fcac. CC BY 4.0 citations, original CSV hashes, duplicate maps and canonical splits are saved.
- Independent full-panel audit reconstructs all 120 episodes and verifies predictions, row/group boundaries, masks, NLL/Brier/AUROC and all 48 aggregate contrasts. Score differences <=1.11e-16; no recorded fit/predict warnings. Six equally weighted datasets are uncertainty units after query-weighted fold averaging; overlapping folds are not independent replicates.
- 76 local tests pass; 30 affected tests reran after the final freeze guard; reviewer independently passed 16 targeted tests. Completed inference total is **3,096 development/confirmation/panel cells**, excluding smoke and CPU baseline predictions.

## Scientific decision

Keep substantial adapter training inactive. The strong simple synthetic baseline nearly closes the oracle gap. At imposed gamma .8, six-dataset TabPFN gains are .009644 [-.010141,.029428] at rate .1 and .003818 [-.002215,.009851] at rate .5. TabICL native gains are near zero/negative; all corresponding TFM native intervals include zero. Generic imputed logistic gains .171407 [.074178,.268635] at rate .5/gamma .8, but loses .033732 at gamma zero. XGBoost native/true-indicator probabilities coincide; shuffled indicators can change them. Full controls remain in reports.

These outcomes support the scoped pairwise representation finding, without a universal mask-blindness, prior-causality, natural-missingness or clinical claim. The panel imposes label association retrospectively; query labels enter the evaluator's mask generator and scoring, never learner fitting/prediction. Active columns show no exact fitted-mean collisions, so this panel does not reproduce the deliberate all-zero TabICL erasure diagnostic. Inactive constant columns and unavailable acquisition groups remain documented limitations.

## Budget and artifacts

Conservative reservations **$9.05 across 19 calls** (18 complete, one historical checkpoint failure). Latest provider-reported MIRA app charges **$0.50322492**; billing may lag and image-build attribution is not independently audited. Reservations are not invoices. $3 reproduction reserve remains within $26; no additional funds requested. Jobs are bounded, single-container and have no automatic retries.

Read `artifacts/reports/day4_8_completion.md`, `day4_8_readiness.md`, `strong_baselines/report.md`, and `real_panel_v1/report.md`. Panel regeneration: `python scripts/report_real_panel.py`, then `python scripts/plot_real_panel.py` using saved raw runs. CPU baseline reproduction: `python scripts/run_strong_baselines.py --help`. Raw GPU predictions/checkpoints remain locally under ignored `artifacts/runs/`; source, configurations, canonical public data, reports, figures, identities and cost records are tracked.

## Remaining Day 9–20 gates

Freeze targeted label-budget/representation ablations before new evaluations; do not call reused confirmation tasks untouched. Resolve contemporary TabPFN v3.5 license/token access only when available; no substitution or repeated blind retries. Review official formatting/checklist, clean-command reproduction, curated anonymous prediction packaging and authorship before final readiness.

The same `paper/main.tex` is updated and remains open. Native compilation was checked after the final edit and failed with environment diagnostic `Unable to find standard directories for platform`; PDF layout is unverified. Structural environment/citation checks pass. No replacement document or terminal TeX installation was made. BrowserOS signed-in ChatGPT was verified previously, but no verified Pro critique obtained. No email sent or manuscript submitted. Daily continuation remains scheduled at 10 AM IST through October 20; next run starts from these saved results.
