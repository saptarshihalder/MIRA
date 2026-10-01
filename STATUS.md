# MIRA status — October 1, 2026

Target: NeurIPS-quality draft by October 20; $26 compute cap; conserve tokens. **Day 1–3 execution is complete for accessible checkpoints.** The paper is a working draft, with one scoped frozen hypothesis now confirmed and real-covariate evaluation pending.

## Completed evidence

- CPU pilot reproduced: 31,680 raw rows, 24 summaries; maximum summary discrepancy 2.22e-16. Eight fresh-clone hash checks pass. Original hardware/BLAS unknown. Portable reproduction is in experiments/cpu_reproduction/.
- Accessible TabPFN v2 and TabICL v2 GPU smoke passed; 12 smoke cells excluded from development counts. TabPFN v3.5 access failed with a license/token requirement; no substitution.
- Initial Gaussian/exact-zero development matrices: 270 cells. Seven Day 3 controls: 1,026 cells. **1,296 development cells total**, three independent tasks per condition, 256 support rows, 1,024 queries, four TFM ensembles; TabPFN v2, TabICL v2 and XGBoost.
- All seven new manifests complete with no unresolved failures. Reports regenerated from saved probabilities and hash/row checks. Final data audit passes: exact paired U/Y/M/oracle/base/IDs, source/environment identity checks, disjoint support/query IDs, and independent posterior recomputation. At the Day 3 checkpoint, confirmation seeds were unused; see the subsequent frozen confirmation below.
- Extended exact generator supports baseline missing rates, partial-zero point masses and quantization; default sampling retains prior arrays. **60 local tests pass**; source whitespace check passes. Modal reservations are locked/atomic, crash reservations retained, warm-container outputs isolated, jobs bounded with no retries.
- Exact TabICL 2.2.0 preprocessing source audit, original scientific plot, six-dataset real-covariate declaration and updated manuscript are saved.

## Scientific decision

Continue the representation study; keep substantial adapter training inactive. Exact-zero native TabICL loses masks during mean imputation/constant filtering. That large effect does not extend to the tested partial-zero controls; observed zeros generally do not equal fitted means. Explicit all-zero imputation loses information for all three models, so restoring indicators is an ordinary imputation result.

A stronger development lead is interaction structure: at gamma .9, TabPFN v2 pairwise native-to-indicator gain is .07654 nats [.04555,.10753], and true-versus-same-width-shuffled gain is .07436 [.03485,.11386]. Value-dependent gain is .03159 [.00933,.05386]. Sparse-pair intervals are wide. Low baseline rate .1 produces selected single/value-dependent gains but changes oracle headroom; it is not an equal-signal comparison. These three-task, unadjusted intervals are exploratory. See artifacts/reports/day3_summary.md and day3_data_audit.md.

## Budget and next gate

Conservative reservations total **$6.05** across thirteen calls, including the failed checkpoint smoke; $3 reproduction reserve retained within $26. Latest provider-reported MIRA app cost is **$0.28619407**; billing may lag and image-build attribution is not independently audited. Reservations are not invoices. Internal allocations were rebalanced, without increasing the cap.

Next: freeze an interaction-focused confirmation design, primary contrasts and multiplicity policy before using fresh seeds; include simple mask-aware baselines and matched label budgets. Implement the predeclared real-covariate panel. A generic residual adapter is not established novelty, and current development does not activate that branch.

## Access and artifact limits

BrowserOS connected to signed-in ChatGPT, but no verified Pro critique was obtained; attempts are logged in artifacts/reports/pro_critique.md. No email sent or manuscript submitted. TabPFN v3.5 needs license acceptance/token. Raw GPU predictions are local under ignored artifacts/runs/; reports, hashes, exact configurations and code are tracked. Preserve raw files for final artifact packaging.

The same manuscript source is paper/main.tex, opened in the native editor. Native compilation has an environment failure: `Unable to find standard directories for platform`; a PDF is unverified. Exact official NeurIPS formatting/checklist and final readiness review remain pending. No terminal TeX/plugin installation is needed. First three smoke estimates mistakenly labeled two physical cores as two vCPUs; use corrected rates/provider billing, preserving original records.

Daily autonomous continuation is scheduled at 10 AM IST through October 20, automation mira-research-sprint. An agent reached the account usage limit at final sign-off; the executable saved-data audit was run directly and passed. Avoid additional agents/reruns without an unresolved gate.

## Frozen confirmation completed after Day 3

Protocol 7ffb13d was committed/pushed before evaluation. The Gaussian pairwise panel completed 360/360 cells on 20 new tasks (seeds 60000–60019), including gamma 0 and .9 controls. The two prespecified TabPFN v2 primary contrasts passed: native-to-indicators .10389 [.07969,.12809] and shuffled-to-indicators .09499 [.06162,.12837] nats. Intervals are 97.5% each (Bonferroni nominal familywise 95% for two primary contrasts); other results are descriptive. The mean-loss table and validated figure are in artifacts/reports/confirmation_pairwise_v1/ and artifacts/figures/confirmation_pairwise.png.

Independent analysis recomputed primary intervals and audited all 40 task files for seed identity, disjoint row IDs, NaN-mask alignment and exact query posterior. No fallback/failure occurred. Updated manuscript source is saved in the same editor; native compilation still reports an environment failure, so no PDF is verified.

These confirmation seeds are now used for this pairwise panel. Do not call changed configs or reruns on it untouched confirmation. Next gate: simple mask-aware operational baselines/label budgets, the six-dataset real-covariate panel, current-checkpoint access and remaining manuscript/artifact readiness checks. Keep adapter inactive pending those comparisons. No additional agents or GPU reruns are needed to regenerate these saved results.
