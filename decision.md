# Research decision — October 1, 2026

**Complete Day 1–3; proceed with an interaction-focused representation study. Defer substantial adapter training.**

The analytic CPU pilot is reproduced, but is not a TFM experiment. Strong simple mask-aware baselines nearly close its oracle gap at 128 labels. Generic frozen residual adaptation, observation-policy adaptation and expert aggregation already have precedents. Oracle headroom alone does not justify a new neural architecture.

Across 1,296 completed development cells, the deliberately all-zero construction reveals version-specific TabICL preprocessing loss. Mean imputation removes native mask distinctions; its constant-feature filter drops nuisance columns. Explicit all-zero imputation causes analogous information loss for all three tested models. Partial-zero and quantized controls do not reproduce the large native failure. Fitted means generally differ from observed zeros, so these are not calibrated exact-collision doses.

The strongest broader candidate is Gaussian pairwise missingness in TabPFN v2: gamma .9 indicator gain .07654 nats [.04555,.10753], actual versus same-width shuffled .07436 [.03485,.11386]. Value-dependent native gain is .03159 [.00933,.05386]. Sparse pairs have large, highly uncertain effects. At baseline missing rate .1, selected single/value-dependent gains occur; rate changes also alter oracle headroom. Gamma-zero harms and negative effects remain in reports. All intervals use only three development tasks and are unadjusted; these are hypotheses for confirmation, not confirmed improvements or pretraining-prior causality.

Day 4–6 should freeze contrasts, controls, independent tasks, multiplicity handling and budget before confirmation seeds 60000–60019. Test simple mask-aware baselines and refreshed contexts with matched labels, then the predeclared real-covariate panel. Activate an adapter only if useful reproducible headroom survives those comparisons and the contribution exceeds established residual adaptation.

Engineering gates pass for named accessible versions: 60 tests, complete manifests, saved probabilities, disjoint row IDs, exact posterior/data/source audits and bounded cost accounting. Contemporary TabPFN v3.5 remains license/token blocked. NeurIPS-quality manuscript readiness is not yet established: independent confirmation, real-covariate evaluation, official formatting and verified PDF remain required.
