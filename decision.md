# Research decision — October 1, 2026

**Proceed with the mechanism study. Defer substantial adapter training.**

The supplied CPU pilot uses an analytic mask-blind predictor, not a TFM. Its strong simple baselines nearly close the full-oracle gap by 128 labels. This establishes the need to test representation behavior and operational baselines before choosing a learned architecture.

The completed development runs now support version-specific descriptive TFM results. They do not support adapter superiority, clinical benefit or a pretraining-prior cause. TabPFN-3.5 already explicitly encodes native missingness information; its inference access is currently blocked by license acceptance/token requirements.

The first development gate is checkpoint-compatible native/indicator/shuffled inference at gamma zero and .8. Evaluate full task-level effects across the declared matrix after throughput is measured. A single smoke seed is engineering validation, not a scientific conclusion.

Conditions for an adapter branch: useful, reproducible development headroom beyond a pooled simple correction set and matched refreshed-context TFM; feasible cache/training cost inside the cap; clear contribution beyond frozen residual adaptation and classical expert aggregation. The existing oracle gap alone fails this gate.

## Development update

Two 135-cell runs completed: Gaussian nuisance values and exact-zero observed values, matched U/Y/M/oracle/row identities; three tasks per gamma. TabPFN v2's Gaussian indicator effects are small and every paired interval includes zero. TabICL v2 effects are near zero with Gaussian values, but rise to 0.4274 nats at gamma .9 under exact-zero collisions (exploratory 95% paired interval .4079–.4468). TabPFN has no corresponding failure; XGBoost native/actual-indicator outputs match.

Exact TabICL 2.2.0 source inspection localizes the special-case effect: ndarray mean imputation maps zeros and NaNs to the same values, and constant-feature filtering removes nuisance columns. Native query predictions are bit-identical across all five gamma values within each seed. This is a preprocessing finding, not prior causality, and the degenerate construction alone is insufficient for a strong paper.

Next: matched imputed/indicator controls, partial collisions/quantization, interaction families, lower missingness, and the predeclared real-covariate panel. Freeze the final confirmation scope after these development checks. Keep the adapter inactive. Confirmation seeds 60000–60019 remain unused.
