# Mask-coordinate extension — working plan, 1 October 2026

The user authorized cheap reversible GF(2) checks now and adding gated follow-up work to the sprint. Existing confirmation/panel results stay fixed; this extension is exploratory until separately frozen.

## Hypothesis and novelty limits

For an observable binary mask m, append z=Am mod 2, with A invertible over GF(2), while retaining identical numerical observations and support labels. This preserves mask information and feature width. Bayesian equivalence, parity features, and feature engineering are established; an invertible reparameterization is not itself a new theorem or method. Compare against the existing [primary parity reference](https://arxiv.org/abs/2402.04248) before making novelty claims.

Falsifiable hypothesis: a map selected using support data alone can make multibit mask associations easier for a frozen predictor than raw mask coordinates or matched random invertible maps. A practical contribution additionally requires useful gains beyond equally informed simple learners. A generator-supplied map is a privileged capacity diagnostic, not a deployable compiler.

## Prioritized Day 9–20 gates

1. **Day 9: CPU checks now.** Verify rank/inverse, round-trip recovery, parity-to-coordinate examples, and identical information across representations by exhaustive small-mask enumeration. Use identity, permutations, random invertible maps, and known-map diagnostics. Preserve failed checks and seeds; no new model inference is needed for these algebra checks.
2. **Days 10–11: feasibility before paid inference.** Freeze a support-only map-selection algorithm, candidate/search budget, nested selection procedure, and unchanged support/query identities. Compare raw indicators, shuffled controls, random maps, explicit parity/interaction logistic models, and XGBoost with the same labels. Stop a method claim if gains require oracle maps or query-target selection, or simple baselines explain them.
3. **Days 12–14: bounded development.** Pin exact accessible TabPFN/TabICL checkpoints, preprocessing, packages, and seeds; require a contemporary-checkpoint comparison for a broad claim. Record inaccessible checkpoints without substitution. First paid batch has a **$1 ceiling**, within the existing $26 cap; protect the **$3 reproduction reserve**. Log reservations separately from billing, disable retries, and stop if costs or failures exceed the bound. Predeclare a useful effect threshold before scores; prioritize low-cost CPU screening.
4. **Days 15–17: independent confirmation only after development passes.** Freeze one hypothesis, primary contrast, multiplicity policy, and aggregation unit. Use new task seeds disjoint from all prior development/confirmation tasks; do not reuse 60000–60019 as untouched confirmation. Stop expansion if gains fail against matched baselines, random maps, or the prespecified threshold. No substantial adapter training follows automatically.
5. **Days 18–20: honest deliverable.** Regenerate results from saved probabilities, independently audit leakage and identities, update the manuscript only with verified claims, and reproduce within reserve. If gates fail, retain a clearly labeled negative extension and finish the existing mechanism paper.

## Current execution and deferred work

CPU checks completed:80 development tasks / 480 saved predictions, including sparse full-parity controls; two targeted tests pass; $0 paid. Read `artifacts/reports/mask_compiler_v0/interpretation.md`. Coordinate gains occur for a restricted logistic class, with mixed sparse-baseline comparisons and zero-signal harm. No TFM effect is established. Uniform marginals apply only to uniform mask populations. Supports32/128 are not nested; compiler selection uses12× more CV fits than each single-basis baseline.

`configs/mask_compiler_gate_v1.json` records a draft120-cell/two-call pilot and reserved fresh seed ranges. GPU runner integration, nested guard/matched-search baselines, novelty audit, contemporary-checkpoint access and fresh confirmation remain pending; none is marked executed. Freeze the runnable design before the first paid call. Raw/compiled/reversed inputs must preserve numerical values, NaNs and label budgets. Stop a broad or method claim if existing baselines explain the result.
