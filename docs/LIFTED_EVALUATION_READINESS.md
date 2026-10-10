# Lifted evaluation readiness — October 10, 2026

**Verdict:** the frozen design supports bounded comparisons; publication efficacy remains unestablished before complete v4 verification. This audit read protocol/source only, without panels, result arrays, partial endpoints, training or cloud calls. Two fabricated-input tests passed (seed averaging and shared-week clustering; 0.29 s).

## Supported

- `confirm4.py` requires exactly seeds 1, 2, 3 and averages their NLL cells before task/mask aggregation and intervals. These are average individual-model scores, not an ensemble predictive density. Synthetic observations are tasks, not seeds or masks.
- Beijing pools pollutants/stations within each saved week, then weights weeks equally. `realdata.py` constructs non-overlapping seven-day test windows; shared dates across pollutants remain one cluster. This estimates an equal-week mean, with episodes equally weighted inside each week; it is not equal-pollutant weighting when episode counts differ.
- Source fine-tuning uses 2013–2015; evaluation uses 2016–February 2017. Each episode samples 48 support and 48 query rows without replacement. Normalization, context and FA anchors use support only. PFN/LCT forward paths ignore query labels; TabPFN fits support labels and receives query labels only in subsequent density scoring. Random same-week support/query sampling supports contemporaneous adaptation, not forecasting.
- PFN-L/LCT-L share the declared architecture scale, task prior, seeds, optimizer and 40,000-step recipe; fine-tuning shares the 4,000-episode source pool, 2,000 steps and seed 11. Final checkpoints are bound before scoring. This matches update/data budgets, not wall-time or all computational work.
- `tabpfn_colab.py` uses eight estimators, random state 0, NaN masks and its returned criterion; it casts density inputs to border dtype and checks criterion means against returned means. Both methods score the same unstandardized **log-concentration target**, not physical concentration density. No cross-model Jacobian correction is needed for paired gains on that common target.

## Unknowns and claim limits

- Intervals use 1.96 times the standard error of task/week gains. They condition on the three fitted seeds and omit training-randomness uncertainty. Week clustering handles within-week dependence, not demonstrated independence between adjacent weeks.
- E8–E12 have five marginal decisions, without a multiplicity adjustment or declared global decision rule. Report all five and call intervals nominal endpoint-wise 95%; selected passes do not establish family-wise superiority. E10 tests the fixed −0.02 non-inferiority margin, not equivalence.
- E12 compares source-fine-tuned LCT with in-context-only TabPFN: identical episode support, unequal prior target-specific training. It cannot isolate architecture or equal total training/compute. Retain this qualification even if E12 passes.
- Local records do not establish TabPFN pretraining/development overlap clearance. Weight/package identity identifies the comparator; it does not certify unseen benchmarks. Do not assert contamination, or contamination-free generalization, without provenance evidence.
- Dequantization followed inspection of test distributions and undequantized PM10/O3 closed-form outputs, before compared-model fits/scores according to the protocol. Fixed integer-reading noise precedes logging and is shared across source/test construction; finer-grid source readings remain unchanged. Disclose this chronology and amended estimand; it is not fully untouched-data preregistration.

## Blockers before efficacy claims

Require complete three-seed checkpoints and arrays, independent endpoint recomputation and preserved failed original time gates. The October 9 extension changes execution limits only. Preserve/check actual TabPFN weight hashes with the main completion evidence: the prepared local verifier now checks fixed pretrained bytes, SHA-256, metadata mapping and unchanged-weight runner evidence; the actual completed bundle has not yet been verified. Score archives lack embedded checkpoint/panel hashes; runner provenance supplies their binding, without independent cryptographic timestamps. These limits remain after numerical verification.
