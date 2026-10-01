# Day 3 saved-data audit

Status: **passed**; 7/7 runs complete.

108 materialized task files cover 27 distinct seed/family draws; gamma and value variants repeat paired underlying draws.

U, labels, masks, oracle/base probabilities, and support/query IDs are compared exactly across paired value variants.

| Run | Tasks | Baseline r | Observed zeros / observed cells | Empirical missing fraction |
|---|---:|---:|---:|---:|
| day3_gaussian_imputed_controls | 9 | 0.5 | 0/46261 (0.0000%) | 49.5456% |
| day3_zero_collision_imputed_controls | 9 | 0.5 | 46261/46261 (100.0000%) | 49.5456% |
| day3_interaction_families | 36 | 0.5 | 0/206113 (0.0000%) | 50.0310% |
| day3_partial_collision_010 | 9 | 0.5 | 4538/46312 (9.7988%) | 49.9322% |
| day3_partial_collision_050 | 9 | 0.5 | 23420/46312 (50.5700%) | 49.9322% |
| day3_low_missingness | 27 | 0.1 | 0/207104 (0.0000%) | 10.1664% |
| day3_quantized_step1 | 9 | 0.5 | 17827/46312 (38.4933%) | 49.9322% |

The missing-fraction summary above averages support/query task fractions equally; per-column/task values and exact zero counts are in JSON.

- day3_gaussian_imputed_controls versus day3_zero_collision_imputed_controls: **passed**, 9 exact task pairs.
- day3_partial_collision_010 versus day3_partial_collision_050: **passed**, 9 exact task pairs.
- day3_partial_collision_010 versus day3_quantized_step1: **passed**, 9 exact task pairs.
- day3_partial_collision_050 versus day3_quantized_step1: **passed**, 9 exact task pairs.

missing_rate is the reference independent Bernoulli rate, not an assertion that all empirical/marginal rates equal it. Value-dependent outcome modulation can alter marginal rates; finite draws fluctuate as well.
Observed-zero counts describe nuisance values only; U stays observed. Counts across gamma variants are descriptive, not independent samples for an uncertainty calculation.
A finite observed zero does not necessarily equal its fitted imputation constant. JSON separately counts observed values exactly equal to the support-only float64 column mean; this reference does not instrument a third-party encoder's arithmetic.
The audit checks saved prediction/evaluation separation through disjoint row IDs; it does not certify every third-party model implementation.
No confirmation data or paid services were used.
