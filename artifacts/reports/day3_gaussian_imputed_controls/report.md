# Saved-prediction mechanism report

Run status: **complete**. Verified 108/108 requested cells.

95% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| label_only | 0 | tabicl:v2 | imputed_to_indicators | 3 | 0.001433 | [-0.002886, 0.005752] |
| label_only | 0 | tabicl:v2 | native_to_width_control | 3 | 0.001545 | [-0.005242, 0.008333] |
| label_only | 0 | tabpfn:v2 | imputed_to_indicators | 3 | -0.001063 | [-0.005726, 0.003600] |
| label_only | 0 | tabpfn:v2 | native_to_width_control | 3 | 0.000179 | [-0.007336, 0.007693] |
| label_only | 0 | xgboost | imputed_to_indicators | 3 | -0.000844 | [-0.011228, 0.009541] |
| label_only | 0 | xgboost | native_to_width_control | 3 | -0.004661 | [-0.011045, 0.001723] |
| label_only | 0.5 | tabicl:v2 | imputed_to_indicators | 3 | 0.003416 | [-0.000004, 0.006836] |
| label_only | 0.5 | tabicl:v2 | native_to_width_control | 3 | 0.000782 | [-0.006839, 0.008402] |
| label_only | 0.5 | tabpfn:v2 | imputed_to_indicators | 3 | 0.004159 | [0.001807, 0.006511] |
| label_only | 0.5 | tabpfn:v2 | native_to_width_control | 3 | 0.004136 | [-0.008078, 0.016351] |
| label_only | 0.5 | xgboost | imputed_to_indicators | 3 | 0.001326 | [-0.028335, 0.030987] |
| label_only | 0.5 | xgboost | native_to_width_control | 3 | 0.000809 | [-0.016250, 0.017868] |
| label_only | 0.9 | tabicl:v2 | imputed_to_indicators | 3 | 0.003648 | [-0.002555, 0.009851] |
| label_only | 0.9 | tabicl:v2 | native_to_width_control | 3 | 0.001713 | [-0.002277, 0.005702] |
| label_only | 0.9 | tabpfn:v2 | imputed_to_indicators | 3 | 0.003951 | [-0.002566, 0.010469] |
| label_only | 0.9 | tabpfn:v2 | native_to_width_control | 3 | 0.005816 | [-0.005635, 0.017266] |
| label_only | 0.9 | xgboost | imputed_to_indicators | 3 | -0.018290 | [-0.046945, 0.010365] |
| label_only | 0.9 | xgboost | native_to_width_control | 3 | 0.003959 | [-0.018440, 0.026357] |

## Interpretation limits

- Synthetic tasks are not independent real datasets.
- Small task counts produce unstable intervals; n=1 has no interval.
- Fractions require baseline-to-oracle gap > 1e-4 and retain negative gains.
- Shuffled indicators use unlabeled support/query batches independently.
- Elapsed model-cell time includes preprocessing/fit/inference but excludes setup and checkpoint hashing.
- Version comparisons do not identify a causal pretraining-prior effect.
- Cold-start cells: 0; these are Laplace fallbacks, not backbone inference.
- Checkpoint identity: hashes recorded.
- Development results inform configuration selection; confirmation results must remain separate.
