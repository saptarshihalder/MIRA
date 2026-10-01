# Saved-prediction mechanism report

Run status: **complete**. Verified 81/81 requested cells.

95% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| label_only | 0 | tabicl:v2 | native_to_indicators | 3 | 0.001304 | [-0.002568, 0.005176] |
| label_only | 0 | tabicl:v2 | native_to_shuffled | 3 | 0.002297 | [-0.002650, 0.007245] |
| label_only | 0 | tabicl:v2 | shuffled_to_indicators | 3 | -0.000993 | [-0.005300, 0.003313] |
| label_only | 0 | tabpfn:v2 | native_to_indicators | 3 | 0.002652 | [-0.004262, 0.009565] |
| label_only | 0 | tabpfn:v2 | native_to_shuffled | 3 | 0.001061 | [-0.013709, 0.015832] |
| label_only | 0 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.001590 | [-0.008740, 0.011921] |
| label_only | 0 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0 | xgboost | native_to_shuffled | 3 | -0.011032 | [-0.045993, 0.023928] |
| label_only | 0 | xgboost | shuffled_to_indicators | 3 | 0.011032 | [-0.023928, 0.045993] |
| label_only | 0.5 | tabicl:v2 | native_to_indicators | 3 | 0.005816 | [-0.001175, 0.012806] |
| label_only | 0.5 | tabicl:v2 | native_to_shuffled | 3 | 0.002798 | [-0.001171, 0.006767] |
| label_only | 0.5 | tabicl:v2 | shuffled_to_indicators | 3 | 0.003018 | [-0.000044, 0.006080] |
| label_only | 0.5 | tabpfn:v2 | native_to_indicators | 3 | 0.016624 | [-0.036325, 0.069574] |
| label_only | 0.5 | tabpfn:v2 | native_to_shuffled | 3 | 0.003279 | [-0.005400, 0.011958] |
| label_only | 0.5 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.013345 | [-0.039420, 0.066111] |
| label_only | 0.5 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.5 | xgboost | native_to_shuffled | 3 | -0.005358 | [-0.029283, 0.018566] |
| label_only | 0.5 | xgboost | shuffled_to_indicators | 3 | 0.005358 | [-0.018566, 0.029283] |
| label_only | 0.9 | tabicl:v2 | native_to_indicators | 3 | 0.003407 | [0.002450, 0.004365] |
| label_only | 0.9 | tabicl:v2 | native_to_shuffled | 3 | 0.001232 | [-0.003265, 0.005729] |
| label_only | 0.9 | tabicl:v2 | shuffled_to_indicators | 3 | 0.002175 | [-0.003202, 0.007552] |
| label_only | 0.9 | tabpfn:v2 | native_to_indicators | 3 | 0.016974 | [-0.001048, 0.034997] |
| label_only | 0.9 | tabpfn:v2 | native_to_shuffled | 3 | 0.011173 | [-0.005568, 0.027913] |
| label_only | 0.9 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.005802 | [-0.002932, 0.014535] |
| label_only | 0.9 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.9 | xgboost | native_to_shuffled | 3 | 0.001141 | [-0.019918, 0.022201] |
| label_only | 0.9 | xgboost | shuffled_to_indicators | 3 | -0.001141 | [-0.022201, 0.019918] |

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
