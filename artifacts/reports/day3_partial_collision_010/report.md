# Saved-prediction mechanism report

Run status: **complete**. Verified 81/81 requested cells.

95% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| label_only | 0 | tabicl:v2 | native_to_indicators | 3 | -0.001138 | [-0.006264, 0.003989] |
| label_only | 0 | tabicl:v2 | native_to_shuffled | 3 | -0.000776 | [-0.002318, 0.000766] |
| label_only | 0 | tabicl:v2 | shuffled_to_indicators | 3 | -0.000362 | [-0.005580, 0.004856] |
| label_only | 0 | tabpfn:v2 | native_to_indicators | 3 | -0.000194 | [-0.001084, 0.000697] |
| label_only | 0 | tabpfn:v2 | native_to_shuffled | 3 | -0.001711 | [-0.006569, 0.003147] |
| label_only | 0 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.001518 | [-0.002742, 0.005777] |
| label_only | 0 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0 | xgboost | native_to_shuffled | 3 | -0.000690 | [-0.004386, 0.003006] |
| label_only | 0 | xgboost | shuffled_to_indicators | 3 | 0.000690 | [-0.003006, 0.004386] |
| label_only | 0.5 | tabicl:v2 | native_to_indicators | 3 | 0.004406 | [-0.000721, 0.009532] |
| label_only | 0.5 | tabicl:v2 | native_to_shuffled | 3 | 0.004180 | [-0.001980, 0.010339] |
| label_only | 0.5 | tabicl:v2 | shuffled_to_indicators | 3 | 0.000226 | [-0.004212, 0.004664] |
| label_only | 0.5 | tabpfn:v2 | native_to_indicators | 3 | 0.024246 | [-0.003062, 0.051553] |
| label_only | 0.5 | tabpfn:v2 | native_to_shuffled | 3 | 0.006404 | [-0.005317, 0.018124] |
| label_only | 0.5 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.017842 | [0.000724, 0.034961] |
| label_only | 0.5 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.5 | xgboost | native_to_shuffled | 3 | -0.001100 | [-0.022950, 0.020750] |
| label_only | 0.5 | xgboost | shuffled_to_indicators | 3 | 0.001100 | [-0.020750, 0.022950] |
| label_only | 0.9 | tabicl:v2 | native_to_indicators | 3 | 0.000658 | [-0.003758, 0.005073] |
| label_only | 0.9 | tabicl:v2 | native_to_shuffled | 3 | 0.000124 | [-0.002156, 0.002404] |
| label_only | 0.9 | tabicl:v2 | shuffled_to_indicators | 3 | 0.000534 | [-0.006113, 0.007181] |
| label_only | 0.9 | tabpfn:v2 | native_to_indicators | 3 | 0.017112 | [-0.000355, 0.034579] |
| label_only | 0.9 | tabpfn:v2 | native_to_shuffled | 3 | 0.007846 | [-0.020917, 0.036609] |
| label_only | 0.9 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.009267 | [-0.010240, 0.028773] |
| label_only | 0.9 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.9 | xgboost | native_to_shuffled | 3 | 0.001619 | [-0.003457, 0.006695] |
| label_only | 0.9 | xgboost | shuffled_to_indicators | 3 | -0.001619 | [-0.006695, 0.003457] |

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
