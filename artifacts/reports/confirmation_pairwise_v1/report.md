# Saved-prediction mechanism report

Run status: **complete**. Verified 360/360 requested cells.

97.5% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| pairwise | 0 | tabicl:v2 | native_to_indicators | 20 | 0.000184 | [-0.001890, 0.002258] |
| pairwise | 0 | tabicl:v2 | native_to_shuffled | 20 | -0.001315 | [-0.003508, 0.000878] |
| pairwise | 0 | tabicl:v2 | shuffled_to_indicators | 20 | 0.001499 | [-0.000601, 0.003599] |
| pairwise | 0 | tabpfn:v2 | native_to_indicators | 20 | -0.002538 | [-0.005454, 0.000377] |
| pairwise | 0 | tabpfn:v2 | native_to_shuffled | 20 | -0.000390 | [-0.000916, 0.000137] |
| pairwise | 0 | tabpfn:v2 | shuffled_to_indicators | 20 | -0.002149 | [-0.004917, 0.000620] |
| pairwise | 0 | xgboost | native_to_indicators | 20 | 0.000000 | [0.000000, 0.000000] |
| pairwise | 0 | xgboost | native_to_shuffled | 20 | -0.005451 | [-0.009536, -0.001365] |
| pairwise | 0 | xgboost | shuffled_to_indicators | 20 | 0.005451 | [0.001365, 0.009536] |
| pairwise | 0.9 | tabicl:v2 | native_to_indicators | 20 | 0.006143 | [0.002819, 0.009467] |
| pairwise | 0.9 | tabicl:v2 | native_to_shuffled | 20 | -0.004996 | [-0.011286, 0.001295] |
| pairwise | 0.9 | tabicl:v2 | shuffled_to_indicators | 20 | 0.011138 | [0.004512, 0.017765] |
| pairwise | 0.9 | tabpfn:v2 | native_to_indicators | 20 | 0.103894 | [0.079695, 0.128093] |
| pairwise | 0.9 | tabpfn:v2 | native_to_shuffled | 20 | 0.008900 | [-0.018175, 0.035974] |
| pairwise | 0.9 | tabpfn:v2 | shuffled_to_indicators | 20 | 0.094994 | [0.061620, 0.128368] |
| pairwise | 0.9 | xgboost | native_to_indicators | 20 | 0.000000 | [0.000000, 0.000000] |
| pairwise | 0.9 | xgboost | native_to_shuffled | 20 | -0.010556 | [-0.031514, 0.010402] |
| pairwise | 0.9 | xgboost | shuffled_to_indicators | 20 | 0.010556 | [-0.010402, 0.031514] |

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
