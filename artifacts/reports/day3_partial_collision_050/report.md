# Saved-prediction mechanism report

Run status: **complete**. Verified 81/81 requested cells.

95% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| label_only | 0 | tabicl:v2 | native_to_indicators | 3 | -0.000228 | [-0.002005, 0.001548] |
| label_only | 0 | tabicl:v2 | native_to_shuffled | 3 | -0.000489 | [-0.006348, 0.005371] |
| label_only | 0 | tabicl:v2 | shuffled_to_indicators | 3 | 0.000260 | [-0.003824, 0.004345] |
| label_only | 0 | tabpfn:v2 | native_to_indicators | 3 | 0.000196 | [-0.000239, 0.000631] |
| label_only | 0 | tabpfn:v2 | native_to_shuffled | 3 | -0.000751 | [-0.004167, 0.002666] |
| label_only | 0 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.000947 | [-0.002431, 0.004325] |
| label_only | 0 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0 | xgboost | native_to_shuffled | 3 | -0.008393 | [-0.022385, 0.005598] |
| label_only | 0 | xgboost | shuffled_to_indicators | 3 | 0.008393 | [-0.005598, 0.022385] |
| label_only | 0.5 | tabicl:v2 | native_to_indicators | 3 | 0.010548 | [-0.002986, 0.024082] |
| label_only | 0.5 | tabicl:v2 | native_to_shuffled | 3 | 0.007739 | [-0.008041, 0.023519] |
| label_only | 0.5 | tabicl:v2 | shuffled_to_indicators | 3 | 0.002810 | [-0.002543, 0.008162] |
| label_only | 0.5 | tabpfn:v2 | native_to_indicators | 3 | 0.024862 | [-0.004401, 0.054124] |
| label_only | 0.5 | tabpfn:v2 | native_to_shuffled | 3 | 0.010231 | [-0.021008, 0.041469] |
| label_only | 0.5 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.014631 | [-0.001980, 0.031241] |
| label_only | 0.5 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.5 | xgboost | native_to_shuffled | 3 | -0.005897 | [-0.030577, 0.018782] |
| label_only | 0.5 | xgboost | shuffled_to_indicators | 3 | 0.005897 | [-0.018782, 0.030577] |
| label_only | 0.9 | tabicl:v2 | native_to_indicators | 3 | 0.001107 | [-0.005444, 0.007658] |
| label_only | 0.9 | tabicl:v2 | native_to_shuffled | 3 | 0.001075 | [-0.003329, 0.005479] |
| label_only | 0.9 | tabicl:v2 | shuffled_to_indicators | 3 | 0.000032 | [-0.002483, 0.002547] |
| label_only | 0.9 | tabpfn:v2 | native_to_indicators | 3 | 0.021520 | [-0.025214, 0.068255] |
| label_only | 0.9 | tabpfn:v2 | native_to_shuffled | 3 | 0.015603 | [-0.015844, 0.047050] |
| label_only | 0.9 | tabpfn:v2 | shuffled_to_indicators | 3 | 0.005917 | [-0.015030, 0.026864] |
| label_only | 0.9 | xgboost | native_to_indicators | 3 | 0.000000 | [0.000000, 0.000000] |
| label_only | 0.9 | xgboost | native_to_shuffled | 3 | -0.000430 | [-0.019808, 0.018948] |
| label_only | 0.9 | xgboost | shuffled_to_indicators | 3 | 0.000430 | [-0.018948, 0.019808] |

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
