# Saved-prediction mechanism report

Run status: **complete**. Verified 108/108 requested cells.

95% paired Student-t intervals across independent task draws. Positive gain favors the second representation.

| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |
|---|---:|---|---|---:|---:|---|
| label_only | 0 | tabicl:v2 | imputed_to_indicators | 3 | 0.000380 | [-0.004698, 0.005457] |
| label_only | 0 | tabicl:v2 | native_to_width_control | 3 | -0.002880 | [-0.016099, 0.010338] |
| label_only | 0 | tabpfn:v2 | imputed_to_indicators | 3 | -0.002795 | [-0.012238, 0.006649] |
| label_only | 0 | tabpfn:v2 | native_to_width_control | 3 | 0.024512 | [-0.041722, 0.090747] |
| label_only | 0 | xgboost | imputed_to_indicators | 3 | -0.083500 | [-0.115989, -0.051011] |
| label_only | 0 | xgboost | native_to_width_control | 3 | -0.030064 | [-0.057327, -0.002801] |
| label_only | 0.5 | tabicl:v2 | imputed_to_indicators | 3 | 0.109432 | [0.104715, 0.114150] |
| label_only | 0.5 | tabicl:v2 | native_to_width_control | 3 | -0.004408 | [-0.026473, 0.017657] |
| label_only | 0.5 | tabpfn:v2 | imputed_to_indicators | 3 | 0.106215 | [0.095830, 0.116600] |
| label_only | 0.5 | tabpfn:v2 | native_to_width_control | 3 | -0.003663 | [-0.028968, 0.021641] |
| label_only | 0.5 | xgboost | imputed_to_indicators | 3 | 0.042263 | [0.001574, 0.082952] |
| label_only | 0.5 | xgboost | native_to_width_control | 3 | -0.034543 | [-0.096238, 0.027152] |
| label_only | 0.9 | tabicl:v2 | imputed_to_indicators | 3 | 0.450211 | [0.430579, 0.469843] |
| label_only | 0.9 | tabicl:v2 | native_to_width_control | 3 | -0.002396 | [-0.014064, 0.009271] |
| label_only | 0.9 | tabpfn:v2 | imputed_to_indicators | 3 | 0.446769 | [0.434278, 0.459260] |
| label_only | 0.9 | tabpfn:v2 | native_to_width_control | 3 | -0.006142 | [-0.013266, 0.000982] |
| label_only | 0.9 | xgboost | imputed_to_indicators | 3 | 0.420492 | [0.395389, 0.445594] |
| label_only | 0.9 | xgboost | native_to_width_control | 3 | -0.004193 | [-0.012043, 0.003658] |

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
