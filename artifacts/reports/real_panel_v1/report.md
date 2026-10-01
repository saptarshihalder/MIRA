# Frozen six-dataset real-covariate panel

Verified 1440 cells across six datasets and five frozen folds each.

Descriptive fixed-panel empirical losses. Folds averaged with query weights within each dataset; dataset-balanced aggregation. Six datasets, not30 folds or query rows, are uncertainty units. Unadjusted nominal95% t intervals; no real-data oracle; retrospective imposed label association, not natural-missingness or clinical benefit.

| Rate | Gamma | Model | Native minus indicators | Interval |
|---|---|---|---:|---|
| 0.1 | 0 | logistic | -0.020668 | [-0.042132, 0.000797] |
| 0.1 | 0 | tabicl:v2 | -0.003071 | [-0.015914, 0.009773] |
| 0.1 | 0 | tabpfn:v2 | -0.002046 | [-0.014949, 0.010858] |
| 0.1 | 0 | xgboost | 0.000000 | [0.000000, 0.000000] |
| 0.1 | 0.8 | logistic | -0.000523 | [-0.026359, 0.025313] |
| 0.1 | 0.8 | tabicl:v2 | 0.000341 | [-0.014901, 0.015584] |
| 0.1 | 0.8 | tabpfn:v2 | 0.009644 | [-0.010141, 0.029428] |
| 0.1 | 0.8 | xgboost | 0.000000 | [0.000000, 0.000000] |
| 0.5 | 0 | logistic | -0.033732 | [-0.064606, -0.002858] |
| 0.5 | 0 | tabicl:v2 | -0.009820 | [-0.026133, 0.006493] |
| 0.5 | 0 | tabpfn:v2 | -0.009470 | [-0.021870, 0.002929] |
| 0.5 | 0 | xgboost | 0.000000 | [0.000000, 0.000000] |
| 0.5 | 0.8 | logistic | 0.171407 | [0.074178, 0.268635] |
| 0.5 | 0.8 | tabicl:v2 | -0.004172 | [-0.009678, 0.001333] |
| 0.5 | 0.8 | tabpfn:v2 | 0.003818 | [-0.002215, 0.009851] |
| 0.5 | 0.8 | xgboost | 0.000000 | [0.000000, 0.000000] |
