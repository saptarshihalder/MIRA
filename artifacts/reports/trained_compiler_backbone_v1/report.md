# Frozen trained-compiler development validation

All saved probability/episode hashes and losses reconstructed. Positive gain means trained loss is lower; no confirmation claim.

| Backbone | Kind | Condition | Native | Identity | Trained | Heuristic | Linear | DeepSets | CV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tabpfn:v2 | synthetic | k2, gamma=0 | 0.64269 | 0.65034 | 0.65034 | 0.65034 | 0.65034 | 0.65034 | 0.65145 |
| tabpfn:v2 | synthetic | k2, gamma=0.9 | 0.38840 | 0.23832 | 0.23316 | 0.23316 | 0.23832 | 0.23316 | 0.23316 |
| tabpfn:v2 | synthetic | k4, gamma=0 | 0.62784 | 0.62967 | 0.62967 | 0.62967 | 0.62967 | 0.62967 | 0.63486 |
| tabpfn:v2 | synthetic | k4, gamma=0.9 | 0.62159 | 0.38790 | 0.22132 | 0.22132 | 0.22132 | 0.22132 | 0.22132 |
| tabpfn:v2 | natural | hepatitis | 0.39815 | 0.41321 | 0.41321 | 0.41611 | 0.41321 | 0.41321 | 0.41047 |
| tabpfn:v2 | natural | horse_colic | 0.52761 | 0.51922 | 0.51922 | 0.51922 | 0.51922 | 0.51922 | 0.52075 |
| tabicl:v2 | synthetic | k2, gamma=0 | 0.64155 | 0.63804 | 0.63804 | 0.63804 | 0.63804 | 0.63804 | 0.64124 |
| tabicl:v2 | synthetic | k2, gamma=0.9 | 0.26990 | 0.25256 | 0.23190 | 0.23190 | 0.25256 | 0.23190 | 0.23190 |
| tabicl:v2 | synthetic | k4, gamma=0 | 0.63141 | 0.63346 | 0.63346 | 0.63346 | 0.63346 | 0.63346 | 0.64186 |
| tabicl:v2 | synthetic | k4, gamma=0.9 | 0.46062 | 0.31187 | 0.21034 | 0.21034 | 0.21034 | 0.21034 | 0.21034 |
| tabicl:v2 | natural | hepatitis | 0.37222 | 0.38415 | 0.38415 | 0.39276 | 0.38415 | 0.38415 | 0.38937 |
| tabicl:v2 | natural | horse_colic | 0.52091 | 0.51986 | 0.51986 | 0.51986 | 0.51986 | 0.51986 | 0.52020 |

tabpfn:v2: high-signal gain 0.085870; null harm -0.000000; numeric screen pass=True.

tabicl:v2: high-signal gain 0.061096; null harm -0.000000; numeric screen pass=True.

No confirmatory intervals. Natural results aggregate overlapping grouped folds within each of two fixed datasets; not six independent replications.

Confidence fallback is not a no-harm guarantee. Natural support encoder uses zero U proxy and numeric category codes; clinical and general missingness benefits are not established.
