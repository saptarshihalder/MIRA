# Day 1–3 development summary

Seven additional runs completed 1026 cells; total development matrix cells: 1296 (12 smoke cells excluded). Reports were regenerated from saved probabilities and data with hash/row-identity checks. Three independently generated tasks per condition; all intervals below are exploratory, unadjusted paired 95% t intervals. No confirmation seeds were used.

The 16-column width control is a stress test; the true/shuffled-indicator comparisons use matched nuisance width. Baseline missing rate .1 changes oracle headroom and is not an equal-signal comparison with .5.

## Effects at gamma .9 (nats; positive favors the second representation)

| Run | Family | Predictor | Comparison | Mean [interval] |
|---|---|---|---|---|
| gaussian_imputed_controls | label_only | tabicl:v2 | imputed_to_indicators | 0.00365 [-0.00256, 0.00985] |
| gaussian_imputed_controls | label_only | tabicl:v2 | native_to_width_control | 0.00171 [-0.00228, 0.00570] |
| gaussian_imputed_controls | label_only | tabpfn:v2 | imputed_to_indicators | 0.00395 [-0.00257, 0.01047] |
| gaussian_imputed_controls | label_only | tabpfn:v2 | native_to_width_control | 0.00582 [-0.00563, 0.01727] |
| gaussian_imputed_controls | label_only | xgboost | imputed_to_indicators | -0.01829 [-0.04694, 0.01036] |
| gaussian_imputed_controls | label_only | xgboost | native_to_width_control | 0.00396 [-0.01844, 0.02636] |
| zero_collision_imputed_controls | label_only | tabicl:v2 | imputed_to_indicators | 0.45021 [0.43058, 0.46984] |
| zero_collision_imputed_controls | label_only | tabicl:v2 | native_to_width_control | -0.00240 [-0.01406, 0.00927] |
| zero_collision_imputed_controls | label_only | tabpfn:v2 | imputed_to_indicators | 0.44677 [0.43428, 0.45926] |
| zero_collision_imputed_controls | label_only | tabpfn:v2 | native_to_width_control | -0.00614 [-0.01327, 0.00098] |
| zero_collision_imputed_controls | label_only | xgboost | imputed_to_indicators | 0.42049 [0.39539, 0.44559] |
| zero_collision_imputed_controls | label_only | xgboost | native_to_width_control | -0.00419 [-0.01204, 0.00366] |
| interaction_families | label_only | tabicl:v2 | native_to_indicators | -0.00190 [-0.00317, -0.00062] |
| interaction_families | label_only | tabpfn:v2 | native_to_indicators | 0.02029 [-0.00468, 0.04525] |
| interaction_families | label_only | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| interaction_families | pairwise | tabicl:v2 | native_to_indicators | 0.00917 [-0.01106, 0.02940] |
| interaction_families | pairwise | tabpfn:v2 | native_to_indicators | 0.07654 [0.04555, 0.10753] |
| interaction_families | pairwise | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| interaction_families | sparse_pair | tabicl:v2 | native_to_indicators | 0.25372 [-0.28770, 0.79514] |
| interaction_families | sparse_pair | tabpfn:v2 | native_to_indicators | 0.34015 [-0.04018, 0.72048] |
| interaction_families | sparse_pair | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| interaction_families | value_dependent | tabicl:v2 | native_to_indicators | 0.00836 [-0.00677, 0.02350] |
| interaction_families | value_dependent | tabpfn:v2 | native_to_indicators | 0.03159 [0.00933, 0.05386] |
| interaction_families | value_dependent | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| partial_collision_010 | label_only | tabicl:v2 | native_to_indicators | 0.00066 [-0.00376, 0.00507] |
| partial_collision_010 | label_only | tabpfn:v2 | native_to_indicators | 0.01711 [-0.00035, 0.03458] |
| partial_collision_010 | label_only | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| partial_collision_050 | label_only | tabicl:v2 | native_to_indicators | 0.00111 [-0.00544, 0.00766] |
| partial_collision_050 | label_only | tabpfn:v2 | native_to_indicators | 0.02152 [-0.02521, 0.06825] |
| partial_collision_050 | label_only | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| low_missingness | label_only | tabicl:v2 | native_to_indicators | 0.01076 [-0.01421, 0.03573] |
| low_missingness | label_only | tabpfn:v2 | native_to_indicators | 0.03329 [0.00697, 0.05960] |
| low_missingness | label_only | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| low_missingness | pairwise | tabicl:v2 | native_to_indicators | -0.00329 [-0.00937, 0.00279] |
| low_missingness | pairwise | tabpfn:v2 | native_to_indicators | -0.00006 [-0.00212, 0.00200] |
| low_missingness | pairwise | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| low_missingness | value_dependent | tabicl:v2 | native_to_indicators | 0.04167 [0.00894, 0.07439] |
| low_missingness | value_dependent | tabpfn:v2 | native_to_indicators | 0.03935 [0.02040, 0.05831] |
| low_missingness | value_dependent | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |
| quantized_step1 | label_only | tabicl:v2 | native_to_indicators | 0.00341 [0.00245, 0.00436] |
| quantized_step1 | label_only | tabpfn:v2 | native_to_indicators | 0.01697 [-0.00105, 0.03500] |
| quantized_step1 | label_only | xgboost | native_to_indicators | 0.00000 [0.00000, 0.00000] |

Conservative reservations: $5.55 of $26; $3 reproduction reserve retained. Reservations include failures and are not invoices. Full gamma-zero results and losses remain in each linked run report and day3_summary.json.
