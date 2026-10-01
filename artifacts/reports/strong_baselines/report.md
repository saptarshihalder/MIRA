# Strong CPU baselines

post-confirmation descriptive follow-up; no new blind confirmation.

Every predictor uses the same saved support labels/query observations as the TFMs. C is selected on support-only folds; selected models refit all support rows. Query targets are opened only after outputs are saved.

| Gamma | Method | Tasks | Expected NLL | Oracle gap |
|---:|---|---:|---:|---:|
| 0 | histgb_native_indicators | 20 | 0.833257 | 0.214136 |
| 0 | interaction_l1_logistic_cv | 20 | 0.644151 | 0.025030 |
| 0 | interaction_l2_logistic_cv | 20 | 0.644927 | 0.025806 |
| 0 | pilot_mixture_known_beta_reference | 20 | 0.619121 | 0.000000 |
| 0 | sparse_mixture_fitted_base | 20 | 0.623478 | 0.004357 |
| 0 | u_logistic_cv | 20 | 0.623478 | 0.004357 |
| 0.9 | histgb_native_indicators | 20 | 0.283861 | 0.099427 |
| 0.9 | interaction_l1_logistic_cv | 20 | 0.193364 | 0.008930 |
| 0.9 | interaction_l2_logistic_cv | 20 | 0.211233 | 0.026799 |
| 0.9 | pilot_mixture_known_beta_reference | 20 | 0.200368 | 0.015934 |
| 0.9 | sparse_mixture_fitted_base | 20 | 0.202286 | 0.017852 |
| 0.9 | u_logistic_cv | 20 | 0.625311 | 0.440877 |

The known-beta mixture is a parameter-informed reference outside the operational pool. The fitted-base mixture reuses support-fitted U-logistic probabilities; both retain the preserved pilot's generator-informed basis/rate and gamma prior, which omits .9.
Paired effects against each saved native/indicator model are in paired_effects.csv; positive values favor the CPU method. Intervals are descriptive 95% paired Student-t intervals across 20 tasks within gamma, without multiplicity correction.
These are stationary, clean support/query synthetic tasks. They do not establish a neural-adapter advantage, online-shift performance, or a real-data benchmark.

Timing: predictor-row times are incremental within the shared pool. For standalone fitted-base mixture cost, add the corresponding U-only model time; `cost_accounting.json` reconstructs it from saved rows. Total task wall time counts the shared U fit once. Setup before the task timer is excluded; CPU timings are not a controlled GPU-versus-CPU cost comparison.
