# Annual-cohort headroom findings — October 4, 2026

The precommitted 15-group screen is complete. All three endpoint gates and the two-family successor gate FAIL. These are conventional diagnostic learners, not a new model contribution.

| Endpoint | Frozen | Platt | Additive | Calibrated tree |
|---|---:|---:|---:|---:|
| brfss_asthma3 | 0.404718 | 0.406001 | 0.405546 | 0.407223 |
| brfss_diabete4 | 0.353262 | 0.353488 | 0.353608 | 0.359491 |
| road_ksi | 0.560858 | 0.550446 | 0.552566 | 0.555172 |

Frozen predictions win on both BRFSS endpoints. Platt wins on roads; calibrated trees lose to Platt in all five road groups. Thus correcting the tree initialization and giving it a support-only four-setting search did not establish conditional nonlinear headroom. This finite screen does not prove that every nonlinear learner will fail. Do not repeat these panels or lower the gate.

The batch used 512 support labels and 1,024 queries per group: 23,040 endpoint-row uses, 7,959 unique BRFSS respondents and 7,680 road records. BRFSS endpoints overlap and are not independent datasets. Source models used historical 2024 CUDA training; source and target identities are disjoint by year. These annual cohorts were unused in recorded model fits/scores before this screen, but raw labels and aggregates were previously exposed. All scored rows are now used development. No confirmation, clinical, causal or universal-safety claim.

Independent audit: all 60 metric arrays and model reloads reproduce exactly; native values/labels join to raw rows; source/target separation and convex stationarity pass. Fitting took 10.25 CPU seconds, excluding preparation/audit. No cloud call or new charge. Protocol/code and input hashes were committed before scoring. Tree search has four extra calibration fits beyond its 12 CV tree fits; this overhead is disclosed.

Reproduce from the repository with Python 3.12, XGBoost 3.4.1, NumPy and SciPy. The saved freeze requires original public raw NPZ files at the paths in configs/large_native_v3.json. Import annual_headroom after adding experiments/crossfit_v1 to the module path, set OUT to a nonexistent reproduction directory, then call run(); do not edit the frozen source or overwrite tracked results. The audit imports the same module; set its OUT consistently. Saved inputs, models, searches and predictions are tracked.

The next candidate is in docs/ACQUISITION_SUCCESSOR_PLAN.md. This is a proposed problem/model change with explicit evidence gates, not demonstrated novelty or efficacy. Repeated frozen-predictor correction trials are closed.
