# Larger native A100 development

Nine actual NVIDIA A100-SXM4-40GB trials: three tasks × three training seeds.
Four official 2024/2025 files contain 1,016,280 raw records; latest cohorts were released in 2026.
The health tasks share respondents. Three seeds are training variation, not independent datasets.
Source fitting uses 50,000 labels per trial; each neural/Adam-linear scorer gets 800 CUDA updates.

| Method | Diabetes | Asthma | Road KSI |
|---|---:|---:|---:|
| frozen | 0.333466 | 0.395842 | 0.611319 |
| native_scratch | 0.333545 | 0.395610 | 0.603685 |
| native_linear | 0.333727 | 0.395844 | 0.603648 |
| native_linear_converged | 0.335339 | 0.396242 | 0.608504 |
| global_simplex | 0.333553 | 0.395690 | 0.603348 |
| global_platt | 0.333638 | 0.395780 | 0.612424 |
| support_intercept | 0.336424 | 0.398488 | 0.603497 |
| support_platt | 0.335693 | 0.398324 | 0.603197 |
| cv_select | 0.348587 | 0.409734 | 0.604457 |
| moment | 0.348738 | 0.411683 | 0.604513 |

NLL in nats, lower is better; equally weighted held-out regions and seeds.
Development has ten states per health target and six police-force groups for road collisions.
The model improves road loss relative to the frozen source, but support Platt and global simplex stacking perform better.
Health effects are tiny; diabetes neural loss is slightly worse than frozen. Distinct neural novelty remains unsupported.

All 1,470 probability arrays, checkpoints, source hashes, public-row labels and group/year boundaries pass independent audit.
Maximum NLL discrepancy 7.05e-8; NumPy checkpoint prediction discrepancy 2.02e-7.
All nine contextual linear fits attain the 1e-7 gradient tolerance; worst simplex KKT residual is 2.01e-6.
XGBoost host/device prediction fallback is retained: CUDA training is verified, every inference kernel is not asserted to use CUDA.

The first A100 call stopped on a calibration L-BFGS line search after four complete trials; all partial results are preserved.
A separately frozen Newton repair uses exactly the same ridge objective, inputs, partitions, seeds and models.
Its first package lacked a required historical source mount and failed before child training.
The final complete mount finished all nine trials in 62.020 seconds. Failures are included in the ledger.
No automatic retry or scientific selection was performed.

Reservations are $16.95 within the latest $20 cap, with $3 protected for reproduction.
Provider app UI reports $0.04/$0.02/$0.05 for the three attempts (rounded and not verified final invoices).
The successful app remains deployed with zero active GPU containers, no schedule and no public endpoint.

[Live Modal app](https://modal.com/apps/saptarshihalder/main/deployed/mira-large-native-a100-final)

[CDC 2025](https://www.cdc.gov/brfss/annual_data/annual_2025.html), [CDC 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html), [DfT](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).
Raw input hashes, codebook/layout rules and fixed protocol are committed; raw data remain ignored.
BRFSS age is already imputed, reporting/state composition changes, and road-specification migration creates structural absence.
These results do not identify causal acquisition-policy effects, population prevalence, clinical effectiveness or venue readiness.
