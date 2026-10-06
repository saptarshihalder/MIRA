**Table A. Repository development panels, all ten two-sensor deletions (mean Gaussian NLL, lower is better).**
Rows marked privileged use the true task parameters; every other row sees only the 48 labeled support rows.

| Method | `cavity_site_v1` panel | `anchored_cavity_v1` panel |
|---|---:|---:|
| Exact Bayes oracle (privileged) | 0.014 | -0.046 |
| Population joint Gaussian (privileged) | 0.114 | 0.057 |
| Lifted cavity K=1, seed 1 (this work) | 0.150 | 0.077 |
| Residual MLP on same anchor, seed 1 | 0.160 | 0.086 |
| FA-Gaussian, support-only | 0.216 | 0.156 |
| EM-Gaussian, support-only | 0.233 | 0.168 |
| Complete-case ridge | 0.258 | 0.195 |
| Repo ridge (the gate's control) | 0.429 | 0.396 |
| Repo cavity (3 seeds) | 1.429 (all seeds collapsed) | 0.403–0.430 |
| Repo full aggregate (3 seeds) | 1.429 (all seeds collapsed) | 0.408–0.413 |
| Repo static sites (3 seeds) | 1.429 (all seeds collapsed) | 0.412–0.441 |
| Repo larger MLP (3 seeds) | 0.727–0.883 | 0.436–0.513 |

**Table B. Fresh panel (256 new tasks, seed 20261007). Mean NLL by number of missing query sensors. Learned models were trained only on masks with at most one missing sensor, so k = 2 and 3 are unseen patterns. Learned rows: mean ± s.d. over training seeds (n in brackets). Last column: share of the FA-Gaussian→oracle gap closed at k = 2.**

| Method | k=0 | k=1 | k=2 | k=3 | gap closed (k=2) |
|---|---:|---:|---:|---:|---:|
| Exact Bayes oracle (privileged) | -0.468 | -0.295 | -0.068 | 0.250 | 1.00 |
| Moment-matched Gaussian oracle (privileged) | -0.464 | -0.289 | -0.058 | 0.266 | 0.95 |
| Population joint Gaussian (privileged) | -0.331 | -0.166 | 0.045 | 0.334 | 0.43 |
| **Lifted cavity, K=1** [3] | -0.308 ± 0.001 | -0.152 ± 0.001 | 0.059 ± 0.001 | 0.358 ± 0.000 | 0.37 |
| Lifted cavity, K=2 [1] | -0.306 | -0.149 | 0.062 | 0.362 | 0.35 |
| Residual MLP on the same anchor + encoders [3] | -0.273 ± 0.000 | -0.123 ± 0.000 | 0.078 ± 0.000 | 0.362 ± 0.000 | 0.27 |
| Lifted sites, no cavity input [3] | -0.265 ± 0.001 | -0.115 ± 0.001 | 0.084 ± 0.001 | 0.363 ± 0.001 | 0.24 |
| FA-Gaussian, support-only (closed form) | -0.202 | -0.058 | 0.132 | 0.398 | 0.00 |
| EM-Gaussian, support-only (closed form) | -0.121 | -0.025 | 0.142 | 0.399 | -0.05 |
| Complete-case ridge (closed form) | -0.061 | 0.023 | 0.170 | 0.409 | -0.19 |
| 1-D sites, same machinery (K=0) [3] | -0.094 ± 0.001 | 0.034 ± 0.001 | 0.205 ± 0.002 | 0.445 ± 0.002 | -0.36 |
| Repo anchored aggregate, 60k tasks/20k steps [1] | 0.043 | 0.157 | 0.305 | 0.511 | -0.87 |
| Repo MLP, 60k tasks/20k steps [1] | 0.030 | 0.150 | 0.307 | 0.523 | -0.88 |
| Repo anchored cavity, 60k tasks/20k steps [1] | 0.044 | 0.159 | 0.310 | 0.522 | -0.89 |
| Repo anchored cavity, 60k tasks/3k steps [1] | 0.081 | 0.192 | 0.334 | 0.528 | -1.01 |
| Repo anchored cavity, repo regime (512 tasks/3k steps) [1] | 0.107 | 0.216 | 0.355 | 0.547 | -1.12 |
| Repo MLP, repo regime [1] | 0.099 | 0.219 | 0.374 | 0.588 | -1.21 |
| Repo ridge, mean-imputed (closed form) | 0.201 | 0.277 | 0.390 | 0.566 | -1.29 |

Paired task-level 95% intervals for the seed-averaged lifted cavity (positive = lifted cavity better):

| Comparison | k=0 | k=1 | k=2 | k=3 |
|---|---:|---:|---:|---:|
| vs FA-Gaussian, support-only (closed form) | +0.106 [+0.090, +0.122] | +0.094 [+0.082, +0.106] | +0.073 [+0.065, +0.082] | +0.040 [+0.035, +0.045] |
| vs EM-Gaussian, support-only (closed form) | +0.187 [+0.160, +0.214] | +0.127 [+0.114, +0.140] | +0.083 [+0.075, +0.091] | +0.041 [+0.036, +0.046] |
| vs Residual MLP on the same anchor + encoders | +0.035 [+0.028, +0.043] | +0.029 [+0.024, +0.034] | +0.020 [+0.016, +0.023] | +0.004 [+0.001, +0.007] |
| vs Lifted sites, no cavity input | +0.043 [+0.036, +0.050] | +0.037 [+0.032, +0.043] | +0.026 [+0.022, +0.029] | +0.005 [+0.002, +0.008] |
| vs 1-D sites, same machinery (K=0) | +0.214 [+0.188, +0.240] | +0.186 [+0.166, +0.206] | +0.146 [+0.131, +0.161] | +0.087 [+0.078, +0.097] |
| vs Repo anchored cavity, 60k tasks/20k steps | +0.352 [+0.323, +0.381] | +0.311 [+0.287, +0.335] | +0.252 [+0.233, +0.271] | +0.164 [+0.151, +0.178] |

**Table C. Out-of-training-distribution panels (128 new tasks each). Share of the FA-Gaussian→oracle NLL gap closed, averaged over k = 0–3 (negative = worse than the closed form). Mean over available training seeds.**

| Panel | oracle NLL (k=2) | FA-Gaussian NLL (k=2) | Lifted cavity, K=1 | Residual MLP on the same anchor + encoders | Lifted sites, no cavity input | 1-D sites, same machinery (K=0) | EM-Gaussian, support-only (closed form) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 sensors (trained on 5) | -0.561 | -0.291 | 0.42 [3] | n/a | 0.24 [3] | -0.45 [3] | -0.89 |
| Nonlinearity 0.8 (trained 0.4) | -0.123 | 0.264 | 0.37 [3] | 0.28 [3] | 0.20 [3] | 0.12 [3] | 0.07 |
| Linear sensors (trained 0.4) | -0.035 | 0.052 | -0.02 [3] | -0.03 [3] | 0.11 [3] | -1.53 [3] | -0.08 |
| 16 support rows (trained 48) | -0.088 | 0.425 | 0.21 [3] | 0.21 [3] | 0.18 [3] | 0.08 [3] | -8.52 |
| 96 support rows (trained 48) | -0.067 | 0.083 | 0.42 [3] | 0.33 [3] | 0.28 [3] | -0.64 [3] | 0.02 |

**Table D. UCI Air Quality, 23 later test weeks (windows starting 6 Oct 2004 – 23 Mar 2005), target log CO(GT), 48 labeled support hours per week; fine-tuning used only weeks before 1 Oct 2004. Brackets: number of pretraining seeds averaged. Gains are paired over test weeks (positive = better than EM-Gaussian).**

| Method | NLL k=0 | NLL k=2 | gain vs EM, k=2 [95% CI] | gain vs EM, k=3 [95% CI] | weeks better (k=2) |
|---|---:|---:|---:|---:|---:|
| EM-Gaussian, closed form (reference) | 0.034 | 0.139 | – | – | – |
| FA-Gaussian, closed form | 0.056 | 0.158 | -0.019 [-0.039, +0.001] | -0.008 [-0.015, -0.000] | 5/23 |
| Complete-case ridge | 0.207 | 0.212 | -0.073 [-0.136, -0.009] | -0.039 [-0.076, -0.001] | 5/23 |
| Repo ridge, mean-imputed | 0.268 | 0.303 | -0.163 [-0.231, -0.095] | -0.117 [-0.169, -0.066] | 3/23 |
| Lifted cavity, synthetic pretrain + fine-tune [3] | -0.099 | -0.017 | +0.157 [+0.080, +0.233] | +0.162 [+0.095, +0.228] | 19/23 |
| Lifted cavity, fine-tune only (no pretrain) [1] | -0.049 | 0.008 | +0.131 [+0.062, +0.201] | +0.143 [+0.084, +0.201] | 19/23 |
| Lifted sites, no cavity input, fine-tuned [1] | -0.103 | -0.019 | +0.159 [+0.081, +0.236] | +0.163 [+0.096, +0.230] | 19/23 |
| 1-D sites (K=0), fine-tuned [1] | -0.036 | 0.002 | +0.137 [+0.070, +0.205] | +0.134 [+0.068, +0.200] | 18/23 |
| Residual MLP on same anchor, fine-tuned [3] | 0.038 | 0.153 | -0.014 [-0.041, +0.014] | -0.013 [-0.042, +0.015] | 10/23 |
| Residual MLP, fine-tune only [1] | 0.025 | 0.143 | -0.004 [-0.028, +0.020] | -0.005 [-0.032, +0.022] | 9/23 |
| Repo 1-D cavity (retrained), fine-tuned [1] | 0.167 | 0.243 | -0.104 [-0.153, -0.054] | -0.130 [-0.174, -0.087] | 5/23 |

**Table E. Same protocol, target log NO2(GT): the frozen confirmation endpoint R1 (one pretraining seed).**

| Method | NLL k=0 | NLL k=2 | gain vs EM, k=2 [95% CI] | gain vs EM, k=3 [95% CI] | weeks better (k=2) |
|---|---:|---:|---:|---:|---:|
| EM-Gaussian, closed form (reference) | -0.271 | -0.180 | – | – | – |
| FA-Gaussian, closed form | -0.318 | -0.182 | +0.002 [-0.008, +0.013] | -0.000 [-0.003, +0.002] | 12/23 |
| Complete-case ridge | -0.011 | -0.087 | -0.092 [-0.137, -0.048] | -0.036 [-0.056, -0.016] | 5/23 |
| Repo ridge, mean-imputed | -0.115 | -0.061 | -0.119 [-0.165, -0.073] | -0.065 [-0.098, -0.032] | 4/23 |
| Lifted cavity, synthetic pretrain + fine-tune [1] | -0.374 | -0.291 | +0.112 [+0.050, +0.173] | +0.123 [+0.057, +0.188] | 17/23 |
| Lifted cavity, fine-tune only (no pretrain) [1] | -0.388 | -0.280 | +0.100 [+0.052, +0.148] | +0.106 [+0.053, +0.159] | 20/23 |
| Residual MLP on same anchor, fine-tuned [1] | -0.325 | -0.175 | -0.004 [-0.050, +0.042] | -0.014 [-0.065, +0.038] | 9/23 |

