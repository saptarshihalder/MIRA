# Lifted cavity: protocol v14 results (TabPFN-3.5 combined with UPT-SF beyond a linear pool)

Protocol: `docs/LIFTED_CAVITY_PROTOCOL_V14.md` with amendment 1 (both committed before any v14 run). Development
selection: `artifacts/reports/lifted_cavity_paper/runs/selection_v14.json`. Confirmation: `confirm14.py` →
`artifacts/reports/lifted_cavity_paper/runs/confirm_v14.json`. All runs on Colab (one L4), October 13, 2026.

**Outcome:** the selected variant, F5 (UPT-SF as a Gaussian site on TabPFN-3.5's predictive, a logarithmic pool with
per-episode cross-fitted exponents), beat TabPFN-3.5 on the development networks but **failed every confirmatory
endpoint** on the v10 networks. The linear pool of protocol v13 remains the combination that improves TabPFN-3.5.

## Development (v9 networks, every second episode; mean NLL, TabPFN-defined episodes)

| Variant | dwd_t2m all / −3 / −6 | aurn_no2 all / −3 / −6 | usgs_q all / −3 / −6 | mean (9 cells) |
|---|---|---|---|---|
| TabPFN-3.5 alone | 1.114 / 1.146 / 1.248 | 0.238 / 0.296 / 0.390 | −4.332 / −4.275 / −4.116 | −1.083 |
| F1 mean column | 1.136 / 1.161 / 1.255 | 0.271 / 0.308 / 0.400 | −4.296 / −4.242 / −4.092 | −1.062 |
| F3 mean + log-sd columns | 1.130 / 1.163 / 1.260 | 0.269 / 0.312 / 0.403 | −4.281 / −4.223 / −4.076 | −1.055 |
| F4 residual | 1.106 / 1.157 / 1.264 | 0.279 / 0.335 / 0.428 | −2.851 / −2.769 / −2.526 | −0.510 |
| **F5 site (selected)** | 1.063 / 1.107 / 1.215 | 0.223 / 0.284 / 0.384 | −4.315 / −4.264 / −4.114 | **−1.096** |
| v13 linear pool (descriptive) | 1.024 / 1.081 / 1.196 | 0.209 / 0.275 / 0.376 | −4.317 / −4.250 / −4.077 | |

Pooled over the development networks, F5 beat TabPFN-3.5 at every condition (−1.171 vs −1.156, −1.119 vs −1.107,
−0.999 vs −0.986), so the rule allowed confirmation. The input-column variants (F1, F3) and the residual variant (F4)
were worse than TabPFN-3.5 alone.

## Confirmation (v10 networks, every episode; 5,303 TabPFN-defined episodes, 112 network-window clusters)

| ID | Comparison | Gain (nats) [95% CI] | Result |
|---|---|---|---|
| R1 | F5 vs TabPFN-3.5, all sensors | +0.007 [−0.013, +0.027] | fail |
| R2 | the same, 3 extra dropped | −0.011 [−0.060, +0.039] | fail |
| R2 | the same, 6 extra dropped | +0.012 [−0.022, +0.047] | fail |

**Secondary.**
- F5 vs the v13 linear pool: −0.035 [−0.050, −0.020], −0.036 [−0.084, +0.011], −0.013 [−0.043, +0.018] (the linear
  pool is better).
- F5 vs TabPFN-3.5 on squared error and CRPS: small gains at every condition, all intervals above 0 (CRPS +0.003,
  +0.003, +0.004).
- Per network (NLL, all sensors): `dwd_p` +0.132, `aurn_pm25` −0.013, `usgs_wa` −0.021, `usgs_ny` −0.056; with three
  dropped, `usgs_ny` −0.118 [−0.295, +0.059].

## Reading

The logarithmic pool can be sharper than both members, but it has no per-query loss bound: when the cross-fitted
exponents over-trust UPT-SF (on rivers, where TabPFN-3.5 fitted on 24 cross-fit rows is much weaker than on 48), it
loses up to 0.12 nats. The linear pool loses at most −log(1 − w) per query (Proposition 5) and passed its v13 test.
Feeding UPT-SF's prediction to TabPFN as an input did not help either. Within this study, the linear pool is the
combination of a structured model with a foundation model that holds up under confirmation.
