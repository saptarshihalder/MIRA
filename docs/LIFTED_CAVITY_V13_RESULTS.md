# Lifted cavity: protocol v13 results (UPT-SF ⊕ TabPFN-3.5)

Protocol: `docs/LIFTED_CAVITY_PROTOCOL_V13.md` (committed before any TabPFN-3.5 output on real data; amendment 1,
execution only). Endpoints: `confirm13.py` → `artifacts/reports/lifted_cavity_paper/runs/confirm_v13.json`.

TabPFN-3.5 (September 2026; tabpfn 9.1.0, `ModelVersion.V3_5`, 8 estimators) ran on Modal L4 GPUs: `usgs_ny` in its
own container (47 min), the other three networks together in one container (47 min). Outputs:
`artifacts/v13_tabpfn35/<net>/`. Pools and endpoints ran on Colab (`run_v13.py`).

## Endpoints (four v10 networks pooled; 5,303 TabPFN-defined episodes, 43 excluded; 112 network-window clusters)

| ID | Comparison | Gain (nats) [95% CI] | Result |
|---|---|---|---|
| Z1 | UPT-SF ⊕ TabPFN-3.5 vs TabPFN-3.5, all sensors | +0.042 [+0.027, +0.057] | pass |
| Z2 | the same, 3 extra dropped | +0.026 [+0.014, +0.038] | pass |
| Z2 | the same, 6 extra dropped | +0.025 [+0.011, +0.039] | pass |
| Z3 | UPT-SF ⊕ TabPFN-3.5 vs UPT-SF, all sensors | +1.327 [+1.127, +1.527] | pass |

**Secondary.**
- Partner check: UPT-SF ⊕ TabPFN-3.5 beats PFN-L ⊕ TabPFN-3.5 by +0.047 [+0.034, +0.061], +0.045 [+0.031, +0.059]
  and +0.069 [+0.044, +0.095] at 0, 3 and 6 dropped.
- Squared error vs TabPFN-3.5: small gains (+0.0016, +0.0016, +0.0020), all intervals above 0.
- Per network: the pool beats TabPFN-3.5 on `dwd_p` (+0.17, +0.13, +0.15) and `aurn_pm25` (+0.04, +0.03, +0.02),
  and loses to it on the two river networks (`usgs_ny` −0.015 to −0.029, `usgs_wa` −0.017 to −0.021).

## Descriptive (mean NLL, TabPFN-defined episodes)

| Network | Inputs | TabPFN v2 | TabPFN-3.5 | UPT-SF | UPT-SF ⊕ TabPFN v2 | UPT-SF ⊕ TabPFN-3.5 | PFN-L ⊕ TabPFN-3.5 | weight on UPT-SF |
|---|---|---|---|---|---|---|---|---|
| usgs_ny | all | −3.634 | **−4.371** | −2.287 | −3.699 | −4.356 | −4.351 | 0.09 |
| usgs_ny | −6 | −2.879 | **−4.168** | −1.980 | −3.146 | −4.139 | −4.131 | 0.10 |
| usgs_wa | all | −4.971 | **−5.766** | −3.174 | −5.048 | −5.748 | −5.745 | 0.05 |
| usgs_wa | −6 | −4.414 | **−5.542** | −2.846 | −4.571 | −5.521 | −5.521 | 0.05 |
| dwd_p | all | −0.152 | −0.469 | −0.566 | −0.595 | **−0.643** | −0.472 | 0.71 |
| dwd_p | −6 | +1.462 | −0.152 | −0.218 | −0.228 | **−0.297** | −0.020 | 0.85 |
| aurn_pm25 | all | −0.432 | −0.726 | −0.405 | −0.561 | **−0.768** | −0.744 | 0.51 |
| aurn_pm25 | −6 | +0.074 | −0.564 | −0.207 | −0.352 | **−0.584** | −0.573 | 0.45 |

## Reading (as fixed in the protocol)

Z1 and Z2 pass: the pool's gain is not specific to an older foundation model. TabPFN-3.5 is far stronger than TabPFN
v2 (by 0.3–1.3 nats) and degrades much less when sensors fail, so the gains are smaller than with v2 (v11: X1 +0.171)
and come from the non-river networks. On pressure, UPT-SF alone is better than TabPFN-3.5 at every condition, and a
plain transformer of the same size is a much weaker partner (−0.472 vs −0.643). On the rivers, the per-episode
cross-fit stacking puts 5–10% weight on UPT-SF and costs about 0.02 nats against TabPFN-3.5 alone.
