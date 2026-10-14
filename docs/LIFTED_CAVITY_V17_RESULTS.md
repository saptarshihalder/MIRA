# Lifted cavity: protocol v17 results (combined fresh test after TabPFN-3.5's release)

Protocol: `docs/LIFTED_CAVITY_PROTOCOL_V17.md` (committed before any v17 download, panel or score). Confirmation:
`confirm17.py` → `artifacts/reports/lifted_cavity_paper/runs/confirm_v17.json`. Data: v9's three networks
(`dwd_t2m`, `aurn_no2`, `usgs_q`), days 2026-09-10..10-12 (after TabPFN-3.5's release on 2026-09-09), 48 new targets
per network never used by v9; 908 episodes, 19 network-window clusters (`data_snapshots/v17`, panels with digests in
`artifacts/v17_tabpfn35/`). Combined with v16's four networks: 2,344 episodes, 49 clusters. TabPFN-3.5 on Modal (one
L4 container per batch: the two weekly networks in about 8 min; the river panel's 528 episodes in 17 min, its 4-fold
cross-fit in 20 min, concurrently), UPT-SF on Colab (18 jobs of 2–7 min, eight at a time), pools and confirmation on
Colab. October 14, 2026.

**Outcome:** G1 passes; G2 passes with 3 inputs dropped and misses with 6 dropped (lower bound −0.0016).

## Endpoints (seven networks pooled; 2,327 TabPFN-defined episodes, 17 excluded; 49 clusters; 97.5% intervals)

| ID | Comparison | Gain (nats) [97.5% CI] | Result |
|---|---|---|---|
| G1 | frozen v15 pool vs TabPFN-3.5, all sensors | +0.031 [+0.011, +0.051] | pass |
| G2 | the same, 3 extra dropped | +0.020 [+0.003, +0.037] | pass |
| G2 | the same, 6 extra dropped | +0.017 [−0.002, +0.036] | fail |

**Secondary.**
- v17's three networks alone (95%; 908 episodes, 1 excluded, 19 clusters): +0.027 [+0.010, +0.044], +0.016 [+0.003,
  +0.029], +0.009 [−0.003, +0.021].
- vs the v13 rule (2-fold stacking), seven networks: +0.014 [+0.012, +0.016], +0.016 [+0.014, +0.018], +0.018
  [+0.015, +0.021].
- vs UPT-SF alone, seven networks: +1.55, +1.58, +1.63 (all intervals above 0).
- Squared error vs TabPFN-3.5, seven networks: +0.0059 [+0.0012, +0.0105], +0.0049 [+0.0011, +0.0086], +0.0045
  [+0.0013, +0.0077].
- Per v17 network vs TabPFN-3.5 (all / 3 / 6 dropped): dwd_t2m_f +0.090 / +0.063 / +0.049; aurn_no2_f +0.025 /
  +0.025 / +0.018; usgs_q_f +0.004 / −0.004 / −0.009 (the last interval, [−0.016, −0.002], is below 0).
  Non-inferiority (lower bound > −0.005) holds for 7 of 9 network-conditions; it fails on the river with inputs
  dropped, as in v15 and v16.

## Descriptive (mean NLL, TabPFN-defined episodes; weight on UPT-SF)

| Network | TabPFN-3.5 | UPT-SF | v13 rule | v15 rule | w |
|---|---|---|---|---|---|
| dwd_t2m_f, all | 1.102 | 1.018 | 1.019 | **1.013** | 0.75 |
| dwd_t2m_f, 6 dropped | 1.246 | 1.227 | 1.201 | **1.197** | 0.69 |
| aurn_no2_f, all | 0.220 | 0.250 | 0.202 | **0.195** | 0.61 |
| aurn_no2_f, 6 dropped | 0.373 | 0.425 | 0.357 | **0.355** | 0.45 |
| usgs_q_f, all | −4.542 | −2.674 | −4.530 | **−4.546** | 0.08 |
| usgs_q_f, 6 dropped | **−4.320** | −2.352 | −4.286 | −4.311 | 0.09 |

## Reading

At the stricter level that a second look requires, the frozen rule beats TabPFN-3.5 on every fresh, post-release task
of seven networks with all sensors (+0.031) and with three inputs dropped (+0.020), the test that v16 alone (30
clusters) could not resolve. With six dropped the gain (+0.017) is positive but its interval reaches zero. The pattern
of v13–v16 repeats on v9's networks: large gains where UPT-SF carries information TabPFN-3.5 lacks (temperature,
where UPT-SF alone again beats TabPFN-3.5, as on pressure), gains on NO₂ (where the pool beats both models), a tie on
the river with all sensors and small river losses (at most 0.009 nats) with inputs dropped. The v15 rule again beats
the v13 rule everywhere.
