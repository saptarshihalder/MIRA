# Lifted cavity: protocol v15 results (learning-curve-aware stacking, UPT-SF ⊕ TabPFN-3.5)

Protocol: `docs/LIFTED_CAVITY_PROTOCOL_V15.md` (committed before any 4-fold output). Development selection:
`artifacts/reports/lifted_cavity_paper/runs/selection_v15.json`. Confirmation: `confirm15.py` →
`artifacts/reports/lifted_cavity_paper/runs/confirm_v15.json`. Inputs: TabPFN-3.5's 4-fold cross-fit of the v10
supports on Modal (one L4, 8 processes, 46 min; `artifacts/v15_tabpfn35_cf4/`), UPT-SF's 4-fold cross-fits and the
development cross-fits on Colab (one L4). October 13–14, 2026.

**Outcome:** the selected rule (P3: 4-fold cross-fit, wide weight grid, UPT-SF seed ensemble) passes V1 and V2 with
larger gains than v13, and improves on the v13 pool at every condition. V3 (non-inferiority on every network at every
condition) fails: 9 of 12 tests pass; the pool stays 0.004–0.008 nats behind TabPFN-3.5 on the rivers with sensors
dropped.

## Development (v9 networks, every second episode; mean NLL, TabPFN-defined episodes)

| Rule | all sensors | 3 dropped | 6 dropped | mean |
|---|---|---|---|---|
| TabPFN-3.5 alone | −1.1563 | −1.1070 | −0.9864 | −1.0832 |
| v13ref (2-fold, G0, per seed) | −1.1888 | −1.1251 | −0.9938 | |
| P0 (2-fold, G1, ensemble; descriptive) | −1.1941 | −1.1287 | −0.9954 | |
| P1 (4-fold, G0, per seed) | −1.1924 | −1.1311 | −1.0023 | −1.1086 |
| P2 (4-fold, G1, per seed) | −1.1936 | −1.1329 | −1.0043 | −1.1103 |
| **P3 (4-fold, G1, ensemble; selected)** | **−1.1989** | **−1.1373** | **−1.0079** | **−1.1147** |
| P4 (extrapolated, G1, ensemble) | −1.1964 | −1.1357 | −1.0076 | −1.1132 |

On the development river network `usgs_q`, the pool's deficit against TabPFN-3.5 fell from +0.015 / +0.025 / +0.038
(v13 rule) to −0.001 / +0.005 / +0.010 (P3).

## Confirmation (v10 networks, every episode; 5,303 TabPFN-defined episodes, 112 network-window clusters)

| ID | Comparison | Gain (nats) [95% CI] | Result |
|---|---|---|---|
| V1 | P3 pool vs TabPFN-3.5, all sensors | +0.055 [+0.040, +0.070] | pass |
| V2 | the same, 3 extra dropped | +0.040 [+0.028, +0.052] | pass |
| V2 | the same, 6 extra dropped | +0.040 [+0.027, +0.054] | pass |
| V3 | non-inferiority (lower bound > −0.005) on each network × condition | 9 of 12 pass | fail |

V3 failures: `usgs_ny` with 3 dropped (−0.004 [−0.006, −0.003]) and 6 dropped (−0.008 [−0.010, −0.006]), `usgs_wa`
with 6 dropped (−0.004 [−0.006, −0.003]).

**Secondary.**
- P3 vs the v13 pool: +0.013 [+0.012, +0.014], +0.014 [+0.013, +0.015], +0.015 [+0.014, +0.017] (all above 0).
- Squared error vs TabPFN-3.5: +0.0021, +0.0022, +0.0030, all intervals above 0.
- v13ref reproduces the v13 pool exactly (max |difference| = 0.0).

## Per network (mean NLL; TabPFN-3.5 / v13 pool / v15 pool; weight on UPT-SF)

| Network | all sensors | 3 dropped | 6 dropped | weight |
|---|---|---|---|---|
| usgs_ny | −4.371 / −4.356 / −4.370 | −4.314 / −4.292 / −4.309 | −4.168 / −4.139 / −4.160 | 0.05–0.07 |
| usgs_wa | −5.766 / −5.748 / −5.763 | −5.698 / −5.678 / −5.695 | −5.542 / −5.521 / −5.538 | 0.02–0.03 |
| dwd_p | −0.469 / −0.643 / **−0.657** | −0.400 / −0.529 / **−0.541** | −0.152 / −0.297 / **−0.312** | 0.77–0.88 |
| aurn_pm25 | −0.726 / −0.768 / **−0.778** | −0.672 / −0.703 / **−0.712** | −0.564 / −0.584 / **−0.591** | 0.49–0.58 |

## Reading

The two-fold cross-fit's bias toward the flat-learning-curve partner was real: fitting on 36 rows instead of 24
roughly halves the weight on UPT-SF on the rivers and cuts the pool's river losses by 75–95%, while the gains on
pressure and PM2.5 grow. The pool now beats TabPFN-3.5 by 0.04–0.06 nats pooled at every condition, is within
0.008 nats of it on every river condition, but is not non-inferior there at the pre-registered margin.
