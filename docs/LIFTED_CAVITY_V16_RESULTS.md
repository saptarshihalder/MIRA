# Lifted cavity: protocol v16 results (fresh test after TabPFN-3.5's release)

Protocol: `docs/LIFTED_CAVITY_PROTOCOL_V16.md` (committed before any panel or score). Confirmation: `confirm16.py` →
`artifacts/reports/lifted_cavity_paper/runs/confirm_v16.json`. Data: v10's four networks, days 2026-09-10..10-12
(after TabPFN-3.5's release on 2026-09-09), 48 new targets per network never used by v10; 1,436 episodes, 30
network-window clusters (`data_snapshots/v16`, panels with digests in `artifacts/v16_tabpfn35/`). TabPFN-3.5 on Modal
(two L4 containers, 10 and 25 min, after a first attempt failed downloading the weights in parallel; weights now cached
once), UPT-SF on Colab (24 jobs, about 15 min). October 14, 2026.

**Outcome:** F1 passes; both F2 tests miss (lower bounds −0.0001 and −0.004).

## Endpoints (four networks pooled; 1,420 TabPFN-defined episodes, 16 excluded; 30 clusters)

| ID | Comparison | Gain (nats) [95% CI] | Result |
|---|---|---|---|
| F1 | frozen v15 pool vs TabPFN-3.5, all sensors | +0.034 [+0.007, +0.061] | pass |
| F2 | the same, 3 extra dropped | +0.023 [−0.0001, +0.045] | fail |
| F2 | the same, 6 extra dropped | +0.022 [−0.004, +0.048] | fail |

**Secondary.**
- vs the v13 rule (2-fold stacking): +0.016 [+0.013, +0.018], +0.017 [+0.015, +0.019], +0.019 [+0.016, +0.023].
- vs UPT-SF alone: +1.84, +1.87, +1.93 (all intervals above 0).
- Squared error vs TabPFN-3.5: +0.0012, +0.0012, +0.0017 (intervals above or at 0).
- Per network vs TabPFN-3.5 (all / 3 / 6 dropped): dwd_p_f +0.199 / +0.159 / +0.178; aurn_pm25_f +0.050 / +0.034 /
  +0.030; usgs_ny_f +0.002 / −0.005 / −0.008; usgs_wa_f −0.0002 / −0.004 / −0.007. Non-inferiority (lower bound >
  −0.005) holds for 8 of 12 network-conditions; it fails on both rivers with inputs dropped.

## Descriptive (mean NLL, TabPFN-defined episodes; weight on UPT-SF)

| Network | TabPFN-3.5 | UPT-SF | v13 rule | v15 rule | w |
|---|---|---|---|---|---|
| usgs_ny_f, all | −5.103 | −2.788 | −5.088 | **−5.104** | 0.05 |
| usgs_wa_f, all | **−6.007** | −3.459 | −5.990 | **−6.007** | 0.04 |
| dwd_p_f, all | −0.646 | −0.740 | −0.827 | **−0.845** | 0.73 |
| aurn_pm25_f, all | −0.745 | −0.292 | −0.787 | **−0.794** | 0.54 |

## Reading

On prediction tasks that no protocol had scored, on days that TabPFN-3.5 cannot have seen, the frozen rule beats
TabPFN-3.5 with all sensors (+0.034) and the pattern of v13–v15 repeats: large gains where UPT-SF carries information
TabPFN-3.5 lacks (pressure, where UPT-SF alone is again better than TabPFN-3.5; PM2.5), and a tie on rivers with all
sensors. With inputs dropped the pooled gains (+0.022, +0.023) are as large as v15 predicted but, with only 30
clusters of post-release data, their intervals reach 0. The v15 rule again beats the v13 rule everywhere.
