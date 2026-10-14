# Lifted cavity: protocol v7: UPT vs TabPFN v2 under sensor loss, on three new networks

Committed October 11, 2026. At commit time:
- no model of this project has been fine-tuned on, or scored on, PeMS district 3, PeMS district 8 or the China air
  network;
- no v7 panel exists;
- the only looks at these data were shapes, missing fractions, value ranges and episode counts, from the pre-panel
  loader check;
- no UPT-L model has been trained.

## Why v7

Protocol v6 confirmed UPT against a matched transformer (P1), a two-seed transformer ensemble (P2) and on width
extrapolation (P4). It failed against the pretrained TabPFN v2 with every sensor present (P3). Descriptively, the
ordering reversed when sensors were removed: UPT was better than TabPFN at +6 on all three v6 networks.

v7 tests that reversal on fresh data, and checks two v6 limitations:
- float32 site algebra;
- small scale.

## Models

- **UPT:** the v6 source checkpoints `upt_s1`–`upt_s3`, unchanged.
- **PFN:** the matched transformer, source checkpoints `pfn_s1`–`pfn_s3` (as in v6).
- **Fine-tuning:** each model is fine-tuned separately per network with the protocol-v2 recipe, as in v6.
- **TabPFN v2:** in context, 8 estimators, native missing values. Amendment-3 handling applies: episodes with a
  constant support target are undefined and excluded.
- **UPT64:** the same fine-tuned UPT checkpoints, scored with the lifted-site algebra in float64
  (`eval_real.py --site-float64`). It has no new parameters and no retraining.
- **UPT-L (secondary):** the UPT head on the v4 PFN-L backbone.
  - Architecture: d = 128, 8 layers, 8 heads, feed-forward width 512.
  - Recipe: 40,000 steps, 32 tasks × 16 queries, lr 5e-4, warm-up 2,000, as v4.
  - Seeds 1–3, trained on Colab.
  - Fine-tuned with the same recipe as UPT.

## Networks (`realdata_v7.py`; panels built with `eval_real.py build --seed 7071`)

The task is the same as v6: in-context virtual sensing of one target from 11 input sensors, with 48 support and 48
query rows per episode.

| Network | Readings | Missing | Inputs | Source | Test | Test targets | Test episodes |
|---|---|---|---|---|---|---|---|
| PeMS03 (Sacramento) | log1p flow, 358 sensors, 5-min | none; 20% simulated support dropout | 11 most correlated over the source period | 2018-09-01 → 10-31 | 11-01 → 11-30 (daily) | 24 | 720 |
| PeMS08 (San Bernardino) | log1p flow, 170 sensors, 5-min | none; 20% simulated support dropout | same | 2016-07-01 → 08-10 | 08-11 → 08-31 (daily) | 24 | 504 |
| China air (outside Beijing) | log PM2.5, 401 stations, hourly | natural | 11 nearest stations | 2014-05-01 → 2015-01-31 | 2015-02-01 → 04-30 (weekly) | 48 | 560 |

The extra dropped-sensor conditions are 0, 3 and 6 (`realdata.condition_masks`, seeded).

## Endpoints (`confirm7.py`)

**Common settings:**
- Per-episode NLL is pooled over the three networks.
- Learned models use the seed average over seeds 1–3.
- Paired 95% intervals are cluster-robust over network-windows.
- TabPFN comparisons use only the episodes where TabPFN is defined.

Gains are in nats, positive when UPT is better.

| ID | Comparison | Condition | Pass condition |
|---|---|---|---|
| Q1 | UPT vs TabPFN v2 | 3 extra sensors dropped | gain ≥ .01 and lower bound > 0 |
| Q2 | UPT vs TabPFN v2 | 6 extra sensors dropped | gain ≥ .01 and lower bound > 0 |
| Q3 | UPT vs fine-tuned PFN | natural (extra = 0) | gain ≥ .01 and lower bound > 0 (replicates v6 P1) |
| Q4 | UPT64 vs UPT | natural | lower bound > −.01 (float64 sites do not hurt) |

**The headline claim** ("UPT is more robust to sensor loss than TabPFN v2") requires both Q1 and Q2 to pass. Because
both must pass (an intersection-union test), no multiplicity correction is applied.

**Secondary:**
- **S1:** UPT-L vs TabPFN v2 at extra 0, 3 and 6, with the same statistics. It is reported as "not run" if UPT-L
  does not finish.
- **S2:** UPT-L vs UPT at extra 0.

**Descriptive:**
- each network and condition, for every model;
- the number of episodes with UPT NLL above 10 (float32 vs float64);
- TabPFN episodes excluded as undefined.

All endpoint outcomes, including failures, are reported. Nothing is tuned on these panels.
