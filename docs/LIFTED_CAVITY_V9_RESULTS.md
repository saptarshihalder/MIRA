# Protocol v9 results: three 2026 networks, scale, and two small changes (October 12, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v9.json`, computed by `confirm9.py`.

**Protocol history:**
- protocol: 4b4a026;
- amendment 1 (PFN-L trained here with the v4 recipe): 1017398.

Both were committed before any v9 panel, fine-tune or score existed.

**Compute:** Colab A100 80 GB, about 8 hours of wall clock.

**Data:** 3,903 test episodes in 82 network-windows, test period April–September 2026.

| Network | Sensors | Episodes |
|---|---|---|
| DWD hourly temperature | 487 stations | 1,247 |
| UK AURN hourly NO2 | 137 sites | 1,220 |
| USGS Pennsylvania 30-min discharge | 304 gauges | 1,436 |

**Operational notes.** These changed only the scheduling, not any recipe:
- TabPFN on `usgs_q` ran in parallel to the chain, with the identical command.
- The fine-tunes of chain A's PFN-L stage were started early.
- Chain A was replaced by a finisher that waited for both and then ran the remaining stages unchanged.
- One browser-side Colab disconnect was recovered without a runtime restart.

## Variant selection (development data, before any v9 score; `select9.py`)

| Candidate | NLL | CRPS / σ |
|---|---|---|
| A: UPT-L | −1.2712 | 0.12075 |
| B: + `tta8` | −1.2783 | 0.12070 |
| C: + CRPS objective | −1.2675 | 0.12049 |
| **D: + CRPS objective + `tta8` (selected)** | −1.2737 | **0.12043** |

All four candidates were within .02 nats of the best NLL, so the lowest CRPS chose D, which is **UPT\***.

The two changes are real but small: −0.3% CRPS and −0.003 nats against A. They do not change any conclusion.

## Endpoints (gain = baseline − UPT\*; positive favours UPT\*)

| ID | Comparison | Gain | 95% CI | Result |
|---|---|---|---|---|
| V1 | UPT\* vs TabPFN v2, NLL, all sensors | −0.374 | [−0.543, −0.205] | **fail** |
| V2 | UPT\* vs TabPFN v2, NLL, 3 dropped | −0.284 | [−0.440, −0.128] | **fail** |
| V2 | UPT\* vs TabPFN v2, NLL, 6 dropped | +0.077 | [−0.088, +0.243] | **fail** |
| V3 | UPT\* vs PFN-L (matched scale; `tta8`), NLL | **+0.029** | [+0.012, +0.047] | **pass** |
| V4 | UPT\* vs TNP-D, NLL | **+0.258** | [+0.194, +0.321] | **pass** |
| V5 | UPT\* vs calibrated IGNNK kriging, NLL | **+0.708** | [+0.552, +0.863] | **pass** |
| V6 | UPT\* vs TabPFN v2, CRPS / σ | **+0.0123** | [+0.0081, +0.0164] | **pass** |

- Eight `usgs_q` episodes with a constant support target are undefined for TabPFN and are excluded from the TabPFN
  comparisons.
- PFN-L comparator: PFN-L with `tta8`, which had a lower NLL than PFN-L-CRPS.

## Each network against TabPFN v2, all sensors (secondary)

| Network | NLL gain | CRPS / σ gain |
|---|---|---|
| DWD temperature | **+0.290** [+0.278, +0.301] | **+0.028** |
| AURN NO2 | **+0.076** [+0.068, +0.084] | **+0.024** |
| USGS discharge | −1.339 [−1.483, −1.194] | −0.012 |

**The pooled V1 and V2 failures come entirely from the river network.** UPT\* beats TabPFN v2 on weather and air
quality at every sensor-loss level. On USGS, TabPFN's NLL is far better, while its standardized RMSE is about the
same: 0.273 for TabPFN against 0.270 for UPT\*.

## Diagnosis of the USGS gap (post hoc, exploratory)

NLL on USGS at all sensors, by the episode's support-target standard deviation σ (log1p discharge units):

| σ | Episodes | UPT\* | PFN-L | TabPFN v2 | TabPFN advantage |
|---|---|---|---|---|---|
| < .02 | 61 | −3.34 | −2.55 | −8.12 | 4.78 |
| .02–.1 | 422 | −2.81 | −2.79 | −4.95 | 2.15 |
| ≥ .1 | 945 | −2.04 | −1.98 | −2.79 | 0.76 |

Two causes, both shared with the matched transformer, PFN-L:

1. **A target-scale floor.**
   - The CellPFN backbone standardizes the target by max(σ, .1), and the free head's log-variance is bounded at
     ±5.
   - The predictive standard deviation therefore cannot fall below e^(−2.5) × max(σ, .1).
   - 34% of USGS episodes have σ < .1; none of the DWD or AURN episodes do.
   - This was inherited from the synthetic prior, whose targets have unit scale.
   - Near-deterministic river flows are exactly where a sharp predictive is rewarded.
2. **Output flexibility.** Even above the floor, TabPFN's histogram output, a 5,000-bin bar distribution, is more
   peaked than a two-component Gaussian mixture at the same RMSE.

The same mechanism may explain some of the old networks where TabPFN keeps the better NLL at all sensors: PEMS-BAY
and METR-LA log speeds are nearly flat at night. This has not been checked. It is unlikely to explain China air. Making the output scale-equivariant (removing the .1 clamp) and adding a richer free head
would be the natural next changes. They need fresh data to confirm.

## Mean test NLL / CRPS / σ / standardized RMSE / 90% coverage, all sensors

| Network | UPT\* | PFN-L (tta8) | UPT | PFN | TNP-D | Calibrated kriging | TabPFN v2 |
|---|---|---|---|---|---|---|---|
| DWD | **1.033** / .102 / .206 / .89 | 1.038 / **.101** / **.204** / .90 | 1.088 | 1.070 | 1.185 | 1.390 | 1.322 / .130 / .254 / .87 |
| AURN | **.227** / .328 / .615 / .90 | .233 / **.326** / **.612** / .90 | .257 | .247 | .308 | .369 | .303 / .352 / .655 / .90 |
| USGS | −2.326 / .094 / .270 / .87 | −2.255 / .087 / **.256** / .93 | −1.969 | −2.000 | −1.824 | −0.823 | **−3.657** / **.082** / .273 / .91 |

The NLL-only columns give mean test NLL.

**Against the matched-scale transformer:**
- UPT\* has the better NLL on all three networks.
- PFN-L has slightly better CRPS and RMSE on all three: pooled −0.003 CRPS / σ and −0.004 squared error / σ².
- The point-accuracy gap to the matched transformer, seen since v6, **persists**. The two small changes did not
  close it.

## Development networks (descriptive): scale everywhere and the matched-scale comparison

Mean NLL, seed averages:

| Network | Extra dropped | UPT-L | PFN-L | TabPFN v2 |
|---|---|---|---|---|
| METR-LA | 0 / 3 / 6 | −1.774 / **−1.688** / **−1.492** | −1.473 / −1.366 / −1.078 | **−1.793** / −1.552 / −1.211 |
| PEMS-BAY | 0 / 3 / 6 | −2.629 / **−2.492** / **−2.244** | −2.412 / −2.295 / −2.008 | **−2.682** / −2.457 / −2.060 |
| Intel | 0 / 3 / 6 | −0.771 / **−0.496** / **+0.078** | −0.297 / +0.063 / +0.909 | **−1.305** / +0.821 / +2.362 |
| PeMS03 | 0 / 3 / 6 | **−1.278 / −1.161 / −0.989** | −1.030 / −0.939 / −0.802 | −1.040 / −0.812 / −0.038 |
| PeMS08 | 0 / 3 / 6 | **−2.428 / −2.308 / −1.957** | −1.119 / −1.098 / −1.012 | −1.820 / −1.504 / −0.624 |
| China air | 0 / 3 / 6 | −0.221 / **−0.143 / −0.006** | −0.118 / −0.050 / +0.059 | **−0.237** / +0.229 / +0.916 |

## Reading

1. **Architecture, not scale.**
   - At equal size and source budget, UPT beats the plain transformer on NLL on all 9 networks, at every
     sensor-loss level.
   - The margins are large on the old networks (+0.10 to +1.31 nats) and small but significant on the 2026 networks
     (V3: +0.029).
2. **Against other learned in-context methods and kriging.** UPT\* beats TNP-D, CNP and calibrated kriging on every
   metric (NLL, CRPS, RMSE), at every sensor-loss level, pooled over the 2026 networks (V4, V5 and the secondary
   comparisons).
3. **Against TabPFN v2, the pretrained foundation model.**
   - **Under sensor loss**, UPT-L or UPT\* has the better NLL on 8 of 9 networks; the river network is the
     exception.
   - **With all sensors**, it is better on 4 of 9 (PeMS03, PeMS08, DWD, AURN).
   - Pooled over the 2026 networks, UPT\* has the better CRPS (V6) and the better RMSE (squared error / σ²: +0.023,
     CI [+0.018, +0.029]).
   - TabPFN v2's NLL advantage sits in near-deterministic, low-variance episodes (diagnosis above).
4. **Pre-registered failures:** V1 and V2. The pooled NLL claim against TabPFN v2 on the 2026 networks does not
   hold, and it is not made.
5. **The two small changes** (test-time ensemble, CRPS objective) help marginally and change no conclusion.
