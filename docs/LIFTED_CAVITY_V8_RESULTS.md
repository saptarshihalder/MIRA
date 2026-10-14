# Protocol v8 results: neural-process and kriging baselines on six real networks (October 12, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v8.json`, computed by `confirm8.py`.

**Protocol history:**
- protocol: c1853a8, committed before any baseline training;
- kriging implementation note: f2194ab, which recomputed all kriging runs.

**Compute:** Colab L4, then A100.

**Data:** METR-LA, PEMS-BAY, Intel, PeMS03, PeMS08 and China air; 179 network-window clusters; seeds 1–3.

## Endpoints (gain = baseline NLL − UPT NLL; positive favours UPT)

| ID | Comparison | All sensors | 3 dropped | 6 dropped |
|---|---|---|---|---|
| R1 | UPT vs TNP-D | **+0.382 [+0.350, +0.415] pass** | +0.527 [+0.487, +0.566] pass | +1.046 [+0.960, +1.132] pass |
| R2 | UPT vs CNP | +1.521 [−0.253, +3.294] **fail** | +0.782 [+0.534, +1.029] pass | +1.004 [+0.740, +1.268] pass |
| R3 | UPT vs calibrated IGNNK kriging | +1.014 [+0.872, +1.157] pass | +1.037 [+0.889, +1.185] pass | +1.324 [+1.137, +1.511] pass |

**Why R2 fails at all sensors.** CNP diverges on Intel (NLL 6.48), so the network-day intervals become very wide. On
the other five networks CNP is uniformly worse than UPT.

## Mean test NLL, all sensors (seed averages; lower is better)

| Network | UPT | PFN | TNP-D | CNP | Calibrated kriging | Plain kriging |
|---|---|---|---|---|---|---|
| METR-LA | **−1.709** | −1.455 | −1.366 | −1.249 | −0.505 | −1.220 |
| PEMS-BAY | **−2.490** | −2.357 | −2.107 | −1.765 | −1.460 | −1.058 |
| Intel | **−0.303** | −0.215 | +0.027 | +6.476 | +0.936 | +32.64 |
| PeMS03 | **−1.198** | −1.013 | −0.931 | −0.819 | −0.419 | −0.332 |
| PeMS08 | **−1.884** | −1.100 | −1.066 | −0.991 | −0.683 | −0.314 |
| China air | **−0.166** | −0.105 | −0.051 | +0.029 | +0.173 | +0.414 |

## Other metrics, all sensors (UPT / PFN / TNP-D)

| Network | CRPS | RMSE | 90% coverage |
|---|---|---|---|
| METR-LA | .042 / .042 / .046 | .136 / **.122** / .132 | **.911** / .940 / .938 |
| PEMS-BAY | .019 / .019 / .024 | .063 / **.060** / .077 | **.902** / .942 / .932 |
| Intel | .221 / **.173** / .235 | 1.067 / **.827** / 1.012 | .869 / .947 / .926 |
| PeMS03 | .063 / .062 / .066 | .170 / **.164** / .175 | **.906** / .943 / .932 |
| PeMS08 | .050 / .051 / .052 | .146 / **.132** / .135 | .879 / .942 / .941 |
| China air | .132 / .129 / .135 | .287 / **.283** / .290 | **.888** / .910 / .906 |

## Reading

1. **Neural processes.**
   - UPT beats TNP-D, the strongest standard NP, on every network and at every sensor-loss level. The gap grows
     with sensor loss: +0.38 → +0.53 → +1.05.
   - TNP-D is a credible baseline: on every network it is close to the matched transformer.
2. **Kriging.** In-context methods beat historically trained kriging by a wide margin under drift, even after
   per-episode recalibration. Our IGNNK is a local-graph, row-wise adaptation (see protocol v8). A full-graph,
   time-window model (IGNNK or GRIN) was not run.
3. **What UPT improves.**
   - **Better:** the predictive distribution. NLL is best everywhere, and 90% coverage is closest to nominal on most
     networks; PFN over-covers.
   - **Not better:** point accuracy. PFN has slightly lower RMSE on every network, and CRPS is roughly tied, with
     Intel worse for UPT.
   - **Framing for the paper:** calibrated uncertainty and robustness to sensor loss, not sharper point predictions.
4. **Pre-registered failure.** R2 at all sensors fails, driven by CNP's divergence on Intel.

## UPT-L (protocol v7 secondary endpoints, now run)

**Endpoints** (PeMS03, PeMS08, China air; 63 clusters):

| ID | Comparison | Gain | 95% CI | Result |
|---|---|---|---|---|
| S1 | UPT-L vs TabPFN v2, extra 0 | **+0.313** | [+0.253, +0.372] | pass |
| S1 | UPT-L vs TabPFN v2, extra 3 | +0.505 | [+0.450, +0.561] | pass |
| S1 | UPT-L vs TabPFN v2, extra 6 | +1.073 | [+1.017, +1.128] | pass |
| S2 | UPT-L vs UPT, extra 0 | +0.230 | [+0.174, +0.286] | pass |

**Mean test NLL, all sensors:**

| Network | UPT-L | UPT | TabPFN v2 |
|---|---|---|---|
| PeMS03 | **−1.278** | −1.198 | −1.040 |
| PeMS08 | **−2.428** | −1.884 | −1.820 |
| China air | −0.221 | −0.166 | **−0.237** |

**Reading.** Scale helps: the 2.1M-parameter UPT beats the pretrained TabPFN v2 even with every sensor present on
the pooled v7 networks, while TabPFN stays slightly better on China air. This is a secondary endpoint, on the v7
networks only.
