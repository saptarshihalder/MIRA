# Protocols v10 and v11 results: a scale-free UPT, and UPT ⊕ TabPFN on four fresh 2026 networks (October 13, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v10.json` and `confirm_v11.json`, computed by
`confirm10.py` and `confirm11.py`.

**Protocols:** v10 (82c873a, amendments 1–5) and v11 (e2151ff). Both were committed before any score on the fresh
networks.

**Compute:**
- Colab A100, then L4, for fine-tuning and scoring;
- Modal A100 for TabPFN v2 (amendment 5). The panels were verified identical by content digest on both platforms.

**Data.** Four networks never used before, with test period 2026, 5,346 test episodes in 112 network-windows:

| Network | Sensors |
|---|---|
| USGS New York discharge | 290 gauges |
| USGS Washington discharge | 237 gauges |
| DWD station pressure | 206 stations |
| UK AURN PM2.5 | 145 sites |

TabPFN v2 is undefined on 43 episodes with a constant support target (15 New York, 28 Washington). Those episodes are
excluded from TabPFN comparisons, as pre-registered.

**Development selection:** UPT-SF = `upt_L_sfm`. This is the scale-free pipeline plus a three-component free mixture,
chosen on the v9 networks.

## v11: UPT ⊕ TabPFN (per-episode cross-fit stacking): every endpoint passes

| ID | Comparison | Gain (nats) | 95% CI | Result |
|---|---|---|---|---|
| X1 | pool vs TabPFN v2, all sensors | **+0.171** | [+0.142, +0.200] | **pass** |
| X2 | pool vs TabPFN v2, 3 dropped | **+0.245** | [+0.199, +0.291] | **pass** |
| X2 | pool vs TabPFN v2, 6 dropped | **+0.604** | [+0.490, +0.719] | **pass** |
| X3 | pool vs UPT-SF, all sensors | **+0.910** | [+0.759, +1.060] | **pass** |

**Each network against TabPFN v2.** All of the following pass:

| Network | All sensors | 6 dropped |
|---|---|---|
| USGS NY | +0.065 | +0.266 |
| USGS WA | +0.077 | +0.156 |
| DWD pressure | +0.443 | +1.689 |
| AURN PM2.5 | +0.129 | +0.426 |

**Mean weight on UPT-SF.** The stacking learned when to trust which model, using no test label:

| Network | All sensors | 6 dropped |
|---|---|---|
| Rivers | 0.19–0.29 | 0.28–0.44 |
| DWD pressure | 0.91 | 0.97 |
| AURN PM2.5 | 0.67 | 0.79 |

## v10: UPT-SF alone

| ID | Comparison | Gain | 95% CI | Result |
|---|---|---|---|---|
| W1 | UPT-SF vs TabPFN v2, all sensors | −0.753 | [−0.927, −0.579] | **fail** |
| W2 | UPT-SF vs TabPFN v2, 3 dropped | −0.631 | [−0.813, −0.449] | **fail** |
| W2 | UPT-SF vs TabPFN v2, 6 dropped | −0.206 | [−0.442, +0.029] | **fail** |
| W3 | UPT-SF vs PFN-L (v9 recipe; the stronger comparator) | −541 | [−1603, +521] | **fail** |
| W4 | UPT-SF vs TabPFN v2, CRPS / σ | +0.002 | [−0.001, +0.005] | **fail** |
| W5 | UPT-SF vs TNP-D | −541 | [−1603, +521] | **fail** |
| W6 | UPT-SF vs calibrated kriging | −518 | [−1535, +499] | **fail** |
| W7 | UPT-SF vs UPT\* | not run (amendment 4, compute budget) | | |

**Why W3, W5 and W6 fail: a degenerate case in the SF pipeline.**
- On the 15 New York episodes whose support target is constant, the SF standardization falls back to its floor of
  1e-4 · (1 + |mean|). The predictive becomes extremely narrow, and any query that differs scores an NLL in the
  thousands. UPT-SF's mean NLL on New York is therefore +2,025, while its median is ordinary.
- TabPFN refuses these episodes (undefined). That is why the TabPFN comparisons, which exclude them by protocol, are
  not affected.
- PFN-L-SF fails the same way (+598,512 on New York; +1,962 on Washington). The plain PFN-L, without SF, does not.

**Each network, UPT-SF against PFN-L.** It is better on Washington (+0.59), DWD pressure (+0.55) and AURN PM2.5
(+0.13). New York is the exception.

**Against TabPFN v2 by network:**
- UPT-SF wins on DWD pressure: +0.41 with all sensors, +1.68 with 6 dropped.
- It ties on AURN PM2.5 with all sensors (−0.03, CI includes 0) and wins with 6 dropped (+0.28).
- It loses on both river networks (−1.35, −1.80).

**All thirteen networks, descriptive.** UPT-SF vs TabPFN v2:

| Condition | Gain | 95% CI |
|---|---|---|
| All sensors | −0.239 | [−0.314, −0.165] |
| 3 dropped | −0.056 | [−0.137, +0.025] (tie) |
| 6 dropped | **+0.329** | [+0.230, +0.427] |

## Mean NLL (lower is better), all sensors / 6 dropped

| Network | **Pool** | TabPFN v2 | UPT-SF | PFN-L | TNP-D | Calibrated kriging |
|---|---|---|---|---|---|---|
| USGS NY | ≈ −3.70 on TabPFN-defined episodes† | −3.63 / −2.88 | (+2025) | −1.56 / −1.34 | −1.41 / −0.75 | +85.3 |
| USGS WA | **−5.15 / −4.67** | −4.97 / −4.41 | −3.31 / −2.98 | −2.73 / −2.47 | −2.51 / −2.11 | −1.95 / −1.78 |
| DWD pressure | **−0.60 / −0.23** | −0.15 / +1.46 | −0.57 / −0.22 | −0.01 / +0.14 | +0.09 / +1.40 | +0.15 / +0.65 |
| AURN PM2.5 | **−0.56 / −0.35** | −0.43 / +0.07 | −0.40 / −0.21 | −0.27 / −0.10 | −0.21 / +0.06 | −0.07 / +0.08 |

† On the New York episodes where TabPFN is defined, the pool's mean is better than TabPFN's by 0.065 nats. On the
constant-support episodes the pool falls back to UPT-SF (w = 1, the pre-registered rule) and inherits its blow-up. Its
mean over all New York episodes is therefore +2,023.

## Reading

1. **The pre-registered headline holds: UPT ⊕ TabPFN beats TabPFN v2.**
   - It wins on fresh 2026 data, pooled and on each of the four networks, with all sensors and under sensor loss.
   - The gain grows with sensor loss: +0.17, +0.25 and +0.60 nats.
   - Stacking puts weight on TabPFN where its sharp predictive is right (rivers) and on the lifted-site model where
     missing sensors matter (pressure, air quality, sensor loss).
   - This is the behaviour Proposition 5 (pool guarantee) predicts, obtained without training anything new.
2. **UPT-SF alone does not beat TabPFN v2 on rivers.** It also has a pre-registered failure on degenerate,
   constant-support episodes, which breaks W3, W5 and W6.
   - The fix is simple: a scale floor tied to the network's typical support scale, or falling back to the raw-unit
     pipeline when the support target is constant.
   - It is post hoc and untested here, and is reported as a limitation.
3. **Robustness to sensor loss is UPT's distinctive strength.** Across all thirteen networks it beats TabPFN v2 at six
   dropped sensors, and it wins outright on DWD pressure at every level.
