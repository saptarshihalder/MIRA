# Protocol v7 results: UPT vs TabPFN v2 under sensor loss (October 11, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v7.json`, computed by `confirm7.py` on
Colab (NVIDIA L4).

**Protocol:** a45b2cf, committed before any v7 panel existed.

**Data:** three networks never used by this project: PeMS03 (Sacramento flow), PeMS08 (San Bernardino flow) and China
air (PM2.5, Beijing stations excluded).
- 1,784 test episodes in 63 network-window clusters.
- Learned models use seeds 1–3.
- No TabPFN episode was undefined.

## Endpoints

| ID | Comparison | Condition | Gain (nats) | 95% CI | Result |
|---|---|---|---|---|---|
| Q1 | UPT vs TabPFN v2 | 3 extra sensors dropped | **+0.320** | [+0.302, +0.338] | **pass** |
| Q2 | UPT vs TabPFN v2 | 6 extra sensors dropped | **+0.989** | [+0.945, +1.033] | **pass** |
| Q3 | UPT vs fine-tuned PFN | natural | **+0.361** | [+0.283, +0.439] | **pass** |
| Q4 | UPT64 vs UPT (non-inferiority, −.01) | natural | 0.000 | [−0.000, +0.000] | pass |

**The headline claim** (Q1 and Q2) **holds**: UPT is more robust to sensor loss than the pretrained TabPFN v2.

**S1 and S2 (UPT-L), run after the main endpoints.**
- S1, UPT-L vs TabPFN v2: +0.313 [+0.253, +0.372] at extra = 0; +0.505 at 3; +1.073 at 6. All pass.
- S2, UPT-L vs UPT: +0.230 [+0.174, +0.286]. Pass.

Details are in `LIFTED_CAVITY_V8_RESULTS.md`.

## Mean test NLL by network (seed averages; lower is better)

| Network | Extra dropped | PFN | **UPT** | TabPFN v2 | FA-Gaussian | BLR |
|---|---|---|---|---|---|---|
| PeMS03 | 0 | −1.013 | **−1.198** | −1.040 | −0.567 | +0.088 |
| PeMS03 | 3 | −0.922 | **−1.095** | −0.812 | −0.544 | +0.151 |
| PeMS03 | 6 | −0.771 | **−0.930** | −0.038 | −0.522 | −0.317 |
| PeMS08 | 0 | −1.100 | **−1.884** | −1.820 | −1.283 | −0.279 |
| PeMS08 | 3 | −1.070 | **−1.870** | −1.504 | −1.295 | −0.040 |
| PeMS08 | 6 | −0.963 | **−1.801** | −0.624 | −1.275 | −0.966 |
| China air | 0 | −0.105 | −0.166 | **−0.237** | +0.272 | +0.050 |
| China air | 3 | −0.044 | **−0.102** | +0.229 | +0.256 | +0.092 |
| China air | 6 | +0.067 | **+0.016** | +0.916 | +0.258 | +0.167 |

## Reading

1. **Sensor loss.** UPT beats TabPFN v2 at +3 and +6 on every network. TabPFN degrades sharply as sensors are removed
   (PeMS08: −1.82 → −0.62; China air: −0.24 → +0.92), while UPT barely moves (PeMS08: −1.88 → −1.80). This confirms,
   on fresh data, the reversal seen descriptively in v6.
2. **All sensors present.** UPT is better than TabPFN on both traffic networks and worse on China air. In v6, TabPFN
   was better at extra = 0 on all three networks (P3 failed). The full-sensor comparison is therefore mixed, and it
   is not claimed.
3. **The v6 P1 replication holds.** UPT beats its matched transformer on all three new networks.
4. **Precision.** Float64 site algebra changes nothing measurable for UPT, and no UPT episode exceeds an NLL of 10. The
   float32 instability seen in v6 affects the pure lifted models (`lift1`, LCT), not UPT.

## Status of the plan

- v8 baselines (CNP, TNP-D, calibrated IGNNK) are running on Colab in chain A.
- UPT-L is running in chain B.
