# Protocol v6 results: the universal pool transformer on three new sensor networks (October 11, 2026)

**Source of numbers:** `artifacts/reports/lifted_cavity_paper/runs/confirm_v6.json`, computed by `confirm6.py` on
Colab (NVIDIA L4).

**Protocol history:**
- protocol: f8eb0c0, committed before any v6 panel or score;
- amendment 1 (log traffic speeds): d65e838, before any panel;
- amendment 2 (NL-FA undefined on singular episodes): 2be998a;
- amendment 3 (TabPFN undefined on constant-support episodes): a712268, before any TabPFN score was read.

**Data:** METR-LA, PEMS-BAY and Intel Lab; extra = 0; 3 seeds per learned model; 116 network-day clusters.

## Endpoints

| ID | Comparison | Gain (nats) | 95% CI | Result |
|---|---|---|---|---|
| P1 | UPT vs fine-tuned PFN | **+0.170** | [+0.136, +0.205] | **pass** |
| P2 | UPT vs two-seed PFN linear pool (non-inferiority, −.01) | **+0.126** | [+0.091, +0.161] | **pass** (also superior) |
| P3 | UPT vs TabPFN v2 in context (3 undefined episodes excluded) | −0.255 | [−0.315, −0.195] | **fail** |
| P4 | UPT vs PFN, synthetic, 16 sensors after training on 5 | **+0.156** | [+0.136, +0.175] | **pass** |

The headline claim needs P1, P2 and P4 to pass. **It holds.** UPT is one network of 203k parameters, about 0.1% more
than the LCT. It beats:
- the transformer trained and fine-tuned identically;
- the two-seed transformer mixture, which costs twice the inference;
- the transformer at a width unseen in training.

P3 fails: the pretrained TabPFN v2 is better with every sensor available.

## Mean test NLL by network (seed averages; lower is better)

| Network | Extra dropped | PFN | LCT | **UPT** | TabPFN v2 | LightGBM | MICE+BLR | BLR (closed form) |
|---|---|---|---|---|---|---|---|---|
| METR-LA | 0 | −1.455 | −1.104 | −1.709 | **−1.793** | −0.395 | −0.811 | −0.804 |
| METR-LA | 3 | −1.343 | −1.032 | **−1.623** | −1.552 | +0.216 | −0.515 | −0.750 |
| METR-LA | 6 | −1.063 | −0.775 | **−1.429** | −1.211 | +0.926 | +0.351 | −0.607 |
| PEMS-BAY | 0 | −2.357 | −2.240 | −2.490 | **−2.682** | −1.176 | −1.909 | −1.665 |
| PEMS-BAY | 3 | −2.223 | −2.103 | −2.365 | **−2.457** | −0.886 | −1.551 | −0.979 |
| PEMS-BAY | 6 | −1.896 | −1.827 | **−2.131** | −2.061 | −0.405 | −0.493 | +1.071 |
| Intel | 0 | −0.215 | +0.410 | −0.303 | **−1.305** | +0.543 | −0.162 | −0.112 |
| Intel | 3 | +0.052 | +0.338 | **−0.080** | +0.821 | +5.343 | +2.681 | +0.023 |
| Intel | 6 | +0.626 | +0.726 | **+0.352** | +2.363 | +11.79 | +27.87 | +0.422 |

**LightGBM.** These values come from Colab; the detached local recomputation is a cross-check only.

**`lift1` is unstable on Intel.** The small lifted network diverges there: mean NLL 3424 at extra = 0. This is the
float32 site-precision problem described below.

## Reading

1. **One UPT matches more than a two-model ensemble.** v5 showed that the training-free pool's gain over PFN is
   ensembling (E16L failed). v6 shows that UPT, as a single network, beats that ensemble by 0.126 nats on three
   networks it never saw. The gated free head plus lifted sites is therefore more than pooling two models.
2. **Extrapolation is preserved.** P4 holds at 16 sensors after training on 5.
3. **TabPFN v2 vs UPT under missingness.**
   - With all sensors present, TabPFN v2 is better. It is a much larger model pretrained on a broad prior.
   - With sensors removed, the ordering flips on most panels. UPT beats TabPFN at +6 on all three networks, and at +3
     on METR-LA and Intel. TabPFN degrades sharply on Intel (−1.31 → +2.36).
   - This is **descriptive**: it was not a v6 endpoint. It is the natural primary endpoint for the next frozen
     protocol.
4. **Numerical stability is a real limitation.**
   - The lifted posterior inverts site precisions in float32. On heavy-tailed data:
     - LCT predictions differ across devices (CPU vs GPU, up to 0.36 nats per episode on Intel), although they replay
       exactly on the same device;
     - `lift1` diverges.
   - UPT is less exposed, presumably because the gate can move mass to the free head, but its mixture NLL needed a
     log-sum-exp evaluation. The fix is commit 62a3051, applied before the reported scores.
   - **Fix to test next:** float64 or damped precision inversion.

## Changes made during execution (all before the affected scores were read)

- **Amendments 1–3**, as listed above.
- **Prediction replay** runs on the scoring device, with tolerance 1e-4 absolute plus 1e-4 relative (commit e6b64ff).
- **UPT mixture metrics.** Computed by log-sum-exp. The first UPT cells had infinite NLL when both component
  densities underflowed; all 15 UPT score files were deleted and recomputed with the fixed code. The extra = 0 cells
  used by P1–P4 had been finite, and are unchanged up to rounding.

## Next (OCT20_PLAN)

- **Protocol v7.**
  - Primary endpoint: UPT vs TabPFN v2 under 3 and 6 dropped sensors, on fresh networks.
  - Plus a float64 site layer.
  - Plus the v4-scale UPT (2.1M parameters).
- **Integration.** Bring the v4 GPU results (PFN-L, LCT-L) in when they are collected.
- **Paper framing.** Write around v5 and v6: the pooling theory, the confirmed v5 mixture win and its ensembling
  control, UPT's P1/P2/P4, the TabPFN comparison as a limitation, and the dropout advantage.
