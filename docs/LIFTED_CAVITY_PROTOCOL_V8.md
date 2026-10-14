# Lifted cavity: protocol v8: neural-process and kriging baselines on the six real networks

Committed October 11, 2026, before any v8 baseline was trained or scored on a test panel.

The only runs of these baselines so far are code-path smoke tests on the v6 Intel test panel:
- 4 training steps of CNP and TNP on synthetic tasks;
- IGNNK trained for 50 and 2,000 steps, with plain and calibrated variants.

These showed that plain IGNNK is badly overconfident under the Intel drift. That is why calibrated IGNNK, the
**stronger** variant, is the comparator. No CNP or TNP score on any real panel exists.

UPT scores are already known on the v6 networks; v7 is in progress. These endpoints are confirmatory for the
baselines, whose recipes are fixed here.

## Baselines

**Neural processes** (`np_models.py`). Same prior, masks and source recipe as `pfn_s*` (`pfn_train.py --model`):
30,000 steps, 16 tasks × 16 queries, lr 1e-3, warm-up 1,000, seeds 1–3, final checkpoint.

| Model | Description | Parameters |
|---|---|---|
| CNP | Garnelo et al. (2018). MLP encoder and decoder, width 192, mean pooling. | 161k |
| TNP-D | Nguyen & Grover (2022). Row tokens, d = 64, 4 layers, 4 heads, feed-forward width 128; queries attend only to support rows. | 145k |

Rows are padded to 16 sensors. Each model is fine-tuned per network with the protocol-v2 recipe (`finetune_real.py`
defaults), exactly like PFN and UPT.

**Kriging** (`kriging.py`).
- **IGNNK-style inductive kriging** (Wu et al. 2021):
  - Graph: the target's local 12-sensor graph, with absolute source-period correlations as diffusion weights.
  - Model: 3 diffusion graph-convolution layers, 2 hops, hidden width 64, Gaussian head.
  - Training: 20,000 steps on the network's source period. Seeds 1–3.
- **Calibrated IGNNK (`ignnk_cal`)**: the same model plus a least-squares affine recalibration on each episode's 48
  support rows. It is the comparator.
- **Classical kriging** is already in every panel: simple kriging, i.e. the conditional Gaussian (best linear
  unbiased predictor) under the EM support covariance (`em_gauss`), and FA-Gaussian.

## Networks

All six real networks from v6 and v7, with their frozen test panels:
- METR-LA, PEMS-BAY and Intel (seed 6061);
- PeMS03, PeMS08 and China air (seed 7071).

## Endpoints (`confirm8.py`)

**Common settings:**
- The extra = 0 condition.
- Per-episode NLL, pooled over the six networks.
- Learned models use the seed average over seeds 1–3.
- Paired 95% intervals are cluster-robust over network-windows.

Gains are positive when UPT is better.

| ID | Comparison | Pass condition |
|---|---|---|
| R1 | UPT vs TNP-D | gain ≥ .01 and lower bound > 0 |
| R2 | UPT vs CNP | gain ≥ .01 and lower bound > 0 |
| R3 | UPT vs calibrated IGNNK kriging | gain ≥ .01 and lower bound > 0 |

**Secondary, with the same statistics:**
- R1–R3 at 3 and 6 extra dropped sensors;
- each network separately.

**Descriptive:**
- plain IGNNK;
- CNP and TNP on the 16-sensor synthetic panel (v6 P4 panel);
- MAE, RMSE (from `se`), CRPS and 90% coverage for every method.

All outcomes are reported.

## Implementation note (October 11, before any kriging result was read)

The kriging training loop now draws its extra dropped inputs in vectorized form: a uniformly random order of the
observed inputs, with the first k removed and at least one input always kept. This is the same distribution as
before, computed about 4× faster. The recipe is unchanged.

All kriging runs are recomputed with this code. The few runs finished earlier, whose results were never read, are
discarded.
