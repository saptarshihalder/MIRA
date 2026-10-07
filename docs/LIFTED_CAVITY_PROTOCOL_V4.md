# Lifted cavity: protocol v4 — scale, a pretrained tabular foundation model, new real targets

Committed October 7, 2026, before generating any panel or target listed here and before training any model listed
here.

## Why v4

Protocols v2 and v3 leave three objections that a reviewer will raise first:
1. Both transformers are CPU-scale (203k parameters).
2. There is no pretrained tabular foundation model.
3. The real-data evidence for the lifted cavity transformer (LCT) rests on one pollutant and varies with the seed.

v4 tests the paper's claims against all three. Every learned v4 model is trained with three seeds, and endpoints use
seed-averaged per-task scores.

## Models (fixed)

**PFN-L (`pfn_L_s{1,2,3}`).** The cell-token transformer of `pfn.py`.
- Architecture: d = 128, 8 layers, 8 heads, feed-forward width 512 (2,119,810 parameters).
- Recipe: fresh tasks every step, 32 tasks × 16 queries, masks with at most one missing sensor, 40,000 AdamW steps.
- Optimizer: learning rate 5e-4, 2,000 warm-up steps, cosine decay to 10%, weight decay 1e-4, gradient clipping at 1.
- Seeds 1, 2 and 3; final checkpoint, no selection.

**LCT-L (`lct_L_s{1,2,3}`).** The PFN-L backbone with the lifted-site head of `lct.py` (2,120,326 parameters). Same
recipe and seeds.

**TabPFN v2 (`tabpfn_v2`).** The pretrained TabPFN v2 regressor (`tabpfn` package, `ModelVersion.V2`).
- 8 estimators, random state 0.
- In context only (no fine-tuning). Missing entries are NaN.
- Every mask of a task is predicted in one call (query rows do not attend to each other).
- NLL is its own predictive density in raw target units. `experiments/lifted_cavity/tabpfn_colab.py` checks this
  against its predictive mean.

**Fine-tuning.** The protocol v2 recipe, unchanged:
- 2,000 steps, learning rate 3e-4, 64 episodes × 8 queries, seed 11.
- Applied to PFN-L, LCT-L and the lifted cavity network `lift1_s{1,2,3}` on each new target.

**Hardware and reproducibility.**
- PFN-L and LCT-L are trained, fine-tuned and scored on a GPU (Google Colab), using
  `experiments/lifted_cavity/colab_v4.py`.
- GPU kernels are not bitwise reproducible; checkpoints let interrupted runs resume.

## Panels and targets (new)

**Synthetic panels**, generator as in protocol v2:

| ID | Seed | Tasks | Sensors | Nonlinearity | Support rows |
|---|---|---|---|---|---|
| H1 | 20261401 | 256 | 5 | .4 | 48 |
| H3 | 20261403 | 128 | 16 | .4 | 48 |

**Real targets.**
- Beijing PM10, SO2 and O3: log concentration at each station from the other 11 stations, natural missingness.
- Same episode construction as protocol v2 (`eval_real.py build --seed 4041`): test 2016-01 to 2017-02, fine-tuning
  episodes from 2013-03 to 2015-12.
- No model has been scored on these pollutants. The endpoints pool the three pollutants ("Beijing-new"). Each model
  is fine-tuned separately per pollutant.

## Endpoints

Gains are in nats of mean NLL, positive when the first-named model is better. Intervals are paired 95% intervals over
tasks; on Beijing they are cluster-robust over weeks, with each week pooling all its pollutants and stations.

| ID | Data | Comparison | Pass condition |
|---|---|---|---|
| E8 | H1, k = 2 | LCT-L vs PFN-L | gain ≥ .01 and lower bound > 0 |
| E9 | H3 (16 sensors), k = 2 | LCT-L vs PFN-L | gain ≥ .01 and lower bound > 0 |
| E10 | Beijing-new, natural missingness | fine-tuned LCT-L vs fine-tuned PFN-L | non-inferiority: lower bound > −0.02 |
| E11 | Beijing-new, six more stations removed | fine-tuned LCT-L vs fine-tuned PFN-L | gain ≥ .01 and lower bound > 0 |
| E12 | Beijing-new, natural missingness | fine-tuned LCT-L vs TabPFN v2 (in context) | gain ≥ .01 and lower bound > 0 |

Computed by `experiments/lifted_cavity/confirm4.py`.

**Secondary, descriptive:**
- every other k and condition;
- each pollutant separately;
- lift1 on the new targets;
- TabPFN v2 on H1, H3 and every earlier panel and target (all used data);
- PFN-L and LCT-L on the earlier panels;
- compute.

## Execution notes (recorded as they happen)
