# Lifted cavity transformer: protocol v3 (frozen before any v3 panel or target exists)

Committed October 7, 2026, before generating or scoring any panel listed here. Protocol v2 found two things: the
lifted cavity network beats a TabPFN-v2-style transformer on every synthetic endpoint, but the transformer wins after
fine-tuning on real data (endpoint E4b failed). v3 tests one hypothesis motivated by that split: using lifted sites
as the transformer's **output layer** keeps the transformer's real-data flexibility and gains the lifted model's
extrapolation. This hypothesis was formed after seeing v2's real-data results, so v3 uses only new synthetic seeds
and real targets never scored before.

## Models (fixed)

**`lct_s1`, the lifted cavity transformer (`lct.py`, 202,822 parameters).**
- Its backbone is identical to `pfn_s1`.
- A zero-initialized linear head maps each observed query cell's token to a correction of that sensor's lifted site
  (offset, slope, nuisance loading and noise scale), linearized at the closed-form FA posterior mean.
- The prediction is the y-marginal of the product of observed sites. It equals FA-Gaussian at initialization, and
  missing sensors contribute no site.
- Training uses the same fresh-task recipe as `pfn_s1`: 30,000 steps of 16 tasks × 16 queries, masks with at most
  one missing sensor, lr 1e-3 with warm-up and cosine decay, and seed 1. The final checkpoint is used, with no
  selection.

**Comparators.**
- `lift1_s{1,2,3}`, `pfn_s1` and `pfnall_s1`, unchanged.
- The closed forms of protocol v2.

## Panels and targets (new)

| ID | Seed | Tasks | Sensors | Nonlinearity | Support rows |
|---|---|---|---|---|---|
| G1 | 20261301 | 256 | 5 | .4 | 48 |
| G3 | 20261303 | 128 | 16 | .4 | 48 |

**Real targets.** These are Beijing multi-site targets never scored by any model, using the same episode
construction as v2 (`eval_real.py build --seed 3031`):
- **Beijing NO2:** log NO2 at each station from the other 11 stations' NO2, with natural missingness. Test
  2016-01..2017-02; fine-tuning on 2013-03..2015-12.
- **Beijing CO:** the same construction for CO. Descriptive only.

**Fine-tuning.** The identical v2 recipe (`finetune_real.py`) is applied to `lct_s1`, `pfn_s1` and
`lift1_s{1,2,3}`.

## Endpoints

Gains are in nats of mean Gaussian NLL, with two missing query sensors on synthetic panels and natural missingness
on Beijing. Intervals are paired 95% intervals over tasks; on Beijing they are cluster-robust over weeks.

| Endpoint | Panel | Comparison | Pass condition |
|---|---|---|---|
| E5 | G1 | `lct_s1` vs `pfn_s1` | gain ≥ .01 and lower bound > 0 |
| E6 | G3, 16 sensors | `lct_s1` vs `pfn_s1` | gain ≥ .01 and lower bound > 0 |
| E7a | Beijing NO2 | fine-tuned `lct_s1` vs fine-tuned `lift1` (seed-averaged) | gain ≥ .01 and lower bound > 0 |
| E7b | Beijing NO2 | fine-tuned `lct_s1` vs fine-tuned `pfn_s1` | non-inferiority: lower bound of (PFN − LCT) > −0.02 |

Secondary, descriptive: every other k and model, Beijing CO, and `lct_s1` on the v2 panels (used data,
descriptive only). Computed by `experiments/lifted_cavity/confirm3.py`.

## Execution notes (recorded as they happened)

1. **October 7, 00:00 IST.** The compute container restarted while `lct_s1` was at step 10,000 of 30,000 and before
   any v3 score involving it existed. The interrupted run was set aside unevaluated
   (`runs_aborted/lct_s1_interrupted_step10000`), and `lct_s1` was retrained from scratch with the identical seed and
   recipe. The fine-tuning of `pfn_s1` on Beijing NO2, which had just started, was likewise restarted. Panels,
   targets and every completed score were unaffected.
2. **October 7, 00:23 IST.** A second container restart interrupted the retrained `lct_s1` at step 2,500 (set aside
   unevaluated as `runs_aborted/lct_s1_interrupted_step2500`). From then on, training and fine-tuning checkpoint the
   model, optimizer, scheduler and every random state every 250 (training) or 50 (fine-tuning) steps
   (`experiments/lifted_cavity/resume.py`). A test interrupts and resumes training and fine-tuning and checks that
   the final weights are bitwise identical to an uninterrupted run, so a resumed run is the protocol's run.
   `lct_s1` was then started from scratch a third time with the same seed and recipe.
3. **October 7, 04:02 IST.** A third restart interrupted the fine-tuning of `lct_s1` on Beijing NO2 at step 450 of
   2,000. It resumed from its checkpoint (bitwise-identical resume), so the scored model is the protocol's model.
