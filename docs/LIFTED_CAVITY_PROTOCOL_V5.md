# Lifted cavity: protocol v5: the transformer as one more site

Committed October 11, 2026. At commit time:
- no pooled score has been computed on any panel;
- no CPU-scale model (`pfn_s*`, `lct_s*`) has been fine-tuned on, or scored on, Beijing PM10, SO2 or O3;
- no protocol-v4 score is in this checkout, and the author of this protocol has seen none.

## Why v5

Protocols v2 and v3 locate the real-data gap in the Gaussian site family: the fine-tuned transformer beats the lifted
models on real networks (E4b fails; E7b is not established). The lifted posterior is a product of factors. v5 adds
the transformer's own predictive as one more factor on the target. Every factor is tempered, so the exponents sum to
one, as in the generalized product of experts (Cao & Fleet 2014) and tempered power-EP sites (Minka 2004):

    p(y) ∝ p_LCT(y)^(1/2) · p_PFN(y)^(1/2)

- **No new parameters and no tuning.** β = 1/2 is fixed here.
- **Closed form.** For Gaussian experts the pool is Gaussian (precision-weighted).
- **Proposition 5 (Hölder).** For every query and every outcome y:

      NLL_pool(y) = ½ NLL_LCT(y) + ½ NLL_PFN(y) + log Z,   with Z = ∫ p_LCT^½ p_PFN^½ ≤ 1.

  The pool therefore never scores worse than the experts' average. It gains −log Z, a Bhattacharyya distance, when the
  experts disagree.
- **Implementation:** `experiments/lifted_cavity/site_pool.py`, which also checks the bound per episode.

## Experts (fixed)

- **Source checkpoints:** `lct_s1` and `pfn_s1`, the protocol-v3 checkpoints, unchanged.
- **Fine-tuning:** each checkpoint is fine-tuned separately per pollutant, with the protocol-v2 recipe as in
  `finetune_real.py`:
  - 4,000 source episodes;
  - 2,000 AdamW steps, learning rate 3e-4 with cosine decay;
  - 64 episodes × 8 queries per step;
  - seed 11;
  - CPU.
- **Final checkpoint only;** no selection.

## Targets (fixed)

- **Data:** Beijing PM10, SO2 and O3. Log concentration at each station is predicted from the other 11 stations.
  - Dequantization follows v4 amendment 1.
  - Test panels: `eval_real.py build --dataset beijing_<p> --seed 4041`, with the same episode construction and split
    as v4.
- **Evaluation setting:** natural missingness (`e0`).
- **Pooling across pollutants ("Beijing-new"):** per-episode NLLs are concatenated over the three pollutants.
- **Intervals:** paired 95% intervals, cluster-robust over weeks. Each week pools all its pollutants and stations, as
  in `confirm4.py`.

## Endpoints

Gains are in nats of mean NLL, positive when the first-named model is better. Computed by `confirm5.py`.

| ID | Comparison | Pass condition |
|---|---|---|
| E13 | pool(LCT, PFN) vs fine-tuned PFN | non-inferiority: lower bound > −0.02 |
| E14 | pool(LCT, PFN) vs fine-tuned PFN | gain ≥ .01 and lower bound > 0 |
| E15 | pool(LCT, PFN) vs fine-tuned LCT | gain ≥ .01 and lower bound > 0 |
| E16 | pool(LCT, PFN) vs pool(PFN seed 1, PFN seed 2) | gain ≥ .01 and lower bound > 0 |

**E16** is the ensembling control. It needs `pfn_s2` fine-tuned on the same three pollutants with the same recipe.
- It runs after E13–E15 if compute allows.
- If it does not run, it is reported as not run, never as passed.
- E16 separates the lifted sites' complementary information from plain seed ensembling.

**Secondary, descriptive:**
- the +3 and +6 extra-dropped-sensor conditions;
- each pollutant separately;
- the linear pool (mixture with weight ½);
- the Hölder slack −log Z.

**Post hoc, labeled as such:** the same pools on the used targets (Beijing NO2, PM2.5, CO; seed-2 pairs where they
exist). They motivate the method and do not confirm it.

All endpoint outcomes, including failures, are reported.
