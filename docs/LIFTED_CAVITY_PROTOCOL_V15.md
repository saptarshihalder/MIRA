# Lifted cavity: protocol v15: learning-curve-aware stacking for UPT-SF ⊕ TabPFN-3.5

Committed October 13, 2026, before any 4-fold cross-fit output exists and before any v15 pool has been computed.
Known at commit time: the v13 results (the v13 pool passes Z1–Z3 but loses to TabPFN-3.5 on both river networks by
0.015–0.03 nats) and the v14 results (`LIFTED_CAVITY_V13_RESULTS.md`, `LIFTED_CAVITY_V14_RESULTS.md`). The v10
networks are therefore not fresh, and this protocol says so.

## Why

The v13 pool chooses its weight w per episode by a two-fold cross-fit: each model is fitted on 24 of the 48 labelled
rows and scored on the other 24. Cross-validation estimates the performance of a model trained on (K − 1)/K of the
data, so with K = 2 it favours the model whose performance changes least with more data. TabPFN-3.5 sharpens a lot
from 24 to 48 rows; UPT-SF, with its pretrained and fine-tuned prior, much less. The cross-fit therefore over-weights
UPT-SF where TabPFN-3.5 is better (rivers: weight 5–10% on UPT-SF, loss 0.015–0.03 nats). The grid floor (w ≥ 0.02)
also prevents the pool from nearly switching UPT-SF off.

## Variants (code: `pool_eval15.py`; inputs: `tabpfn_cfk.py`, `eval_real.py --cf-k 4`)

The pool is always p = w p_UPT + (1 − w) p_TabPFN-3.5 per query, with w chosen per episode and condition by maximizing
the summed log score of the pool on the cross-fitted support rows. No test label is used.

| Variant | Cross-fit | Grid | UPT-SF partner |
|---|---|---|---|
| v13ref (reference; must reproduce the v13 pool exactly) | 2-fold (24 rows) | G0 = {.02, …, .98} | per seed, pools averaged |
| P0 (descriptive) | 2-fold | G1 | seed ensemble |
| P1 | 4-fold (36 rows) | G0 | per seed |
| P2 | 4-fold | G1 | per seed |
| P3 | 4-fold | G1 | seed ensemble |
| P4 | learning-curve extrapolation to 48 rows | G1 | seed ensemble |

- **4-fold cross-fit:** contiguous folds of 12 rows (`realdata.kfolds`); each fold is predicted by both models fitted on
  the other 36 rows, under the shared cross-fit masks of v11 (`realdata.crossfit_masks`, seed 20000 + extra). UPT-SF's
  anchors are refitted on each training part. TabPFN-3.5 exactly as in v13 (8 estimators, random state 0); folds with
  a constant training target are left out, as in v11.
- **G1** = G0 ∪ {.001, .003, .01} ∪ {.99, .997, .999}.
- **Seed ensemble:** the partner is the equal mixture of UPT-SF seeds 1–3 (its log density is the log-mean-exp of the
  seeds' log densities), instead of one pool per seed averaged afterwards.
- **P4 extrapolation:** per support row and model (the seed ensemble for UPT-SF), the log density at 48 training rows
  is extrapolated linearly in 1/n from the 2-fold (24 rows) and 4-fold (36 rows) cross-fits:
  l48 = l36 + 0.5 (l36 − l24), the correction clipped to ±1 nat; where l24 is missing, l36 is used.
- Where TabPFN-3.5 is undefined (constant support target), the pool is UPT-SF alone, as in v11.

## Development (the v9 networks `dwd_t2m`, `aurn_no2`, `usgs_q`; already used)

Every second episode, the episodes scored by v14's development (TabPFN-3.5's test outputs: `tabpfn_v3_5_s2`). New
inputs: TabPFN-3.5's 4-fold cross-fit on those episodes; UPT-SF's 4-fold cross-fit on all episodes.

**Selection, fixed here.** Among P1–P4, the selected variant has the lowest mean development NLL over the three
networks and three conditions (scored, TabPFN-defined episodes). Go to confirmation only if it beats TabPFN-3.5 alone
at each of the three conditions, pooled over the development networks.

## Confirmation (the v10 networks `usgs_ny`, `usgs_wa`, `dwd_p`, `aurn_pm25`; every episode)

New inputs, computed in parallel with development: TabPFN-3.5's 4-fold cross-fit (Modal, `modal_tabpfn35_cfk.py`) and
UPT-SF's 4-fold cross-fit (Colab). These score support rows only. No v10 pool or test score of a v15 variant is
computed before the selection is written.

**Statistics:** as v9–v14 (paired 95% intervals cluster-robust over network-windows; gain = TabPFN-3.5 − pool;
TabPFN-undefined episodes excluded).

| ID | Comparison | Condition | Pass |
|---|---|---|---|
| V1 | selected pool vs TabPFN-3.5, NLL, pooled | all sensors | gain ≥ .01 and lower bound > 0 |
| V2 | the same | 3 and 6 dropped (two tests) | gain ≥ .01 and lower bound > 0 |
| V3 | selected pool vs TabPFN-3.5, NLL, **each network** | all, 3 and 6 dropped (twelve tests) | non-inferiority: lower bound > −0.005 |

V3 tests the claim "never worse than TabPFN-3.5 on any network, at any level of sensor loss", which v13 failed on
the rivers. It passes only if all twelve tests pass.

**Secondary:** selected pool vs the v13 pool (pooled, each condition); per-network superiority vs TabPFN-3.5 (gain
≥ .01, lower bound > 0); squared error vs TabPFN-3.5; the mean weight on UPT-SF per network; v13ref vs the v13 pool
(must be identical).

All outcomes are reported, in the paper as well.

## Execution

| Step | Where |
|---|---|
| TabPFN-3.5 4-fold cross-fit on the v10 panels (support rows only) | Modal, one L4 container, 8 processes |
| UPT-SF 4-fold cross-fit (v9, v10); TabPFN-3.5 4-fold cross-fit on the v9 development episodes | Colab (`run_v15.py inputs`) |
| Development pools and the selection (`run_v15.py dev` → `selection15.json`) | Colab |
| Confirmation: the selected pool and v13ref on v10, then `confirm15.py` (`run_v15.py confirm`) | Colab |
