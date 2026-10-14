# Lifted cavity: protocol v17: more fresh tasks after TabPFN-3.5's release, and a combined test

Committed October 14, 2026, before any v17 data were downloaded or any v17 panel was built or scored. Known at
commit time: the v16 results (`LIFTED_CAVITY_V16_RESULTS.md`): F1 passed; F2 missed with positive gains (+0.023,
+0.022) whose 95% intervals reached zero with 30 clusters.

## Why

v16's dropped-sensor tests were underpowered: the gains were the size v15 predicted, but post-release data on four
networks give only 30 network-window clusters. The three v9 networks have trained UPT-SF models and data after
TabPFN-3.5's release, so they add fresh tasks. Because we have already looked at v16, a combined test is a second
look at the same question; it pays for that with a stricter interval (Bonferroni split of the 5% level, 97.5%
intervals), and v17's networks are also reported on their own.

## Data (code: `realdata_v17.py`, `fetch_v17.py`)

- **Networks:** v9's `dwd_t2m`, `aurn_no2`, `usgs_q`, with the same sensors, transforms and neighbour selection (the
  river's correlation neighbours from the same source period of the same snapshot).
- **Days:** after TabPFN-3.5's release only. River: 2026-09-10..10-13 (3-day windows); DWD temperature and AURN NO₂:
  2026-09-10..10-11 (weekly windows). The river snapshot is extended to 2026-10-12 with the v16 fetcher's method; the
  other two already run to 2026-10-10.
- **Targets:** 48 per network, drawn with seed 17171 + 7 from the sensors that were never a v9 test target.
- **Episodes:** as v16 (48 + 48 rows, 20% support dropout; conditions with seeds 17171 + 100 + e). Panel seed 17171.

## Models (frozen; nothing is trained)

UPT-SF: v9's `upt_L_sfm` fine-tunes (seeds 1–3, tta 8). TabPFN-3.5: as v13–v16. The pool: v15's rule P3
(`pool_eval15.py`), and, secondary, the v13 rule.

## Endpoints (`confirm17.py`)

**Combined (primary):** every fresh, post-release task of v16 (four v10 networks) and v17 (three v9 networks), pooled;
TabPFN-3.5-undefined episodes excluded; paired intervals cluster-robust over network-windows, **at the 97.5% level**.

| ID | Comparison | Condition | Pass |
|---|---|---|---|
| G1 | P3 pool vs TabPFN-3.5, NLL | all sensors | gain ≥ .01 and 97.5% lower bound > 0 |
| G2 | P3 pool vs TabPFN-3.5, NLL | 3 and 6 dropped (two tests) | gain ≥ .01 and 97.5% lower bound > 0 |

**Secondary:** v17's three networks alone (95% intervals), each of them, the v13 rule, UPT-SF alone, squared error.

All outcomes are reported, in the paper as well.

## Execution

| Step | Where |
|---|---|
| Data (`fetch_v17.py`), panels (`eval_real.py build --dataset <net>_f --seed 17171`), digests | local |
| TabPFN-3.5 test predictions, 2-fold and 4-fold cross-fits (`modal_tabpfn35_v16.py`, seed 17171) | Modal L4 |
| UPT-SF per-query outputs and 4-fold cross-fit; pools; `confirm17.py` (`run_v17.py upt`, then `final`) | Colab L4 |
