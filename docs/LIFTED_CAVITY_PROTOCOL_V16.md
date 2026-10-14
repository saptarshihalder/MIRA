# Lifted cavity: protocol v16: a fresh test after TabPFN-3.5's release, with new targets

Committed October 14, 2026, before any v16 panel was built and before any model was scored on the data below. The
river data for 2026-10-01..10-12 were downloaded just before (`fetch_v16.py`; content digests in
`data_snapshots/v16/*.json`); nothing else was computed on them except the checks listed under Data.

## Why

Protocols v13–v15 compared UPT-SF ⊕ TabPFN-3.5 with TabPFN-3.5 on the four v10 networks, which earlier protocols had
already used, and v15's rule was designed after seeing v13's results there. The v10 test periods (to 2026-09-30) also
largely precede TabPFN-3.5's release on 2026-09-09, so its training data may overlap with them. v16 tests the frozen
v15 rule once on prediction tasks that no protocol has scored, on days after TabPFN-3.5's release.

## Data (code: `realdata_v16.py`, `fetch_v16.py`)

- **Networks and sensors:** v10's four networks, with the same sensors, transforms and neighbour selection (the
  correlation neighbours of the rivers are computed on the same source period of the same snapshot). Checked before
  this commit: neighbour sets and sensor ids identical to v10.
- **Days:** only after TabPFN-3.5's release. Rivers: 2026-09-10..10-13, 11 three-day windows. DWD pressure and AURN
  PM2.5: 2026-09-10..10-11, 4 weekly windows.
- **Targets:** 48 per network, drawn with seed 16161 + 7 from the sensors that were never a v10 test target (checked:
  no overlap). Every episode is a new prediction task.
- **Episodes:** as v10 (48 support + 48 query rows, 20% support dropout, natural query masks); 528, 527, 190 and 191
  episodes (1,436), 30 network-window clusters. Conditions: all sensors, 3 and 6 extra dropped (seeds 16161 + 100 + e).
  Panel seed 16161.

## Models (frozen; nothing is trained)

- **UPT-SF:** v10's `upt_L_sfm` fine-tunes, seeds 1–3, 8-member test-time ensemble.
- **TabPFN-3.5:** tabpfn 9.1.0, `ModelVersion.V3_5`, 8 estimators, random state 0 (as v13).
- **The pool:** v15's selected rule P3 (`pool_eval15.py`): w per episode and condition from a 4-fold cross-fit of the
  support, grid G1, UPT-SF seed ensemble. Also, secondary, v13's rule (`v13ref`: 2-fold, G0, per seed).

## Endpoints (`confirm16.py`; four networks pooled; TabPFN-3.5-undefined episodes excluded)

**Statistics:** as v9–v15 (paired 95% intervals cluster-robust over network-windows; gain = baseline − pool).

| ID | Comparison | Condition | Pass |
|---|---|---|---|
| F1 | P3 pool vs TabPFN-3.5, NLL | all sensors | gain ≥ .01 and lower bound > 0 |
| F2 | P3 pool vs TabPFN-3.5, NLL | 3 and 6 dropped (two tests) | gain ≥ .01 and lower bound > 0 |

**Secondary:**
- P3 pool vs UPT-SF alone; P3 pool vs the v13 rule;
- per network vs TabPFN-3.5, superiority and non-inferiority (lower bound > −0.005), as v15's V3;
- squared error vs TabPFN-3.5;
- mean weight on UPT-SF.

With only 30 clusters, the intervals are about twice as wide as v15's. All outcomes are reported, in the paper as well.

## Execution

| Step | Where |
|---|---|
| Panels (`eval_real.py build --dataset <net>_f --seed 16161`), built once, with content digests | local |
| TabPFN-3.5: test predictions and 2-fold outputs (`tabpfn_colab.py --crps`), 4-fold cross-fit (`tabpfn_cfk.py`) | Modal L4 |
| UPT-SF: per-query outputs (`eval_real.py --per-query --tta 8`) and 4-fold cross-fit (`--cf-k 4`) | Colab L4 |
| Pools (`pool_eval15.py --upt-net`), then `confirm16.py` (`run_v16.py`) | Colab |
