# Lifted cavity: protocol v13: does the pool also improve the newest foundation model, TabPFN-3.5?

Committed October 13, 2026, before TabPFN-3.5 has produced any output on real data. The only TabPFN-3.5 call before
this commit's runs is the smoke test of `modal_tabpfn35.py`, which uses synthetic inputs (random numbers shaped like a
panel episode). It checks the license, that the predictive is in raw target units, and the time per call.

The v10–v12 results on these networks are known (`LIFTED_CAVITY_V10_V11_RESULTS.md`, `LIFTED_CAVITY_V12_RESULTS.md`).
The data are therefore no longer fresh; this protocol tests whether the v11 result carries over to a newer foundation
model, and says so.

## Why

Protocols v11 and v12 pooled UPT-SF with TabPFN v2 (January 2025). Since then Prior Labs released TabPFN-2.5, 2.6, 3
and 3.5 (September 9, 2026; the package default). A pool that only helps an older model would be of limited use.

## Model (fixed here; code: `modal_tabpfn35.py`, `tabpfn_colab.py --version v3.5`)

- TabPFN-3.5: tabpfn 9.1.0 (the package of the v10 TabPFN v2 runs), `ModelVersion.V3_5`, the default checkpoint
  `tabpfn-v3.5-20260909.safetensors`, no fine-tuning, in context.
- Settings exactly as TabPFN v2 in v10/v11:
  - 8 estimators (the package default, `DEFAULT_N_ESTIMATORS = 8`), random state 0;
  - missing inputs as NaN;
  - `--allow-constant-support --crps`: constant-support episodes recorded as undefined; CRPS and 90% coverage from
    102 quantiles;
  - the per-query and cross-fit outputs (`pq_<tag>.npz`) in the same pass, with the same cross-fit masks
    (`realdata.crossfit_masks`, seed 20000 + extra).
- Panels: the four v10 panels on the Modal volume, identical by digest to the panels of record.
- Hardware: chosen for the remaining budget (about $8 on Modal, about 20 Colab units), not for the results: Modal L4
  GPUs, one network per container. Any network Modal cannot cover runs on Colab with the same code. The hardware
  changes predictions only at floating-point precision.

## Pools (identical rule: `pool_eval.py --tab tabpfn_v3_5`, per-episode two-fold cross-fit stacking, grid {.02, …, .98})

| Pool | Partner |
|---|---|
| **UPT-SF ⊕ TabPFN-3.5** | `upt_L_sfm` with `tta8`, seeds 1–3 (the v10-selected model, outputs of v11) |
| PFN-L ⊕ TabPFN-3.5 (secondary) | `pfn_L` with `tta8`, seeds 1–3 (outputs of v12) |

Where TabPFN-3.5 is undefined (constant training target), the pool is its partner alone (w = 1), as in v11.

## Endpoints (`confirm13.py`), on the four v10 networks (`usgs_ny`, `usgs_wa`, `dwd_p`, `aurn_pm25`), pooled

**Statistics:** exactly as v9–v12.
- Paired 95% intervals are cluster-robust over network-windows.
- Gain = baseline − pool.
- Pass: gain ≥ .01 nats and lower bound > 0.
- Episodes where TabPFN-3.5 is undefined are excluded from comparisons with TabPFN-3.5.

| ID | Comparison | Condition |
|---|---|---|
| Z1 | UPT-SF ⊕ TabPFN-3.5 vs TabPFN-3.5, NLL | all sensors |
| Z2 | UPT-SF ⊕ TabPFN-3.5 vs TabPFN-3.5, NLL | 3 and 6 extra dropped (two tests) |
| Z3 | UPT-SF ⊕ TabPFN-3.5 vs UPT-SF, NLL | all sensors |

**Secondary:**
- Z3 at 3 and 6 extra dropped;
- squared error vs TabPFN-3.5;
- each network separately;
- partner check (as v12 Y1): UPT-SF ⊕ TabPFN-3.5 vs PFN-L ⊕ TabPFN-3.5, NLL, at 0, 3 and 6 extra dropped, on the
  episodes where TabPFN-3.5 is defined.

**Descriptive:**
- TabPFN-3.5 vs TabPFN v2 alone (NLL, RMSE, CRPS, 90% coverage, undefined episodes);
- UPT-SF ⊕ TabPFN-3.5 vs UPT-SF ⊕ TabPFN v2;
- the mean stacking weight;
- TabPFN-3.5's time per episode.

**Reading, fixed here:**
- If Z1 and Z2 pass, the pool's gain is not specific to an older foundation model.
- If Z1 fails and Z2 passes, TabPFN-3.5 is as good as the pool with all sensors, and the pool helps when sensors fail.
- If Z2 fails, the pool's robustness gain does not carry over to TabPFN-3.5.

All outcomes are reported, in the paper as well.

**A caveat, stated in advance.** We do not know whether TabPFN-3.5's training data include public sensor records of
2026. If they do, this favours TabPFN-3.5, not the pool.

## Execution

| Step | Where |
|---|---|
| Synthetic smoke test; then TabPFN-3.5 on the four panels, one container each (`modal_tabpfn35.py`, `modal_spawn.py`) | Modal |
| Outputs (`cells_*`, `pq_*`, `train.json`) copied to `artifacts/v13_tabpfn35/<net>/` and committed | local |
| `run_v13.py`: copies them to `runs_real/<net>/tabpfn_v3_5`, runs `pool_eval.py --tab tabpfn_v3_5` for `upt_L_sfm` and `pfn_L`, then `confirm13.py` | Colab (Drive holds the UPT-SF and PFN-L outputs) |

## Amendment 1 (October 13, 2026, before any real-data output exists): execution only

The Modal workspace runs one GPU container at a time, so the four networks queued one after another (about 3.4 hours).
`usgs_ny` runs as planned in its own container. The other three run together in one container
(`modal_tabpfn35_many.py`): one `tabpfn_colab.py` process per network, with the same arguments, on one L4 (one process
used about a quarter of it). This changes neither the model, the settings nor the outputs.
