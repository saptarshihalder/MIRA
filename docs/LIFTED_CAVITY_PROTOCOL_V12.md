# Lifted cavity: protocol v12: is the lifted-site model what makes the pool with TabPFN work?

Committed October 13, 2026, before any of the pools below was computed. At commit time no per-query output of PFN-L or
of the closed-form FA-Gaussian exists on the v10 networks, so neither pool can have been looked at.

The v10/v11 results on these networks are known (`LIFTED_CAVITY_V10_V11_RESULTS.md`). This protocol tests a
mechanism on data that are therefore no longer fresh, and says so.

## Question

UPT ⊕ TabPFN (v11) beat TabPFN v2 on every v10 network. A linear pool with stacked weights can help with many partners.
Does the gain come from the lifted-site model, or would any reasonable partner do?

## Pools (identical rule: `pool_eval.py`, per-episode two-fold cross-fit stacking, grid {.02, …, .98})

| Pool | Partner | Recipe |
|---|---|---|
| **UPT-SF ⊕ TabPFN** | `upt_L_sfm` with `tta8` | The v11 pool, already computed. |
| **PFN-L ⊕ TabPFN** | PFN-L, the matched-scale transformer without lifted sites, v9 recipe (`pfn_L`), with `tta8` | Fine-tuned in v10. |
| **FA-Gaussian ⊕ TabPFN** | The closed-form lifted-site model, UPT's anchor (`pq_closed.py`; no learning) | |

**Shared ingredients:**
- TabPFN v2's per-query and cross-fit outputs are those of v10/v11 (Modal).
- All partners use the same cross-fit masks (`realdata.crossfit_masks`, seed 20000 + extra).

## Endpoints (`confirm12.py`), on the four v10 networks, pooled

**Episodes:** those where TabPFN v2 is defined. There, each pool is a pool; elsewhere it is its partner alone.

**Statistics:** as v9–v11.
- Paired 95% intervals are cluster-robust over network-windows.
- Gain = baseline pool − UPT-SF pool.
- Pass: gain ≥ .01 nats and lower bound > 0.

| ID | Comparison | Conditions |
|---|---|---|
| Y1 | UPT-SF ⊕ TabPFN vs PFN-L ⊕ TabPFN, NLL | all sensors, 3 dropped, 6 dropped (three tests) |
| Y2 | UPT-SF ⊕ TabPFN vs FA-Gaussian ⊕ TabPFN, NLL | the same three conditions |

**Secondary:**
- each partner's pool vs TabPFN alone;
- squared error;
- each network separately.

**Descriptive:** mean NLL and mean stacking weight for every pool.

**Reading, fixed here:**
- If Y1 passes, the lifted-site transformer is a better partner than a plain transformer of equal size and training.
- If Y2 passes, learning on top of the closed-form sites adds something to the pool.

All outcomes are reported.

## Execution

**Colab:** `eval_real.py --per-query --tta 8` for `pfn_L_s{1,2,3}_ft` on the four panels, and `pq_closed.py` for the
four panels. Then `pool_eval.py` (`--model pfn_L`; `--model fa1_closed --single --sfx ''`), then `confirm12.py`.
