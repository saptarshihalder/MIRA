# Lifted cavity: protocol v9: three 2026 sensor networks in new domains, scale and two small model changes

Committed October 12, 2026, before any v9 panel was built, any v9 model was fine-tuned and any development run of the
two model changes was scored. The only runs so far are code-path smoke tests (3 synthetic training steps, 5 fine-tuning
steps on 64 DWD source episodes, scoring and 20-step kriging on 16 DWD test episodes); their numbers are meaningless and
are not used.

## Why

Reviewers can fairly say that the v5–v8 networks are old (2012–2018) benchmarks, that the domains are mostly traffic,
that the full-sensor comparison with TabPFN v2 was mixed (v6 P3 failed; v7 China air), and that point accuracy (RMSE)
was not better than the matched transformer. v9 tests the method on networks recorded in 2025–2026, after the
release of every pretrained model compared here (TabPFN v2: January 2025), in three domains the project has never used.
It also scales up the model and tests two small changes aimed at the point-accuracy gap.

## Networks (`fetch_v9.py`, `realdata_v9.py`)

| Network | Domain | Source | Sensors kept | Step | Values | Inputs | Episodes | Source period | Test period |
|---|---|---|---|---|---|---|---|---|---|
| `dwd_t2m` | weather | DWD hourly 2 m air temperature, `recent` archive | ≥ 75% observed in both periods | 1 h | °C | 11 nearest | weekly | 2025-05-01 – 2026-04-01 | 2026-04-01 – 2026-10-01 |
| `aurn_no2` | air quality (UK) | Defra AURN hourly NO2, 2025 + 2026 site files | same | 1 h | log1p(µg/m³) | 11 nearest | weekly | 2025-01-01 – 2026-04-01 | 2026-04-01 – 2026-10-01 |
| `usgs_q` | hydrology | USGS NWIS discharge (00060), active Pennsylvania stream gauges | same | 30 min | log1p(ft³/s) | 11 most correlated over the source period | 3-day | 2026-03-01 – 2026-07-01 | 2026-07-01 – 2026-10-01 |

- **Snapshot.** Recent data are provisional and may be revised by their publishers. The snapshot is the Colab download
  (`mira_v9/data_v9/*.npz`). Its SHA-256 is recorded in every panel, and that snapshot is the data of record.
- **Test targets.** 48 per network, seeded.
- **Episodes.** 48 support and 48 query rows from rows where the target is observed (`realdata._episode`).
- **Support dropout.** Support rows get 20% simulated per-sensor dropout on all three networks, because they are
  nearly complete. Query rows keep the natural mask.
- **Panels.** Panel seed 9091, with extra-dropout conditions 0, 3 and 6 (`eval_real.py build`).
- **Clusters.** Network-windows.
- **Local check.** A local download for checking the code found:
  - 487 DWD stations kept, 98% observed, 1,247 test episodes in 26 weeks;
  - 137 AURN sites kept, 96% observed, 1,220 test episodes in 26 weeks.
  Only counts were looked at, never a model score. The Colab snapshot may differ slightly.

## Models

**Source checkpoints:** existing, never retrained here.

| Model | Checkpoint | Size | Notes |
|---|---|---|---|
| PFN, UPT | v6 | small | |
| UPT-L | v7 | d = 128, 8 layers, 2.1M parameters | |
| PFN-L | v4 | same size as UPT-L | The matched-scale transformer. |
| CNP, TNP-D | v8 | | |

**Fine-tuning.** Every model is fine-tuned on each network with the protocol-v2 recipe (`finetune_real.py` defaults),
seeds 1–3.

**Baselines with their own pipelines:**
- TabPFN v2, in context, 8 estimators, as in v4–v7. It now also records CRPS and 90% coverage from 100 predictive
  quantiles: CRPS = 2 × the mean pinball loss at levels (i + .5)/100.
- IGNNK and calibrated IGNNK, recipe of v8.
- The closed forms in every panel.

### Two small changes (UPT-L only), evaluated on development data first

1. **Test-time ensemble (`tta8`).**
   - The predictive is the equal-weight linear pool of the model's predictives under feature-embedding seeds 0–7. Seed
     0 is the single-model default (`eval_real.py --tta 8`).
   - This is the same device as TabPFN's 8-member ensemble, which every TabPFN score already uses.
   - A linear pool of linear pools is again a linear pool. It therefore keeps the marginalization property of
     McConway (1981) that motivates UPT.
   - No training cost.
2. **CRPS-augmented fine-tuning (`upt_L_crps`).**
   - Objective: NLL + 1 × CRPS / σ, where σ is the episode's support-target standard deviation (`upt.nll_crps`). The
     CRPS is the closed form of Grimit et al. (2006) for Gaussian mixtures.
   - Both terms are strictly proper scoring rules (Gneiting & Raftery 2007), so their sum is too.
   - The CRPS term rewards a well-placed predictive centre, which targets the RMSE gap. Weight 1 is fixed here and is
     not tuned.

## Development stage (chain B), before any v9 score

**Development networks:** the protocol-v7 networks PeMS03, PeMS08 and China air. They are already used, at extra = 0.

**Runs:**
- `upt_L_crps` is fine-tuned on these three networks with seeds 1–3.
- `upt_L` and `upt_L_crps` are scored with and without `tta8`.

**Selection rule (`select9.py`), fixed here:**

- **Candidates:**
  - A: UPT-L;
  - B: UPT-L with `tta8`;
  - C: UPT-L with the CRPS objective;
  - D: UPT-L with the CRPS objective and `tta8`.
- **Statistics:** per-episode NLL and CRPS / σ, averaged over seeds and then over all episodes.
- **Rule:** among the candidates whose mean NLL is within .02 nats of the best, choose the lowest mean CRPS / σ. Ties
  go to the earlier letter.
- **UPT\*:** the chosen candidate. It is the model of the v9 endpoints.

**Also on development data** (descriptive only):
- UPT-L on the three v6 networks (METR-LA, PEMS-BAY, Intel);
- PFN-L on all six v6 and v7 networks;
- `tta8` scores for both.

These complete the "scale everywhere" picture and give the matched-scale comparison on old data. They are reported,
and are not confirmations.

**If UPT\* uses the CRPS objective,** PFN-L is also fine-tuned with it on the v9 networks (`pfn_L_crps`). The PFN-L
comparator is then the one of {PFN-L, PFN-L-CRPS} with the lower pooled extra = 0 NLL on the v9 panels. It always uses
the same test-time setting as UPT\*. Choosing the stronger comparator is conservative for UPT\*.

## Endpoints (`confirm9.py`), on the three v9 networks, pooled

**Statistics:**
- Per-episode metrics.
- Learned models use the seed average over seeds 1–3.
- Paired 95% intervals are cluster-robust over network-windows.
- Gain = baseline − UPT\*; positive favours UPT\*.
- Standardized metrics divide by the episode's support-target standard deviation σ: CRPS / σ and squared error / σ².
- Episodes on which TabPFN is undefined (constant support target) are excluded from TabPFN comparisons and counted.

**Pass rule:**
- NLL endpoints: gain ≥ .01 nats and lower bound > 0.
- CRPS endpoint: lower bound > 0.

| ID | Comparison | Metric | Condition |
|---|---|---|---|
| V1 | UPT\* vs TabPFN v2 | NLL | all sensors (extra = 0) |
| V2 | UPT\* vs TabPFN v2 | NLL | 3 and 6 extra dropped (two tests) |
| V3 | UPT\* vs PFN-L comparator (matched scale, same test-time setting) | NLL | extra = 0 |
| V4 | UPT\* vs TNP-D | NLL | extra = 0 |
| V5 | UPT\* vs calibrated IGNNK kriging | NLL | extra = 0 |
| V6 | UPT\* vs TabPFN v2 | CRPS / σ | extra = 0 |

**Secondary** (same statistics, all reported):
- Every comparison above at extra = 0, 3 and 6, for NLL, CRPS / σ and squared error / σ².
- The same comparisons against CNP, small UPT and small PFN.
- UPT-L (candidate A, unmodified) against TabPFN v2.
- Each network separately: UPT\* vs TabPFN, PFN-L, TNP-D and calibrated kriging, for NLL and CRPS.

**Descriptive** (v9 and development networks):
- NLL, RMSE (raw and standardized), CRPS / σ and 90% coverage for every method;
- the closed forms.

All outcomes are reported, whichever way they go.

## Execution

**Hardware:** Colab A100 (`colab_v9.py`, launched by `colab_launch.py v9`).

| Chain | Steps |
|---|---|
| A9 | data → panels → pools → fine-tuning of PFN, UPT, CNP, TNP-D, UPT-L and PFN-L → kriging → TabPFN → wait for the selection → scoring (with `tta8` for the transformer models) → `confirm9.py` |
| B9 | development fine-tuning → development scoring → `select9.py` → the selected variant on the v9 networks |

**Amendments:** committed before the affected scores are computed.

## Amendment 1 (October 12, before any v9 panel, fine-tune or score)

**The PFN-L source checkpoints do not exist.** The protocol-v4 PFN-L checkpoints named above were never kept on Drive.
PFN-L is therefore trained here, seeds 1–3, into `mira_v9/runs`, with the protocol-v4 recipe exactly:
- `pfn_train.py`;
- 40,000 steps, batch 32 tasks × 16 queries;
- lr 5e-4, warm-up 2,000;
- d = 128, 8 layers, 8 heads, feed-forward width 512.

This is the same size and source-training budget as UPT-L.

**Chain C** trains it. Every PFN-L fine-tune waits for it:
- PFN-L on the v9 networks, in chain A;
- PFN-L-CRPS, if selected, in chain B;
- PFN-L on the development networks, moved to the end of chain B. These are descriptive and not needed by the
  selection rule.

**Concurrency** changed to 6 and 4 parallel jobs (speed only).

Nothing else changes.
