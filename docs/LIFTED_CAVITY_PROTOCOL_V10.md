# Lifted cavity: protocol v10: a scale-free UPT, confirmed on four fresh 2026 networks

Committed October 12, 2026. At commit time:
- no v10 network has been built, fine-tuned on or scored;
- no development run of the changes below has been scored.

The only runs are CPU smoke tests: 5 fine-tuning steps on 64 DWD source episodes, and scoring on 16 v9 DWD test
episodes. Their numbers are meaningless and are not used.

## Why

**v9 failed V1/V2** (NLL vs TabPFN v2, pooled) only because of the river network, USGS Pennsylvania. There, TabPFN's
predictive is far sharper; see `LIFTED_CAVITY_V9_RESULTS.md`. The diagnosis found two fixed scales in our model
family, which UPT and PFN share:

1. **The target is standardized by max(σ, .1).** This is the backbone's .1 clamp. The free head's log-variance is
   also bounded at ±5 around 2 log σ. The predictive therefore cannot be sharper than e^(−2.5) · max(σ, .1).
2. **The closed-form anchors have absolute regularizers** (variance + 1e-3, ridge 1e-4), which act on the raw target.

TabPFN standardizes the target per task and has no such floor.

## The change (code: `realdata.zpool`, `finetune_real.py --yz --lv-bound`, `upt.UPT(free_k)`)

**SF, the scale-free pipeline.**
- Each episode's target is standardized by its support mean and standard deviation, with a floor of
  1e-4 · (1 + |mean|).
- The anchors are refitted on the standardized target.
- Every predictive is mapped back to raw units before scoring: NLL + log s, CRPS × s, squared error × s².
- The inputs were already standardized per episode, so the predictive becomes equivariant to affine maps of the
  target. This is tested exactly in `tests/test_lifted_upt.py`.
- The log-variance bound becomes ±10.
- It is a pipeline change, so it applies to any of our learned models. The matched transformer gets it too.

**M, the mixture free site (`--free-k 3`).** UPT's free site becomes a 3-component Gaussian mixture (Bishop 1994):
- It starts from the single-Gaussian checkpoint as a scale mixture: log-variance offsets −1.5, 0 and +1.5, equal
  weights.
- The pool remains a linear pool of sites: the lifted site plus three free components.

The objective stays UPT\*'s (NLL + 1 × CRPS / σ). Scoring uses the 8-member test-time ensemble (`tta8`).

## Development stage (chain D10), before any v10 score

**Development networks:** the three v9 networks, now used: DWD temperature, AURN NO2 and USGS Pennsylvania.
Seeds 1–3, extra = 0, `tta8`.

**Runs fine-tuned there:**
- E = `upt_L_sf` (SF);
- F = `upt_L_sfm` (SF + M);
- `pfn_L_sf`, for information.

**Selection rule (`select10.py`), fixed here:** choose F if its mean NLL is lower than E's by at least .005 nats;
otherwise choose E, the simpler model. The choice is called **UPT-SF**.

**Descriptive only (chain O10):** UPT-SF and PFN-L-SF on the six v6/v7 networks. Together with the v9 networks, this
reports all thirteen networks.

## Fresh networks (`fetch_v10.py`, `realdata_v10.py`; never used by this project)

| Network | Domain | Source | Sensors found | Values | Inputs | Episodes | Source / test period |
|---|---|---|---|---|---|---|---|
| `usgs_ny` | hydrology | USGS NWIS discharge, active New York stream gauges, 30 min | 304 | log1p(ft³/s) | 11 most correlated (source period) | 3-day | 2026-03-01 – 07-01 / 07-01 – 10-01 |
| `usgs_wa` | hydrology | the same for Washington State (snowmelt- and glacier-fed) | 278 | same | same | same | same |
| `dwd_p` | weather | DWD hourly station pressure P0, `recent` archive | 214 | hPa | 11 nearest | weekly | 2025-05-01 – 2026-04-01 / 2026-04-01 – 10-01 |
| `aurn_pm25` | air quality | Defra AURN hourly PM2.5, 2025 + 2026 | 170 | log1p(µg/m³) | 11 nearest | weekly | 2025-01-01 – 2026-04-01 / 2026-04-01 – 10-01 |

**Common settings:**
- Keep sensors with ≥ 75% coverage in both periods.
- 20% simulated support dropout.
- 48 test targets.
- Panel seed 10101, with conditions 0, 3 and 6 extra dropped sensors.
- The snapshot of record is the Colab download (SHA-256 in each panel).

**Local check:** a local download for checking the code kept 206 DWD pressure stations, 98% observed. Only counts were
looked at.

## Models on the fresh networks (seeds 1–3, protocol-v2 fine-tuning recipe, `tta8` for the transformers)

| Model | Role |
|---|---|
| UPT-SF | The model under test. |
| PFN-L-SF (`pfn_L_sf`: SF + CRPS objective) and PFN-L-SFN (`pfn_L_sfn`: SF, NLL only) | Matched-scale transformer with the same pipeline. The comparator is whichever has the lower pooled extra = 0 NLL (conservative). |
| UPT\* (v9 recipe, `upt_L_crps`) | Ablation: the effect of SF. |
| TNP-D | Fine-tuned from v8. |
| Calibrated IGNNK kriging | v8 recipe. |
| TabPFN v2 | 8 estimators, with CRPS and coverage from quantiles as in v9. |

## Endpoints (`confirm10.py`), on the four fresh networks, pooled

**Statistics:** as in v9.
- Paired 95% intervals are cluster-robust over network-windows.
- Gain = baseline − UPT-SF; positive favours UPT-SF.

**Pass rule:**
- NLL endpoints: gain ≥ .01 and lower bound > 0.
- CRPS endpoint: lower bound > 0.

| ID | Comparison | Metric | Condition |
|---|---|---|---|
| W1 | UPT-SF vs TabPFN v2 | NLL | all sensors |
| W2 | UPT-SF vs TabPFN v2 | NLL | 3 and 6 extra dropped (two tests) |
| W3 | UPT-SF vs PFN-L-SF comparator | NLL | extra = 0 |
| W4 | UPT-SF vs TabPFN v2 | CRPS / σ | extra = 0 |
| W5 | UPT-SF vs TNP-D | NLL | extra = 0 |
| W6 | UPT-SF vs calibrated kriging | NLL | extra = 0 |
| W7 | UPT-SF vs UPT\* | NLL | extra = 0 (does SF help on fresh data?) |

**Secondary, all reported:**
- every comparison at extra = 0, 3 and 6, for NLL, CRPS / σ and squared error / σ²;
- each network separately.

**Descriptive:**
- UPT-SF vs TabPFN v2 NLL pooled over all thirteen networks;
- per-network tables for every method.

All outcomes are reported, whichever way they go.

## Execution

**Hardware:** Colab A100 (`colab_v10.py`, launched by `colab_launch.py v10`).

| Chain | Steps |
|---|---|
| D10 | development fine-tuning → scoring → `select10.py` |
| F10 | fresh data → panels → pools → kriging → fine-tuning of PFN-L-SF, PFN-L-SFN, UPT\* and TNP-D → UPT-SF once selected → scoring → `confirm10.py` |
| T10 | TabPFN v2, two networks at a time |
| O10 | the old-network descriptives |

**Amendments:** committed before the affected scores are computed.

## Amendment 1 (October 12, before any v10 panel or score)

**The bug.** `fetch_v9.parse_rdb` converted only Eastern-time timestamps (EST/EDT) to UTC. Every Washington gauge
(PST/PDT) was therefore dropped, and the `usgs_wa` download failed with no gauges.

**The fix.** The parser now maps every US time zone. The Eastern-time networks are unaffected: v9 `usgs_q` and v10
`usgs_ny`.

**Effect.** Chain F10 stopped at its data stage before building any panel. It is relaunched with the fix. Nothing
else changes.

## Amendment 2 (October 12, before any v10 score)

**What happened.** On Colab, the `usgs_ny` download returned data for only 22 of the 304 active gauges. Probing showed
why: the legacy NWIS instantaneous-value service answers "503 service temporarily unavailable" for New York gauges.
The Pennsylvania and Washington gauges are unaffected. Only 19 gauges were kept, which is fewer than the 48 test
targets, so the panel build stopped. No panel of this network, and no score of any v10 network, was produced.

**The fix.** For each gauge, the download first tries the legacy service. If that fails, it falls back to the USGS
Water Data OGC API (`api.waterdata.usgs.gov`, collection `continuous`, parameter 00060). This is the same continuous
discharge record in UTC, fetched in one request per gauge, plus a second request when a 5-minute record
reaches the 50,000-row page limit. The API allows 1,000 requests per hour per address. In addition:
- requests are sent at most 4 at a time, with back-off;
- a second, sequential pass retries the gauges that failed;
- the JSON summary records how many requests still failed;
- the `usgs_ny` snapshot is deleted and downloaded again.

The network, splits and every other choice are unchanged. The coverage rule (≥ 75% in both periods) still decides
which gauges are kept.

## Amendment 3 (October 12, before any v10 score)

**What happened.** The OGC fallback of amendment 2 also failed on Colab. The API allows 1,000 requests per hour per
address, and Colab's shared outbound address had already used the quota (HTTP 429). The relaunched download was
stopped before it wrote a snapshot. The legacy service still answers 503 for New York gauges.

**The fix.**
- `usgs_ny` is downloaded from a local machine by `fetch_v10.py`, with the same code, sequentially and paced. It sends
  at most one OGC request per 3.7 s and waits out any 429.
- The snapshot is committed to the repository (`data_snapshots/v10/usgs_ny.npz` and `.json`).
- `colab_v10.py data` copies committed snapshots into the Drive data folder before downloading anything else.

The data source, the network definition and the coverage rule are unchanged.

## Amendment 4 (October 12, before any v10 score; compute budget)

About 74 Colab compute units remain, roughly 5.5 A100-hours. The remaining work is cut to the endpoints that matter:

1. **Not run:**
   - `pfn_L_sfn`;
   - `upt_L_crps` (UPT\*) on the fresh networks. **W7 is therefore not run** and is reported as such.
2. **Added comparator:** PFN-L with the v9 recipe (`pfn_L`: NLL, no SF, `tta8`). The W3 comparator is the best
   available PFN-L variant by pooled extra = 0 NLL, now chosen from {PFN-L-SF, PFN-L}.
   - **Why:** the development cells showed PFN-L-SF failing on near-flat river episodes.
   - **Effect:** adding the v9 recipe can only make the comparator stronger, so the change is conservative for UPT-SF.
3. **Chain O10 is stopped** after the UPT-SF fine-tunes on the six v6/v7 networks. Those runs are scored in chain F10
   (`scoreoldsel`). PFN-L-SF on these networks and the descriptive v9 pool are not run.
