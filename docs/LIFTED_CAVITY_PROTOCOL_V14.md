# Lifted cavity: protocol v14: TabPFN-3.5 with UPT's prediction as an input column

Committed October 13, 2026, before any TabPFN-3.5 run on the development networks and before any run of the method
below. The TabPFN-3.5 outputs of `usgs_ny` (v13) have been looked at (TabPFN-3.5 alone, not pooled); no other v13
output had finished.

## Why

On `usgs_ny`, TabPFN-3.5 alone is far sharper than TabPFN v2 and than UPT-SF ⊕ TabPFN v2, and it degrades little when
sensors fail. A linear pool can never be sharper than its sharper member; its gain comes only from hedging. UPT-SF's
advantage is different in kind: it is fine-tuned across the training windows of each network, while TabPFN sees only
the episode's 48 labelled rows. Giving TabPFN UPT-SF's prediction as an input lets TabPFN learn, within each episode,
how far to trust it and how to correct it. Under sensor failure the column stays complete, because UPT-SF's lifted
sites marginalize missing sensors exactly.

## Method (fixed here; code: `tabpfn_feat.py`)

TabPFN-3.5 exactly as in v13 (tabpfn 9.1.0, `ModelVersion.V3_5`, 8 estimators, random state 0, NaN for missing
inputs, constant-support episodes undefined), with one more input column:
- query rows: UPT-SF's test prediction under the condition's masks;
- support rows: UPT-SF's two-fold cross-fitted prediction (each half from the other half, under the shared cross-fit
  masks of v11), so no row's own label enters its column;
- UPT-SF = `upt_L_sfm` with `tta8`, predictions averaged over seeds 1–3.

Two variants, identical with all sensors:

| Variant | Support inputs at 3 / 6 extra dropped |
|---|---|
| F1 `uptcol` | natural missingness (as TabPFN alone) |
| F2 `uptcol_mirror` | the cross-fit masks of the condition, under which their column was computed |

## Development (the v9 networks `dwd_t2m`, `aurn_no2`, `usgs_q`; already used)

Runs: TabPFN-3.5 alone (`tabpfn_colab.py --version v3.5 --allow-constant-support --crps`, which also writes the
per-query and cross-fit outputs), F1 and F2, the v13 pool rule with TabPFN-3.5 (descriptive).

**Selection, fixed here.** On TabPFN-defined episodes, take the mean NLL over the three dev networks and the three
conditions. The selected variant is the better of F1 and F2. Go to confirmation only if the selected variant's
pooled mean NLL is lower than TabPFN-3.5 alone's at every one of the three conditions. Otherwise stop, and report the
development results.

## Confirmation (the v10 networks `usgs_ny`, `usgs_wa`, `dwd_p`, `aurn_pm25`; not fresh: used by v10–v13)

**Statistics:** as v9–v13. Paired 95% intervals cluster-robust over network-windows; gain = baseline − selected
variant; pass: gain ≥ .01 nats and lower bound > 0; TabPFN-undefined episodes excluded.

| ID | Comparison | Condition |
|---|---|---|
| R1 | selected variant vs TabPFN-3.5 alone, NLL | all sensors |
| R2 | selected variant vs TabPFN-3.5 alone, NLL | 3 and 6 extra dropped (two tests) |

**Secondary:** the selected variant vs UPT-SF ⊕ TabPFN-3.5 (v13 pool) at 0, 3, 6 dropped; squared error, CRPS and 90%
coverage vs TabPFN-3.5; each network separately.

All outcomes are reported, in the paper as well.

## Execution

Colab (Drive holds the panels and UPT-SF's per-query outputs; the TabPFN license token is a Colab secret, read into
the environment and never printed): `run_v14.py dev`, then `run_v14.py confirm` if the selection rule allows it.

## Amendment 1 (October 13, 2026, before any v14 run): more variants, a smaller development set

Nothing of v14 had run. On request, the development stage now compares four ways of combining TabPFN-3.5 with UPT-SF
beyond a linear pool (code: `tabpfn_feat.py`, `run_v14.py`), and drops F2, which had the least support:

| Variant | Combination |
|---|---|
| F1 `uptcol` | UPT-SF's predictive mean as an input column (as above) |
| F3 `uptcol2` | UPT-SF's predictive mean and log standard deviation as input columns: TabPFN can learn when to trust UPT-SF |
| F4 `uptres` | TabPFN models the residual y − m with the two columns; the predictive of y is m + residual (TabPFN corrects UPT-SF) |
| F5 `uptsite` | UPT-SF as a Gaussian site on TabPFN-3.5's predictive: p(y) ∝ p_TabPFN(y)^a · N(y; m, v)^b on TabPFN's bars (half-normal tails kept). (a, b) on {1, .8, .6} × {0, .05, .1, .2, .35, .5, .75, 1} per episode and condition, by the two-fold cross-fit log score of the support. (1, 0) is TabPFN alone and wins ties. |

- m, v: UPT-SF's predictive mean and variance, seed-averaged as the equal mixture of seeds 1–3 (`eval_real.py
  --per-query --pq-var`, which writes `pqv_*` files and never overwrites a `pq_*` file of record). Support rows always
  use the two-fold cross-fitted predictive.
- TabPFN-3.5 alone on the development networks comes from F5's pass (its (1, 0) member: the calls of
  `tabpfn_colab.py`), in `tabpfn_v3_5_s2`.
- Development scores every second episode (0, 2, 4, …) of each v9 network, for time. Confirmation scores every
  episode.

**Selection (replaces the rule above):** the selected variant has the lowest mean development NLL over the three
networks and three conditions (scored, TabPFN-defined episodes). Go to confirmation only if it beats TabPFN-3.5 alone
at each of the three conditions, pooled over the development networks. The endpoints R1–R2 are unchanged and test only
the selected variant.
