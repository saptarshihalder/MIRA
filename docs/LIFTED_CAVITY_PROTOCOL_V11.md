# Lifted cavity: protocol v11: lifted sites make a foundation model robust (UPT ⊕ TabPFN)

Committed October 12, 2026, before any score on the v10 networks exists. No v10 panel has been scored by any model:
- the TabPFN v2 runs of chain T10 have not started;
- no v10 fine-tune has been scored.

The development selection of v10 (`selection10.json`: UPT-SF = `upt_L_sfm`) was computed on the v9 networks, which
are already used.

## Why

**The v9 and v10 development results.**
- TabPFN v2, a foundation model pretrained on a vast prior, has the sharper predictive on smooth, near-deterministic
  networks (rivers). Its NLL degrades sharply when input sensors go missing.
- UPT's lifted sites handle missing inputs exactly.

**The proposal.** The linear opinion pool is the pooling rule that commutes with marginalization. Proposition 5
bounds its NLL by each component's NLL plus the log of that component's weight. So pooling TabPFN with UPT should
keep TabPFN's sharpness where it is right, and UPT's robustness where TabPFN fails. The weight must be chosen per
episode without test labels.

## Method (fixed here; code: `pool_eval.py`, `realdata.crossfit_*`, `tabpfn_colab.py`, `eval_real.py --per-query`)

**The pool.**
- p(y | x_S, D) = w · p_UPT-SF(y | x_S, D) + (1 − w) · p_TabPFN(y | x_S, D).
- UPT-SF is the v10-selected model `upt_L_sfm` with its 8-member test-time ensemble, for each seed 1–3.
- TabPFN v2 runs exactly as in v9/v10: 8 estimators, NaN for missing inputs.

**The weight: per-episode stacking (Yao, Vehtari, Simpson & Gelman 2018).**
1. Split the episode's 48 labelled support rows into halves A (rows 0–23) and B (rows 24–47).
2. Each model predicts B from A and A from B. The held-out rows' inputs use the episode's support mask, with the
   condition's extra dropped inputs applied by `realdata.crossfit_masks`, seed 20000 + extra. The masks are identical
   for both models.
3. Pick w on the grid {0.02, 0.04, …, 0.98} that maximizes the summed log density of the pool on the 48 held-out rows.

This is done separately for each episode, condition and seed. No test label is used and nothing is trained.
Inputs keep the episode's standardization. UPT-SF's SF target standardization uses only the half that is fitted.

**Undefined TabPFN.** Where TabPFN is undefined (constant training target), its density is left out: w = 1 for that
query, and the cross-fit uses the rows where both are defined. If none are, w = 1.

**Metrics.** Per-query NLL of the pool, and squared error of the pool mean, averaged per episode. Episode metrics are
averaged over the three UPT seeds.

## Endpoints (`confirm11.py`), on the four fresh v10 networks (`usgs_ny`, `usgs_wa`, `dwd_p`, `aurn_pm25`), pooled

**Statistics:** exactly as v9/v10.
- Paired 95% intervals are cluster-robust over network-windows.
- Gain = baseline − pool.
- Pass: gain ≥ .01 nats and lower bound > 0.
- TabPFN-undefined episodes are excluded from TabPFN comparisons.

| ID | Comparison | Condition |
|---|---|---|
| X1 | pool vs TabPFN v2, NLL | all sensors |
| X2 | pool vs TabPFN v2, NLL | 3 and 6 extra dropped (two tests) |
| X3 | pool vs UPT-SF, NLL | all sensors |

**Secondary:**
- pool vs UPT-SF at 3 and 6 extra dropped;
- squared error vs TabPFN;
- each network separately.

**Descriptive:**
- the same pool on the three v9 networks (development, already used);
- the mean stacking weight.

All outcomes are reported.

## Execution

| Chain | Steps |
|---|---|
| T10 | TabPFN v2's v10 run (`--crps`) also writes the per-query and cross-fit outputs (`pq_<tag>.npz`) in the same pass. The cross-fit calls come after each episode's main call, so the v10 scores are unchanged. It runs four networks at a time. |
| F10 | scores UPT-SF per query (`--per-query`) → `pool_eval.py` → `confirm11.py` after `confirm10.py` |
| O10 | adds the v9 descriptive pool (`devpool`) |
