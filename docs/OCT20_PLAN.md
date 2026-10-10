# October 11–20 plan: a NeurIPS-level submission (assumed deadline October 20)

**Supersedes:** `OCT10_PLAN.md`.
**Read first:** `LIFTED_SITE_POOL_THEORY.md` and `LIFTED_CAVITY_PROTOCOL_V5.md`.

## The paper in one sentence

Lifted sites are the free (universal) construction for missing sensors. The transformer enters as one more site
through the unique pool that is natural for marginalization. One network, the **Universal Pool Transformer (UPT)**,
therefore keeps the lifted model's exact extrapolation and the transformer's real-data fit, at the cost of one
backbone pass.

## Why this is the fastest credible route

- **Theory exists.**
  - U1: lifted sites are the free commutative-monoid construction.
  - U2: Genest's externally Bayesian pool (log pool).
  - McConway's marginalization property (linear pool).
  - The Hölder bound.
  - Propositions 2–3.

  All of these are tested in `tests/test_lifted_site_pool.py`.
- **Code exists.** `CellPFN`, `LCT`, `LiftedCavity`, the real-data episode builder, fine-tuning and scoring, and
  `site_pool.py`.
- **Current reviewer-facing weakness:** "it is an ensemble" (no significant E16 on NO2), plus one real data source.
  UPT answers the first: one model, one pass. Three new sensor networks answer the second.

## Model (fixed before any v6 run)

**UPT** is the `CellPFN` backbone with three heads:
1. the LCT lifted-site head on query cell tokens, giving the lifted posterior p_site;
2. a free Gaussian head on the target-column token, giving p_free;
3. a gate π = σ(g(target token)).

The output is the McConway-natural linear pool:

    p = π · p_site + (1 − π) · p_free

The loss is the NLL by logsumexp.

**Initialisation:** zero site head, gate bias +4. The untrained model is therefore π ≈ 0.98 times the FA-Gaussian
closed form, preserving "exact at initialisation" up to a stated ε.

**Parameters:** the LCT's plus 3d + 3 (about 0.1% more). Inference is one backbone pass, half the cost of the
two-model pool.

**Variants, as ablations only:**
- log pool instead of linear;
- fixed π = ½;
- no free head (= LCT);
- no site head (= PFN).

## Data (fresh data marked *)

1. Beijing, six pollutants, natural missingness. Loader exists.
   - PM10, SO2 and O3 are v5 data.
   - PM2.5, NO2 and CO are used data and descriptive only.
2. UCI Air Quality CO and NO2. Loader exists; used data, descriptive.
3. *UCI Gas Sensor Array Drift.* Loader `realdata.load_gas` exists. Check that it was never scored before using it as
   fresh data.
4. *Intel Berkeley Research Lab* (54 motes; temperature and humidity; heavy natural missingness). New loader: predict
   one mote from the others, weekly episodes, temporal split.
5. *METR-LA traffic speed* (207 sensors; natural zero-readings as missing). New loader: one sensor from its 15
   nearest, daily episodes, temporal split.

**Rules for new loaders:**
- Develop only on source periods.
- Freeze each test panel by hash before any model is scored on it.

## Baselines (every table)

- Existing: FA-Gaussian, EM-Gaussian, BLR, NL-FA, the oracle on synthetic data, PFN, LCT, `lift1`, and the
  training-free pool (v5).
- **TabPFN v2, in context**, with native NaN (code exists from v4).
- **LightGBM with native missing values**, fitted per episode.
- **MICE + BLR**, using sklearn `IterativeImputer`.
- **Two-seed PFN ensemble** (linear pool), the compute-favoured control.
- **Optional, if time allows:** NeuMiss (Le Morvan et al. 2020).

## Endpoints (protocol v6; freeze before the first v6 fine-tune)

All are pooled over the fresh real networks (3–5), with week/day-clustered 95% intervals; 3 seeds per learned model.

| ID | Comparison | Pass condition |
|---|---|---|
| P1 | UPT vs fine-tuned PFN | gain ≥ .01, lower bound > 0 |
| P2 | UPT vs two-seed PFN linear pool | non-inferiority, lower bound > −.01 (UPT uses half the compute) |
| P3 | UPT vs TabPFN v2 in context | gain ≥ .01, lower bound > 0 |
| P4 | UPT vs PFN, synthetic, 16 sensors after training on 5 | gain ≥ .01, lower bound > 0 (extrapolation is kept) |

**Secondary, descriptive:**
- the share of the closed-form→oracle gap closed;
- each dataset and target;
- extra dropped sensors;
- latency;
- the ablations;
- v4 at scale (PFN-L / LCT-L) when integrated.

## Schedule with exit gates

| Date | Work | Exit gate |
|---|---|---|
| Oct 11 | v5 result (running on CPU). Audit the LCT replay mismatch (below). Implement UPT plus tests: exact at init, site-part homomorphism, gate bounds. Freeze protocol v6. **Secure GPU.** | v5 endpoints reported; UPT tests pass; v6 committed |
| Oct 12 | Intel Lab and METR-LA loaders; gas eligibility check. Build source pools and test panels (CPU). GPU smoke. Launch source training: PFN, LCT, UPT × 3 seeds, shared recipe. | panels hashed; 9 source models training |
| Oct 13 | Fine-tune every model × seed × target (GPU batch). Baselines on CPU in parallel; TabPFN on GPU. | every score archive complete and nonfinite-free |
| Oct 14 | `confirm6.py`: all endpoints. Decide the headline (see fallback). | endpoints final |
| Oct 15 | Ablations, latency table, synthetic oracle table with UPT. Integrate v4 if collected. | every table cell traced to a file |
| Oct 16–17 | Write the paper from the outline below. `make_paper.py` macros only; no hand-typed numbers. | full draft, 9 pages |
| Oct 18 | Adversarial review by a separate agent acting as a NeurIPS reviewer. Related-work audit (learned EP, gPoE, PFN/TabPFN, NeuMiss, Deep Sets/Wagstaff). | every objection answered or listed as a limitation |
| Oct 19 | Clean-checkout reproduction, anonymous supplement, checklist. | supplement verified |
| Oct 20 | Submit. Check the abstract-registration date now; it is usually days earlier. | submitted |

**Fallback, decided October 14.**
- If P1 or P2 fails but the v5 pool passes: the headline is the training-free universal pool, and UPT is an
  ablation.
- If both fail on the fresh networks: submit to TMLR with the honest mixed result rather than overclaim.

## Compute

- **Estimate:** about 180 fine-tunes plus 9 source trainings, roughly 12–15 GPU-hours on a T4 or L4.
- **CPU is not an option:** one CPU fine-tune takes 1–4 h, so this would be about 600 CPU-hours.
- **Modal cap:** the $20 cap is effectively used up. Raising it to about $45, or using Colab Pro, is the user's
  decision.

## Open audit item

`site_pool.py` replays the committed per-episode scores of `pfn_*` exactly. On NO2 it also replays `lct_*`
(< 1e-4). On Beijing PM2.5 and CO, however, `lct_s1_ft` differs by up to 0.018 and 0.0045 nats per episode. Those
cells were last rewritten in commit 9bb0481. Establish which code produced them before any LCT number from those two
targets is used.

## Paper outline

1. **Introduction.** In-context sensor fusion under missingness: extrapolation vs fit.
2. **Setup.** Episodes and masks; NLL.
3. **Lifted sites as the free construction.** U1, Props 2–3, exact init.
4. **The transformer as a site.** U2 / McConway; the Hölder bound; UPT.
5. **Experiments.**
   - synthetic oracle headroom and extrapolation;
   - five real networks;
   - ensemble and compute controls;
   - TabPFN v2;
   - ablations.
6. **Related work.**
7. **Limitations.**
   - Gaussian outputs;
   - CPU-scale backbone unless v4 integrates;
   - small number of real networks.

## Rules for every coding agent

1. **One protocol per batch,** committed before scoring. Report every endpoint, including failures.
2. **Never tune on test panels.** Used data stays descriptive.
3. **Same recipe for all learned models.** 3 seeds; final checkpoint; no selection.
4. **Replay before you report.** Every pooled number must come from replayed per-query predictions.
5. **No new mechanisms after October 13.**
