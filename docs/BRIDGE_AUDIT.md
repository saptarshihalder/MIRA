# Independent MIRA-Bridge audit — October 3, 2026

**Recommendation:** test one frozen successor, not continuous architecture changes against the same scores. Bridge is a plausible small-support adaptation model; novelty and usefulness remain hypotheses. The completed A100 study found no neural-specific advantage: road scratch NLL .603685 versus support Platt .603197 and global simplex .603348; health gains were tiny or negative.

## Mechanism and limitations

Proposed correction is `logit(f_source(q)) + h(q,S_target) - h(q,S_source)`, with a shared query-conditioned set encoder. Observed-value/mask interactions and local support residuals can represent corrections that global calibration cannot. Variable feature width removes the earlier four-mask restriction, provided feature identity, units and categorical semantics remain explicit. Variable width alone does not establish transfer across unrelated schemas.

Identity, reversal and additive path consistency are algebraic consequences of subtracting potentials at the **same observed query**, independent of training. They neither demonstrate correct predictive transport nor identify acquisition mechanisms. They need not hold when the query's observed values/mask change between environments. Clip/bound each potential before subtraction if these properties are intended; nonlinear clipping or a pair-specific gate applied afterward can destroy additive consistency.

The main risks are statistical. Small labeled supports may not reveal high-order conditional changes, especially on rare mask patterns. Attention may become an ordinary local supervised predictor. Source subtraction can remove useful common calibration, and arbitrary query-dependent offsets shared by all environments cancel, making absolute potentials unidentifiable. Supports must use residuals from the same frozen backbone; using different fitted predictors changes the anchor and interpretation. No observational construction guarantees missing-not-at-random identification, improvement, clinical safety or venue acceptance. Native year/region panels mix measurement, population, reporting and outcome changes; they cannot establish causal policy transport.

## Required comparators and ablations

Match source labels, separate source/target support labels, meta-query labels, pretraining episodes, validation search budget and inference information. Include frozen prediction, global/support intercept and Platt, convex stacking, converged conditional linear correction, a capacity-matched query-conditioned MLP/set-attention residual, and a direct contextual predictor. Compare target-only potential, concatenated source/target contextual correction, no-query potential, no-mask model, and ordinary unconstrained two-support correction. Compare native-only and equally pretrained variants. NeuMiss/NeuMISE are relevant architecture baselines; DAMS is a setting/assumption reference, not automatically a like-for-like labeled-support competitor.

## Preregistered gate

Freeze implementation, optimization/search counts, episode distributions, seed namespaces and stopping rule **before scores**. Use controlled latent-outcome tasks with feature-dependent nonlinear observation mechanisms, plus no-shift tasks with independently sampled supports, prevalence-only shifts and shifts explainable by linear calibration. Include ignorable shifts: they may leave the Bayes predictor unchanged, so a nonzero correction is not inherently desirable. Privileged latent mechanisms may diagnose synthetic performance but must not enter inference. Split source fit, support, meta-training, development and evaluation by independent underlying rows/tasks; query labels may train the meta-objective but never condition their predictions.

Require task-paired mean NLL improvement at least **.003** over the strongest simple calibration/stacking control selected on separate validation, with a positive lower 95% task-level confidence bound; also beat the matched contextual learner. Preregister an upper 95% null-harm bound of .001 NLL. No threshold relaxation after outcomes. Reuse native panels only as development and report their failures. Fresh confirmation seeds 98000–98019 stay untouched until the frozen development gate passes. Failed gate means retain the negative result and stop this candidate; architectural revisions require a new version and fresh gate.

Primary references: [DAMS](https://proceedings.mlr.press/v206/zhou23b.html), [missingness-shift robustness](https://arxiv.org/abs/2406.16484), [NeuMiss](https://arxiv.org/abs/2007.01627). These support ingredients and limitations, not a first-of-its-kind claim.

## Pre-freeze runner audit addendum

Independent source inspection, no training or paid calls. Initial runner bugs were reported before scientific dispatch: model logits treated as probabilities, a mean rather than upper-confidence null gate, ignored validation seeds, and missing trained ablations. Live edits now address these. Recheck the committed snapshot, since line numbers below refer to the pre-freeze inspection.

**Remaining blocker:** `run.py:90` fits every frozen source under label-dependent missingness; `observe:55` makes only the `ignorable_shift` target label-independent. This is MNAR-to-MAR adaptation, not an ignorable-to-ignorable null. For this null, generate and fit its source under an ignorable policy too; otherwise rename it as a shifted stress test and add a true MAR-to-MAR null. Under suitable ignorable shifts the Bayes predictor can remain unchanged, while MNAR-to-MAR need not.

**Configuration:** `configs/bridge_v1.json:18–24` uses `width`, `heads`, `useful_effect_nll`, `max_null_harm_nll`; runner constructors/gate use `model_width`, `model_heads`, `minimum_gain`, `maximum_null_harm`. Current defaults happen to coincide, but the frozen config does not control these quantities. Use canonical keys and reject unknown scientific settings before hashing.

**Comparability/numerics:** `model.py:110/115/118` permits a Bridge correction in [-4,4] but target-only in [-2,2]. Give the target-only ablation the same correction bound to isolate subtraction. Train using binary cross-entropy with logits; sigmoid-clamp-BCE (`run.py:194–195`) introduces dead gradients outside its clamp. Recorded logistic iteration caps do not establish convergence; fail or disclose any unconverged source/support controls rather than attributing their loss to architecture.
