# What the new trained result establishes

The257-parameter source/target error-mode filter completed500 meta-training updates on an actual Modal A100, with a500-update scalar-gate control and an untrained full-step control. Eight fresh latent seeds, two widths, four regimes. All384 score arrays and192 CPU checkpoint replays pass audit. The largest observed increase in the convex support objective is nonpositive within numerical precision. The mathematical descent statement applies to that objective only.

| Regime | Frozen | Support logistic | Trained scalar | Trained mode filter |
|---|---:|---:|---:|---:|
| Sign flip | .595114 | .499888 | .521327 | .502013 |
| Nonlinear shift | .586971 | .607071 | .604623 | .596412 |
| No shift | .587613 | .499912 | .515568 | .491946 |
| Ignorable shift | .644914 | .680778 | .666993 | .657967 |

Lower NLL is better. Positive trained selectivity: sign-flip gain versus trained scalar .019314 [.004884,.033744]; no-shift gain .023622 [.010601,.036643]. Nonlinear gain versus support logistic .010659 [.004019,.017299]. These are unadjusted exploratory95% t intervals over eight latent seeds, with widths averaged within seed. They are not confirmation or simultaneous guarantees. Against frozen, nonlinear harm .009441 and ignorable harm .013053 remain; ignorable harm interval [.006387,.019719] rules out a no-harm reading for this panel. The development gate fails. Support logistic still slightly wins on sign flip. Do not hide these controls or call this venue-ready.

This is a stronger measurable trained component than the pooled JEPA prototype. The JEPA diagnosis also rejects simply solving its head harder: sign-flip NLL worsens from .616819 to .637938; random embeddings with the same solved head reach .630923. This used-panel diagnosis is not fresh evidence. Representation collapse cannot be solved by confidence, recursive depth or norm inflation alone.

## A focused next architecture

**Cross-fitted label-response operator**: retain the learned spectral filter and add a learned field-interaction representation, held fixed during each inner logistic solve. Divide target support into two disjoint parts. On one part compute candidate error-mode updates; use the other part's gradient agreement and uncertainty as gate context, then reverse roles and average predictions. This exposes agreement on unseen labeled support to the model instead of reusing the fitting residual as evidence of generalization. All gates remain in[0,1], preserving the support-objective descent bound for each fit. A separate exact frozen action is necessary. This is a candidate design, not a demonstrated guarantee of test risk or novelty.

Learn the feature map and shared gate jointly through the unrolled solver using outer query-label supervision on training worlds only. Add label-discrimination diagnostics and, if justified by a committed ablation, supervised contrastive loss within each task. A context-only embedding or a constant embedding must fail the representation gate. Do not reintroduce the EMA matching shortcut until it beats supervised-only and fixed/random-feature controls. The feature map can be nonlinear while the inner head remains convex in its own weights, so the descent bound survives for a fixed episode feature map.

Before paid expansion: a CPU pilot must show that held-out gradient agreement predicts harmful modes on USED development worlds, without reading evaluation query labels as inputs. Freeze label allocations and compute-matched controls, including cross-fitted scalar/fixed gates, fully converged regularized logistic, v2 learned regularizer, supervised-only learned embeddings, and source-ablated/shuffled-label controls. This hypothesis may fail because64 labels per fitting half increase variance. No threshold may be chosen from future query outcomes. New development and later native-domain/unused confirmation remain separate. Preserve a finite search budget and negative results.

## Paper backing and novelty boundary

- [OptNet](https://proceedings.mlr.press/v70/amos17a.html) and [MetaOptNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html) establish learning through optimization; this is not itself new.
- [Learned optimizer training pathologies](https://proceedings.mlr.press/v97/metz19a.html) motivate careful horizon/generalization controls.
- [PAC-Bayesian learning of optimization algorithms](https://proceedings.mlr.press/v206/sucker23a.html) already studies learned optimization with guarantees. Do not claim that a descent-constrained learned optimizer is a new general principle.
- [Supervised contrastive learning](https://arxiv.org/abs/2004.11362) is a possible label-discriminative training ingredient, not an implemented or novel feature of this pilot.

The potential contribution is a demonstrated transferable missingness-adaptation operator with label-budget-aware mode selection and useful nonlinear learned representations. It requires empirical separation from the existing methods above. No100% distribution-shift safety or acceptance claim is justified.

Compute:15.65 conservative provisions plus3 reproduction reserve leaves1.35 under20. Current phase retained.80, no retry and zero active tasks after completion. Further GPU work needs diagnostic justification and a new frozen protocol. Manuscript, checkpoints, full predictions and audits retain all failures.
