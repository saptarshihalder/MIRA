# Support-verified trained correction: October4 checkpoint

The370-parameter query operator passes the prespecified exploratory synthetic robustness gate on20 fresh latent seeds (two widths/four regimes). It gains sign-flip NLL .045838 [.034527,.057149] over frozen, .032354 [.020077,.044631] over equally verified logistic, and .018036 [.008201,.027872] over the no-query-head learner. Ignorable adaptation is rejected in all80 branch decisions. Nonlinear harm is .0000123 with upper95 .0000379, also below.001 even though this was not part of the primary null gate.

This is meaningful progress, not JMLR/NeurIPS readiness. Against the trained scalar-filter/query-head model the sign-flip gain .006202 has interval[-.002545,.014948]; the neural-specific gate requiring superiority to every comparator FAILS. Unverified full-support logistic is better on sign flip (.511657 vs.553499), but is harmful on ignorable shifts (.684695 vs frozen.640957). The contribution must be assessed as an adaptation/rejection tradeoff, not universal predictive dominance. All intervals are unadjusted exploratory95% t intervals over20 latent seeds, with widths averaged within seed. No new confirmation data were used.

Zero observed ignorable adaptations do not tightly bound rare future events. Treating each independent seed group as one Bernoulli trial, zero events among20 gives an exact one-sided95% upper acceptance-probability bound of1-.05^(1/20)=.1391 under the same IID generator. This concerns acceptance, not the probability or magnitude of harm. The nearly zero t interval on observed mean harm is therefore a development-screen statistic, not a rare-event safety certificate. Future evaluation must report this distinction.

| Regime | Frozen | Trained query operator | Same-rule logistic | No-query learner | Scalar/query learner |
|---|---:|---:|---:|---:|---:|
| Sign flip | .599337 | .553499 | .585853 | .571535 | .559700 |
| Nonlinear shift | .597942 | .597954 | .597942 | .598188 | .598434 |
| No shift | .598576 | .555911 | .587586 | .553294 | .568979 |
| Ignorable shift | .640957 | .640957 | .640801 | .640622 | .640957 |

## What was implemented

Three500-update models were trained with CUDA in a Modal A10040GB allocation. A source/target-conditioned spectral correction module is followed by a trained query head that uses baseline logit, correction magnitude/sign, curvature score and source/target residual difference. Two branches fit64 target support labels and verify against the disjoint64. The rule rejects an entire branch if its paired support-verification improvement is insufficient. Predictions average the two branches. No query labels enter candidate fitting, query-head context, verification or acceptance. This is not an ensemble of models retrained on query outcomes.

The model is trained using query BCE on both a smooth guard surrogate and the unguarded candidate; inference uses the prespecified hard1.645-SE rule. The surrogate discrepancy needs ablation. Field interactions are handcrafted. The inner descent bound applies to its convex fitting objective only. The verification threshold is a nominal heuristic; neither it nor the observed zero-harm panel constitutes100% distribution-free safety.

## Failure recovery and reproducibility

All three training checkpoints survived, but the remote evaluation aborted after132/160 files when the Platt-control L-BFGS-B solver returned ABNORMAL. The paid run remains FAILED and charged/reserved. A committed CPU recovery uses damped exact Newton for the identical convex Platt objective/ridge. Maximum gradient8.33e-11; maximum probability difference from original files5.96e-8. All132 original neural files replay from checkpoints with GPU/CPU maximum difference1.87e-9. No retraining, remote retry, seed replacement, threshold tuning or case exclusion occurred. All160cells were completed.

Audit verifies1,600 probability arrays,960 CPU output-array replays,960 split decisions and640 identity arrays. Per-update training traces and exact device-name output were not persisted before the abort; the frozen CUDA-only worker and checkpoint placement establish completion of500 updates after nonzero-gradient checks. This limitation is explicit. Future runners must persist each model's trace before evaluation.

## Strongest next steps

1. **Resolve the remaining model claim:** add a task-conditioned shared scalar-filter MLP with the same370parameters and a support-only risk head. Current scalar comparator has114parameters; current no-query comparator has257. These controls distinguish query information and mode-specific learning from capacity or general calibration. Predeclare their training labels, optimization budget and fresh development sample size. A supported narrow query-conditioning claim is preferable to an unsupported claim about every component.
2. **Fixed replication, not optional stopping:** freeze one independent development extension or later confirmation only after control selection is complete. Keep98000-98019 untouched. Do not repeatedly add seeds until intervals become positive. Quantify regret and adaptation coverage across support-label budgets, threshold/ridge choices and missingness shifts; include the zero-adaptation frontier.
3. **Native transfer:** apply the same frozen checkpoint/rule to defensible native missingness cohorts with patient/site/time boundaries and strong calibrated baselines. Existing older-model native results do not validate this candidate. Native panels already examined remain development. Claim predictive transfer, not acquisition-policy causality, when group/year changes are confounded.
4. **Learned representation only if justified:** a non-collapsing field-interaction map can be trained through the fixed inner convex solve, but its gain must exceed equally trained/random/fixed features and the present handcrafted basis. The JEPA auxiliary is inactive because matching its teacher did not create useful representations.
5. **Paper identity:** lead with the verified correction operator, precise label budget and adaptation/rejection tradeoff. Keep unsuccessful predecessors as supporting evidence. Related validation-guided learning, Meta-Weight-Net, MetaOptNet and learned optimizers prevent claiming these ingredients alone as novel. Direct novelty and native utility remain unresolved.

Budget16.45 provisions +3 protected reproduction leaves.55 under20. New GPU job has zero active tasks and no retry. Reconcile posted charges before another paid phase; free analysis can continue. The same manuscript source remains open; native compiler environment failure still prevents PDF verification.
