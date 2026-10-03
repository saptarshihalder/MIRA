# Support-conditioned acquisition-policy correction: first development protocol

Authorized October 3, 2026 by the user's instruction to continue the proposed idea. The failed compiler remains fixed history. New code/data/checkpoints use isolated paths. No learned-method benefit or new novelty is claimed before direct evidence.

## Model and predictive objective

A shared DeepSets row encoder pools labeled source and target support. Concatenated learned pooled features, observed feature means, centered residual moments and label/residual means condition a neural hypernetwork producing coefficients for a query probability correction. Query features contain one fully observed outcome-related covariate, four partly observed auxiliary covariates, fifteen declared mask parities and a frozen source-logistic logit. Neural parameters are meta-trained directly on query BCE; privileged mechanism parameters and oracle targets are never model inputs or training targets. The final fixed epoch is saved. Validation curves are diagnostic only.

The first backbone is a CPU logistic proxy fit to source U only. Its coefficients remain frozen during meta-learning and evaluation. A passing result would justify a separate frozen TFM/GPU protocol; it would not establish TFM efficacy.

## Distribution and boundaries

Each independent outcome world has fixed beta/intercept across paired observation policies. Intercepts vary between worlds. Label-dependent mask policies span all fifteen nonzero parities, both signal signs, null signal, rates, feature-dependent collection and correlated blocks. Source/target policies are sampled independently during shifted training. The no-shift ablation receives the same latent rows, labels and episode/update counts with target policies equal to the source policy. No prediction-invariance penalty is imposed: informative policy changes alter the optimal conditional posterior.

Labeled target support is an explicit deployment requirement (64 labels, plus 128 source labels). Source/target support/query rows are disjoint; paired policy cells share latent target rows within a world and are not independent uncertainty units. Predict accepts observed source/target support and observed queries only. Missing hidden values are ignored. Each query is processed independently. No evaluation query labels, oracle or true policy enter preprocessing, fitting, selection or inference. Meta-training query labels are permitted only in the training objective. The scalar always-observed covariate and exact four-bit parity family are strong structural assumptions.

## Controls, freeze and decision

Compare an ordinary query-only residual MLP with similar parameter count; an identical context architecture trained without policy shifts; frozen predictor; target intercept calibration; target support-CV ridge corrections with mask bits and with full parity; pooled source/target-domain interaction ridge; and strongest centered parity moment with fitted scale/intercept. Every model may use all observed source labels and the same target labels; the ordinary model intentionally lacks target-context inputs. All tuning grids, seeds, labels, updates and checkpoint rules are fixed in configs/policy_adapter_v1.json. Model parameter counts and training traces must be reported because approximate capacity/update matching does not prove optimization fairness.

Forty-eight disjoint development outcome worlds have five policy cells each: unchanged, reversal, novel extreme rate, stronger correlated/feature-dependent acquisition, and null. Rate/block shifts contain parameter combinations beyond meta-training. Reversal with a null source remains null and will be disclosed. Primary expected NLL is computed from exact generative posterior, empirical NLL secondary. Average cells within worlds before 95% paired task intervals. These are nominal development intervals, without a confirmatory multiplicity claim.

Advance only with mean gain >=.01 over ordinary adapter, positive interval lower bounds over every named operational comparator/ablation, and null/unchanged mean harm <=.01 over frozen. No method or safety claim if this screen fails. Paid compute remains zero for this phase; project reservations stay $10.55, total cap $26 and reproduction reserve $3. No retries or confirmation-seed use. A future passing model needs fresh native-data/time/site evidence, contemporary TFMs, targeted prior-art review and clean reproduction.

Known adjacent work includes [Domain Adaptation under Missingness Shift](https://proceedings.mlr.press/v206/zhou23b.html), [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484), and [TFM-Retouche](https://arxiv.org/abs/2605.06047). Neither residual correction nor missingness adaptation alone establishes novelty.
