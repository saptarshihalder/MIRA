# Recursive label-memory JEPA prototype (v1)

Status: prospective engineering pilot, not a scientific efficacy gate. The previous Bridge v1/v2 failures remain unchanged. This prototype responds to the user's request for recursive backpropagation and learned label-pattern embeddings. New code must not modify frozen predecessors. Architecture novelty is unestablished.

## Mechanism and information boundary

A shared field encoder uses observed values, missingness indicators and source/target support residual associations to form nonlinear row embeddings. Support labels give the fields task semantics; field identities are not treated as universally transferable across datasets. Three applications of the SAME learned refinement block update the embeddings from source/target residual and class memories. A residual linear head receives differentiable, support-only gradient updates. Outer training-query cross entropy backpropagates through all three steps into the shared encoder/refiner; step sizes are fixed bounded constants. This is finite unrolling, not an equilibrium solver or unlimited recursion.

In schematic form, with observed query embedding h, support-derived memory M and frozen offset z:

    h[k+1] = h[k] + bounded_step * bounded_refiner(h[k], M[k])
    w[k+1] = w[k] - .25*tanh(gradient_w regularized_support_BCE)
    query_logit = z_query + 2*tanh((h_query[K] @ w[K])/2)

Exact implementation details and all constants are authoritative in the hashed model source. Support memories use CURRENT corrected support predictions. Supports are refined first and their refreshed memories condition query refinement. The head includes its declared bias coordinate, bounded correction Jacobian and ridge penalty. The bounded updates and bounded final correction prevent unrestricted correction magnitude; they do not prove contraction, loss descent, calibration, monotonic improvement, or no harm. An explicit `force_frozen` intervention returns the frozen prediction exactly. It is not yet an independently validated selection rule.

Inference accepts only observed query values/masks/frozen logits and labeled source/target supports (the predecessor's eleven-input contract). It never accepts query labels, generator parameters, oracle probabilities or initially unobserved values. Training query labels enter the explicitly supervised outer loss only. Class/residual memories come from supports, not guessed query labels. Recurrence is adaptation to a fixed labeled support set; it is not autonomous online learning over calendar time.

## JEPA-inspired auxiliary

The student predicts a stop-gradient EMA teacher representation of the SAME row under nested additional missingness views. The teacher's richer view contains only values originally observed in that row. Neither view reconstructs true originally missing values. No QUERY label enters this representation target. Both encoders are conditioned on labeled support statistics, so this is a support-conditioned JEPA-inspired auxiliary, not fully self-supervised training. Query labels supervise the prediction loss explicitly. The EMA teacher has no optimizer gradients and updates only after an optimizer step, never inside forward or inference. Variance and off-diagonal covariance penalties on student embeddings monitor and discourage collapse; measured collapse statistics must be retained, including failure. The implemented variance floor is1 on the unbounded, LayerNorm row encoder; this is a regularization target, not a guaranteed property of bounded recurrent latents.

This borrows representation prediction from [I-JEPA](https://arxiv.org/abs/2301.08243), explicit anti-collapse regularization from [VICReg](https://arxiv.org/abs/2105.04906), and differentiation through support adaptation from [MAML](https://arxiv.org/abs/1703.03400). These papers motivate ingredients; none establishes this tabular combination's novelty or efficacy. JEPA is not a guarantee that hidden query labels can be recovered.

## Bounded GPU engineering pilot

One Modal A100-40GB call; 600-second worker/60-second startup/540-second child limits, one container, zero retries, no idle service/schedule. Reserve $0.80 before deployment and preserve $3 reproduction funds under the $20 total cap. Resource estimates are not invoices; retain the entire provision until reconciled. Closed predecessor provisions total $14.05, so this call leaves $2.15 unallocated.

Frozen generator: experiments/bridge_v1/run.py. Source-fit labels2048; distinct labeled source support128 and target support128;64 training query rows/128 evaluation query rows. Train seeds440001-440004; widths6/10; all four existing regimes in fixed order. Evaluation seeds450001-450002; widths6/10. Seeds445001-445002 are named unused validation placeholders, not consulted. No selection on evaluation scores. Worlds/regimes share rows as declared by the generator; they are not independent datasets. Confirmation98000-98019 stays unused.

Train three models on identical data, initialization seed, optimizer and64 updates/batch4/lr.001: three-step full model, three-step supervised-only model without JEPA/variance/covariance auxiliaries, and one-step model with the same JEPA auxiliaries. All have identical trainable architecture capacity; the auxiliary-only predictor is not effectively trained in the supervised-only model. Recurrent operations and FLOPs differ. This small pilot checks functioning and gradients, not architectural attribution, fair computational matching, significance, robustness or convergence. Save every model and all diagnostic predictions, including regressions. No best-epoch selection or new-candidate retries.

Required checks: no query-label input; missing-value sentinel invariance; support-label dependence; actual nonzero gradient paths through every step; teacher stop-gradient/explicit EMA; exact frozen intervention; finite losses/coefficients; checkpoint restoration and CPU/GPU prediction replay; independently recomputed NLL; provider resource report and zero active tasks afterward. If implementation checks fail, repair before freezing/paid dispatch. Scientific failures are retained, not silently refrozen.

## Subsequent scientific gate (not performed by this pilot)

1. Diagnose whether label memories and nonlinear embeddings improve nonlinear shifts without harming the ignorable null. Validate fallback with a fixed disjoint support validation allocation; do not use evaluation query labels for accepting adaptation.
2. Freeze fresh development seeds, a finite search budget, label budgets, a per-regime gate and stopping rule. Compare equally trained feature maps, randomized/fixed nonlinear bases, optimized support logistic/Platt, v2 solver, target-only/no-source, no-recursion, no-JEPA, shuffled support-label and capacity/compute-matched controls. All auxiliary views and additional labels count toward budgets.
3. Require useful conditional benefits and predeclared null protection before native external evaluation. Evaluate defensible acquisition/group/time partitions and strong contemporary tabular and missing-data baselines. Existing panels remain development. Teacher or embedding loss reduction is not downstream utility.
4. Only after a frozen development gate passes, evaluate untouched confirmation and document dataset-level uncertainty, calibration, failure modes and clean reproduction. Until then no robust-model, novel-method or venue-readiness claim.
