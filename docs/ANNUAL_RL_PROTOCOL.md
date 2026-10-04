# Adaptive RL nonlinear-correction pilot

User-authorized October 4, 2026. This explicitly reopens a distinct RL comparison after the closed annual tree screen. No previous model or result is relabeled. Protocol/code are committed before preparing support-fitted anchors; inputs are frozen before policy fitting and query scoring. Previous cohorts remain closed.

## Boundaries and controls

For each of BRFSS diabetes, asthma and roads, choose ascending eligible annual groups 6 through 10 (zero-based slice 5:10) among 2025 group codes modulo five below three, with at least 1,536 rows. Choose no group by labels. Order rows by SHA256 annual-rl-v1:2025:row_id. First 512 support, next 1,024 queries; 384 support fit and 128 support selection. Check all selected groups/IDs against the prior annual screen and recorded cohort audit. Full raw-data/label preprocessing was previously exposed, so these are unscored development cohorts, not confirmation. BRFSS endpoints overlap and are dependent.

Historical 2024 CUDA backbones are fixed. Fit an anchor Platt correction with ridge .001 on 384 fit labels. Encode source-only categorical vocabularies as 34-way one-hot (32 possible known codes, unknown, missing); standardize numeric features with source moments, clip to [-10,10], append missing flags and frozen/anchor logits. No query statistics or labels enter fits or selection.

Run the prior annual baseline recipe unchanged on these new groups: frozen, 512-label three-fold/four-choice Platt and additive controls, and calibrated tree search. RL uses the same 512 total labels but only 384 for gradients and 128 for selection. Baselines have more fitted labels and different search/compute costs; disclose rather than call this exact compute matching. The three neural learners below are exactly matched for architecture, support split, initialization, batches and optimizer-update count.

## Trainable contextual-bandit policy

One action per row changes its calibrated logit by {-1,-.5,-.25,0,.25,.5,1}. This is one-step RL, not a multi-step environment. Nonlinear actor: feature input -> 16 tanh units -> seven softmax actions. Initialize final weights to zero and zero-action bias to 2; retain all learned parameters. For each of seeds 991001/991002/991003 train 300 AdamW updates, batch size 128 drawn only from 384 fit rows, learning rate .001, weight decay .0001, clip gradient norm 1, one CPU thread. Fixed update count; record train loss/entropy every 50 steps, not a convergence claim.

RL uses REINFORCE with sampled actions and the zero-action reward as an action-independent baseline: reward is reduction in label NLL minus .01 times squared correction. Entropy bonus .001. Compare an exact policy-gradient control summing the same rewards over all actions (full information from the same support labels), and a supervised mixture actor minimizing NLL of its predictive mixture with the same correction penalty/entropy bonus. Identical parameter count/initialization for all three; no learned critic or privileged pretraining. Full support labels permit exact reward computation, so RL is an empirical alternative, not a necessary formulation.

At inference output the deterministic policy-weighted mixture of action probabilities. On the 128 selection rows choose shrinkage toward the anchor from {0,.25,.5,1}, breaking ties toward less correction. This selection is available equally to all three models. Save raw and selected predictions, action probabilities, weights and traces. Selection may reject all corrections; report that and unguarded results. No query feedback or adaptation. Frozen/anchor comparisons do not establish risk safety.

## Finite evaluation

Fifteen groups x three seeds x three learners = 135 checkpoints. Aggregate training seeds within group; groups do not become independent datasets. Report all per-seed and per-group NLL/Brier. Descriptive RL gate per task: >=.003 mean NLL gain over EACH frozen, selected Platt, additive, calibrated tree, exact policy gradient and supervised actor, and positive in >=4/5 groups against each. General successor gate requires roads and at least one BRFSS endpoint. No significance or 100% safety claim. Close this run regardless of result; no threshold, group or seed extension until positive.

The run is CPU-only, zero new cloud reservation; $16.45 provisions plus $3 protected reproduction leave $.55 under the $20 cap. Bound neural fitting to 600 seconds between models (finish the current model before stopping), no automatic retries. Preserve partial failures, checkpoints, input hashes and query-label separation. Audit raw row/value joins, source/target and previous-panel separation, all saved predictions/weights and arithmetic. Existing synthetic confirmation 98000–98019 remains untouched.

References: [contextual-bandit formulation](https://proceedings.mlr.press/v15/beygelzimer11a.html) and [policy-gradient variance research](https://proceedings.mlr.press/v162/gargiani22a.html). These motivate the comparison; their theoretical guarantees do not apply to this neural pilot. RL or this architecture is not claimed novel.
