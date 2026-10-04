# Three-gate roadmap to a defensible trained-model paper

Planning date: October 4, 2026. Target: October 20 draft. This plan is not evidence of novelty, model superiority or acceptance. The RL correction module is complete as a negative result; all earlier used panels remain closed. The order is novelty, development advantage, then score-blind external validation. Each gate can fail honestly.

## 1. Novelty and problem suitability: October 4–6

Working candidate: a trainable predictive-state model for two-step feature acquisition. Encode the observed feature/value set, availability and budget. A learned stochastic transition predicts possible predictive states after a measurement; an acquisition policy evaluates an affordable second measurement before choosing the first. Actual observed measurements update the encoder/recurrent state. Train the encoder, transition and policy on source-only reveal trajectories. The hypothesis is that the model learns complementary measurements whose joint value is missed by one-step policies. This is not yet a novel architecture claim.

Specify the distinction in one sentence, expressed as a changed objective, representation, learning procedure or computational capability. Audit full methods and code of [EDDI](https://proceedings.mlr.press/v97/ma19c.html), [discriminative CMI acquisition](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html), [L2M 2026](https://proceedings.mlr.press/v306/kobayashi26a.html), [model-based active acquisition](https://arxiv.org/abs/2011.00825), their closest cited competitors and newer follow-ups. Existing abstract checks already show substantial overlap; full review is unfinished. Build a cited table of available observations, training supervision, planning horizon, latent transitions, inference cost and empirical evaluation. Novelty cannot be established by a new name, recursion, RL or a JEPA-style loss alone.

Pass only when a precise distinction survives this comparison and predicts an ablation effect. If already covered, materially revise once before fitting or stop this candidate. Do not spend on another architecture solely because it sounds more complex.

Audit data in parallel: public license, real feature meanings, identities/duplicates, feature availability at prediction time, label proxies, revealable values and acquisition costs. Naturally missing entries have no known reveal value. Artificially hiding recorded values is simulated acquisition on real records, not validated clinical acquisition. Use documented costs or explicitly simulated normalized unit costs. Do not select complete cases without reporting that restriction and its bias. Freeze a benchmark panel selected by eligibility rather than model scores. Reserve distinct dataset families for external evaluation; related endpoints/releases share one evaluation unit. Exact dataset IDs must be committed before fitting; no panel is claimed already established.

Deliverables: novelty matrix, one-sentence contribution, audited data manifest and frozen partitions. Lack of verified revealable data is a stop condition.

## 2. Demonstrate useful superiority: October 7–12

First build a small CPU implementation check with synergistic, additive, irrelevant-feature and excessive-cost tasks. This verifies behavior only; toy success is not a publication result. Test a source-trained two-step baseline using only observations it has acquired to establish achievable development headroom. A label-aware oracle is not evidence that an admissible policy can attain the improvement. If two-step behavior has no practical development value, stop rather than scale blindly.

Train one main model with three initializations and fixed convergence/checkpoint rules selected using training/development data only. Expose no unrevealed query values or labels to planning. Imagined transitions must integrate over learned possibilities, not peek at the recorded row. A full-information teacher can use additional source features, but matched ablations must receive the same supervision. All target adaptation is restricted to the same predefined support-label allowance.

Required baselines: random and static acquisition, calibrated static prediction, a strong discriminative CMI method, the closest L2M/model-based competitor, and a one-step version with the same architecture and labels. Verify faithful baseline implementations on their own documented checks where feasible. A missing closest competitor is an unresolved limitation, not permission for a state-of-the-art claim. Account for pretrained-data overlap if using existing foundation models.

Required ablations: remove two-step planning; remove learned transition; remove order-consistency loss; replace sampled RL by full-information training where rewards are available. Keep training labels, teachers, hyperparameter-search budget and stopping rules matched. Compare against an equal-inference-compute alternative, since extra rollouts can explain gains. Report gradient calls, wall time and prediction cost, not just parameter counts. Fit-label inequalities from the old RL pilot must not recur silently.

Primary metric: dataset-level area under NLL versus normalized acquisition cost, lower is better, on a fixed budget grid. Compare each method on identical rows, reveals and costs. Average seeds/splits within each family before cross-family inference. Proposed practical threshold: at least 2% mean relative area reduction over the strongest baseline selected using development data. Choose and commit the final threshold before model outcome scores; it is a research decision, not a venue requirement. Secondary cost-at-matched-NLL, calibration and runtime must not replace a failed primary endpoint.

Development gate: positive primary effect across at least three prespecified development families, useful aggregate gain, repeatability across three seeds, and an ablation showing that the claimed component adds value at matched resources. Predeclare a harm margin; a suggested threshold is at most 1% relative area deterioration on designated additive/no-synergy controls. Report violations, never claim universal safety. Freeze one candidate, all baselines, seeds, thresholds and evaluation code after passage. Failure closes the batch without extending seeds until significant.

Deliverables: checkpoints, complete trajectories, learning curves, baseline reproduction notes, ablation table, measured cost and signed/hash-pinned evaluation protocol.

## 3. External validation: October 13–16

Reserve a proposed minimum of eight distinct external dataset families before training. Final sample size depends on a power/precision calculation using development variability and the fixed meaningful-effect threshold. Eight is not an automatic guarantee of adequate power. If resources cannot support the required sample, narrow the claim and report insufficient precision. Existing UCI, BRFSS and road panels do not become fresh evidence by changing seeds or group names.

Freeze model architecture, preprocessing, checkpoints and all allowed support adaptation. Keep external query labels out of the training/selection process; a separate scoring stage may read them only after prediction artifacts and protocol hashes are fixed. Public data are not literally secret, so describe this as score-blind evaluation with documented access boundaries. Evaluate once. Record identities, dataset versions, masks, costs, random seeds and failed runs.

Primary external gate: prespecified practical effect versus the development-selected comparator, a paired 95% interval over independent dataset families excluding zero, and gains on a majority of families without an undisclosed catastrophic subgroup loss. Freeze the interval construction and assumptions before scoring; do not treat seeds or multiple endpoints as independent datasets. Report every family and sensitivity analysis, and adjust secondary confirmatory comparisons for multiplicity. A positive mean with an interval crossing zero is inconclusive, not a win. Do not expand the panel after seeing the result.

If the held-out score fails, preserve the checkpoint/results and narrow or reject the claim. Any later redesign requires a newly reserved evaluation panel. Confirmation seeds 98000–98019 remain untouched and cannot automatically confirm a different acquisition problem.

## Completion, budget and hard decision rules

October 17–19: reproduce from a clean checkout; verify every headline number against saved predictions; test inference for hidden-value/label access; publish all negative controls, failures, environments and scripts; verify the same manuscript source and PDF layout. October 20 is a draft/readiness review, not a promise that empirical gates will pass. If earlier gates slip, report the incomplete stage rather than compressing validation.

Current Modal cap: $20. Conservative provisions $16.45 plus protected reproduction $3 leave $.55. Reconcile invoices before committing to new GPU training. Plan compute from a bounded timing pilot, multiply by the full model/seed/baseline matrix and include setup/failures; reserve the worst case before dispatch. If it does not fit, reduce model/runtime scope while preserving controls or request a specific evidenced budget increase. Do not silently consume the reproduction reserve or assume Colab Pro access; it remains unverified.

Token policy: reuse saved artifacts, one concise record per gate, bounded outputs, no duplicate agent sweeps. No additional correction/RL variants on the closed panels. The primary paper should have one clear trained-model contribution; earlier investigations become relevant supporting or negative evidence rather than a chronology of failed architectures.

We may describe an internally complete research package when the module/reproduction work is done. We may describe a submission-ready draft only after novelty, meaningful model advantage, external evaluation, closest-baseline comparisons and manuscript verification have each passed. None of those outcomes guarantees acceptance.
