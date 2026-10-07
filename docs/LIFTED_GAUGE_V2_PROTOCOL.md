# One-factor orientation repair: bounded source gate (October 7, 2026)

Freeze this protocol, executable source and imported checkpoint hashes before
generating the source comparison. No checkpoint fitting, selection or retry.

## Repair

Use the same trained K=1 site network at D and -D and moment-match their equally
weighted Gaussian predictions. This is a deterministic, differentiable Gaussian
prediction with twice the inference passes and no added learned parameters.
It is invariant to the factor sign without selecting an orientation using labels.
This is not the exact mixture density and does not address K>1 rotations or a
degenerate leading eigenspace. Imported models and historical scores stay intact.

## Fixed engineering gate

- Checkpoints: `lift1_s1`, `lift1_s2`, `lift1_s3` from the imported run. Hashes in
  `configs/lifted_gauge_v2_source.json`. No retraining or seed selection.
- New source-family development panels: 128 tasks each, seeds 620281 (5 sensors)
  and 620282 (8 sensors), nonlinearity .4, 48 labeled support rows, 20% support
  missingness. Evaluate only zero or one missing query sensor. Width 8 is an
  engineering stress test, not a new independent confirmation claim.
- Compare original and repaired checkpoints on identical queries; report the
  untrained batched FA anchor descriptively. Use Gaussian NLL throughout.
- For EVERY width/seed: mean repaired-minus-original NLL <= .005 and the paired
  task-level normal 95% upper bound <= .01. Average masks/queries within each task.
  This is an engineering noninferiority tolerance, not proof of equality or
  downstream scientific superiority. Report all six comparisons, pass or fail.
- Mathematical checks: original closed-form initialization, factor-sign invariance,
  sensor permutation INCLUDING anchor refitting, finite gradients, masked-value
  exclusion, and explicit rejection of K>1. Tolerance 1e-4 except sign swapping,
  which must be exact. Also verify each saved trained checkpoint under sign swaps.
- Fixed one-thread CPU job, no cloud calls, no external data, no protected seeds,
  no old confirmation panels. Keep all per-task/mask scores and source hashes.
- If any gate fails, close this repair without tuning on these tasks. If it passes,
  adopt the versioned inference contract for the next prospectively frozen model
  comparison; the old confirmation scores remain evidence for version 1 only.

This gate introduces no architectural-novelty claim. Beijing eligibility, matched
trainable variable-width/static controls, independent training reproduction,
related-work clearance and manuscript readiness remain separate unresolved work.
