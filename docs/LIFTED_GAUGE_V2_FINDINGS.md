# One-factor sign repair passes its source gate — October 7, 2026

Version 2 wraps each fixed K=1 checkpoint with `SignAveragedCavity`: evaluate
both equivalent factor signs and moment-match their Gaussian predictions. The
choice is label independent, differentiable, and uses no extra learned parameters.
It costs two forward passes. It returns a Gaussian, not the exact mixture density.

Protocol, code and original checkpoint hashes were committed and pushed at
`b8c183aa68b1a8fa800c0b0b25bfa01251156e0f` before source comparison data were
generated. There was one run, no training, no retries and no external scoring.

| Sensors | Original training seed | Repaired minus original NLL | Paired 95% upper bound | Gate |
|---|---:|---:|---:|---|
| 5 | 1 | -.000665 | .000806 | PASS |
| 5 | 2 | -.000637 | .000976 | PASS |
| 5 | 3 | -.000806 | .000501 | PASS |
| 8 | 1 | -.000851 | .000820 | PASS |
| 8 | 2 | -.000904 | .001068 | PASS |
| 8 | 3 | -.001267 | .000397 | PASS |

Each row uses 128 source-family tasks and zero/one missing query sensor. The fixed
tolerances were mean deterioration <= .005 and upper bound <= .01 for EVERY row.
Every prediction was exactly invariant under reversing D. The intervals cross zero:
this is an engineering noninferiority check, not evidence of improved efficacy.
All six means happen to improve slightly. Seed rows share task panels and are not
independent dataset replications.

Nine unit tests pass, including initialization at the closed-form anchor, sign
invariance, sensor permutation with anchor REFITTING, masked-value exclusion and
finite gradients. A separate aggregation check replays 11,520 saved score cells
and all six decisions (maximum discrepancy 5.56e-9); frozen file/checkpoint hashes
match. Reports: `artifacts/reports/lifted_gauge_v2_source/`.

Use `SignAveragedCavity(base)` with an imported K=1 `LiftedCavity` loaded from its
original state dictionary. Model identity is the base-checkpoint hash plus
`gauge.py` hash, both in `configs/lifted_gauge_v2_source.json`. Future fitting must
use this same wrapper during training and inference. Existing training/evaluation
commands still select the original model unless explicitly wrapped; do not silently
mix version-1 and version-2 scores.

Scope: this handles the two signs of ONE factor. K>1 is explicitly rejected;
arbitrary rotations and degenerate leading eigenspaces are not solved. Historical
confirmation remains version-1 evidence and is not rescored or renewed. This fix
does not establish novel architecture, real-data performance or venue readiness.

Next bounded prerequisite: Beijing raw-data eligibility and label boundaries, then
a prospective protocol with static and trainable variable-width controls, matched
source learning and three seeds. Independent training reproduction and prior-art
clearance remain outstanding. No Beijing/HAR labels or protected seeds were read.
CPU gate runtime: 38.35 seconds; cloud calls/cost: zero. Manuscript unchanged.

To reproduce, use the frozen protocol checkout; run the nine targeted tests and
`python experiments/lifted_cavity/gauge_source_gate.py`. Existing output is refused;
use a clean checkout instead of deleting or overwriting saved evidence.
