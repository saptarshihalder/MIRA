# Lifted cavity import audit — October 6, 2026

Decision: use the lifted model as the main candidate through October 10. Branch
`lifted-cavity` at `54bd6b169e0552359aded3c319a54728fb9ae4fb` was fast-forwarded
into local `codex/mira-research`. Circuit and quantum-inspired attention are deferred.
This decision does not establish architectural novelty or venue readiness.

## Verified

- All six imported unit tests pass locally (PyTorch 2.8.0+cpu).
- The official UCI archive was downloaded; its CSV matches the development mirror
  byte for byte: SHA256 `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.
  Provenance is in `artifacts/reports/lifted_cavity_import_audit/provenance.json`.
- Nine checkpoint hashes match the imported confirmation manifest.
- All three synthetic confirmation panels and the NO2 test panel were regenerated.
  Fifteen synthetic checkpoint/panel combinations and the fine-tuned NO2 checkpoint
  were replayed at two missing sensors. No training or model selection occurred.
- All five endpoint pass decisions reproduce. Replayed gains E1/E2/E3/E4/R1 are
  .074020/.015877/.114728/.028105/.111602; maximum interval-endpoint discrepancy
  is .00002301. The original normal-approximation intervals are reproduced, not
  newly certified for temporal independence or multiple testing.
- The supplied figure shows CO development results. NO2 is the separately declared
  confirmation target, on the same device; these are not two independent datasets.

## Reproducibility issue: factor orientation

The full per-cell replay gate **fails**: C2 errors reach .012804 nats, despite its
aggregate endpoint reproducing. All other audited panels differ by less than
.000023 nats. The factor loading D and -D encode the same covariance, but the
learned site network is not constrained to respect this equivalent representation.
Eigenvector signs can differ across numerical implementations.

A diagnostic on C2 seed 1 found one task exceeding .001 error. Flipping the factor
sign for diagnostic comparison explains that task: the minimum error across the
two equivalent orientations is below .00000281 for every task. Across all tasks,
changing the sign can alter cell NLL by .08460. This is evidence of sign sensitivity,
not an allowable way to select signs using evaluation losses. See
`gauge_diagnostic.json` and `experiments/lifted_cavity/audit_gauge.py`.

Preserve imported checkpoints and scores. Before a successor real-data gate, fix
the factor-orientation contract without evaluation labels (for example, a
deterministic, sensor-permutation-compatible anchor convention shared by fitting
and inference, or an explicitly sign-invariant architecture). Treat altered
predictions as a new version and validate on source development; do not relabel
the existing confirmation panels as fresh. For K>1, sign handling alone does not
resolve general rotations or degeneracy. Existing unit tests permute an already
computed anchor and do not establish end-to-end invariance after refitting it.

## Claims and next work

The original finite gates remain valid failures of their tested configurations.
They did not prove zero oracle headroom. The new architecture and training recipe
supply better evidence, but their success does not invalidate old observations.
The scalar-site sign argument assumes fixed sensor-local contributions; it is not
an impossibility proof for arbitrary mask- or context-conditioned scalar sites.

Cavity conditioning adds no measurable gain on Air Quality versus lifted static
sites. Eight-sensor operation establishes compatibility, not superiority over a
trainable variable-width control. The next protocol must include that control,
lifted static sites, and strong support-only EM/FA references. NO2 currently has
one fine-tuning run; CO/NO2 weeks may be temporally dependent. Support and query
hours are sampled within each week, so this is within-week calibration, not a
causal forecasting test. Fine-tuning filters timestamps strictly before October 1;
the source-report episode grouping instead uses window start dates and can straddle
that cutoff. Test windows start October 6 and do not overlap the fitting interval.

The protocol is an ancestor of the results commit and its original version contains
no results. Checkpoint hashes are first committed with results, not in the freeze;
Git ordering alone cannot independently prove when data were generated or accessed.
Record checkpoint/data hashes before access in the next protocol.

Prior-art clearance remains open. DIFNet already learns fusion with unknown
correlated sensor noise; distinguish episodic support-conditioned lifted inference
from that sequential filtering setting, and audit learned EP, MVAE and NeuMiss.
The Beijing dataset is a candidate only: audit natural gap frequency, station/hour
identity, leakage, licensing and usable weekly support before freezing an endpoint.
No fresh Beijing labels, HAR labels or protected seeds were opened here.

Sources: [official UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality),
[DIFNet](https://arxiv.org/html/2508.18854v1). UCI displays both a research-only
notice and CC BY 4.0; retain attribution and raw data locally while clarifying
redistribution terms. No raw CSV is committed.

The replay took 41.51 CPU seconds, with zero cloud calls. It is inference/aggregation
reproduction, not an independent rerun of training. The manuscript is unchanged;
its prior PDF compilation limitation remains unresolved.
