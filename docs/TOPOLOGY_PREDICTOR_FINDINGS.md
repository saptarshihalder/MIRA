# Source-only topology predictor result — October 5, 2026

Both frozen sampling gates FAIL. Eighteen neural predictors and six tree controls were fitted before validation scoring; all24 model/score replays pass exactly. No original Wine validation/development row was scored. Candidate-pool feature groups are excluded from source validation.

| Selection / predictor | Red NLL | White NLL |
|---|---:|---:|
| topological | 0.692505 | 0.676785 |
| uniform | 0.736863 | 0.647930 |
| farthest | 0.694287 | 0.702192 |
| topological_tree | 0.704331 | 0.686694 |

H0 selection improves over uniform on red but not over farthest-point by the required .01. It harms white relative to uniform. These overlapping source pools belong to one dataset family. No architecture novelty, independent-validation, general safety or venue-readiness claim follows. This finite batch is closed; do not select its favorable domain or rerun until positive.

Equal64-label budgets,256-candidate access and matched neural updates isolate selection. Neural models have3618 parameters and300 updates per fit; tree training differs. Total run9.422 CPU seconds,zero cloud calls. Source normalization and all selected identities, checkpoints, masks and scores are retained. Run `python experiments/topology_predictor_replay.py` to verify raw grouping, selection, frozen hashes and all predictions with no new fit.
