# Support-conditioned Bellman extension — October 5, 2026

Implemented the L2M/BRiG-inspired integration in RELATED_PAPER_EXTENSION.md; full novelty remains unresolved. The frozen synthetic mechanism gate fails in all three seeds. Privileged source teachers and the same six Boolean families limit all interpretations. No native/external evidence was obtained.

| Seed | Support Q | Myopic Q | Blind Q | Support-count DP | Shared-predictor DP |
|---|---:|---:|---:|---:|---:|
| 814001 | 0.623071 | 0.622244 | 0.622278 | 0.583254 | 0.637359 |
| 814002 | 0.625227 | 0.624790 | 0.622638 | 0.583254 | 0.633038 |
| 814003 | 0.618033 | 0.616764 | 0.615449 | 0.583254 | 0.627886 |

The learned support-Q variant loses to myopic, support-blinded and count controls in every initialization, despite beating the plug-in shared-predictor planner. That latter comparison has different planning supervision and is not a pure architecture or amortization advantage.

Post hoc exact reference audit diagnoses an upstream constraint: with the frozen learned predictor, the best possible policy gains over count-DP are -.002544, -.004357 and +.003660 on the used test tasks, all below the required .01. On source tasks those ceilings are -.007040, -.009318 and -.001626. Thus additional Q optimization alone cannot clear this gate with these predictors. This does not distinguish insufficient training from unsuitable representation or model capacity. A source-only predictor-adequacy gate should precede future policy training.

Three predictor and nine equal-parameter Q checkpoints were saved before scoring; 1,950 updates took 15.844 CPU seconds. The replay checks all 13,824 cells, twelve checkpoints and exact shared predictor weights. Maximum float32/float64 error is 1.146e-7. Independent DP found 69 equivalent acquisition-order ties; identical acquired sets, costs and predictions were verified. Three audit-only failed attempts exposed tie-order and precision expectations, then the corrected audit passed in 17.312 seconds. No scientific code, checkpoints, scores or gate changed; no training retry. Cloud charges remain zero.

Close this finite probe. User-requested H0 topology sampling is now implemented separately (TOPOLOGICAL_SAMPLING.md), on Wine source features only, with matched uniform selections. Its geometric checks are not predictive results or a novel trained model. Next: preregister source-only predictor adequacy and sampling controls before any further fitting; preserve all used panels and protected confirmation seeds.
