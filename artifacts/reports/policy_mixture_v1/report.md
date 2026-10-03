# policy mixture v1

| Method | Mean NLL |
|---|---:|
| frozen | 0.544556 |
| intercept | 0.553351 |
| cv_mask | 0.529770 |
| cv_parity | 0.481822 |
| cv_pooled | 0.486292 |
| moment | 0.447637 |
| neural_mixture | 0.431864 |
| linear_mixture | 0.486320 |
| cv_select | 0.446479 |
| calibrated_guard | 0.435509 |

961-parameter neural predictive mixture, equally pretrained linear scorer, calibrated guard
and support-only controls. Fixed60-epoch checkpoint;768 training worlds/1,536 episodes.
Development48 independent worlds/five cells each;2,400 saved probability arrays.
Moment gain .015773 [.008210,.023337], null harm .001930. Calibrated-guard gain
.003645 [-.000058,.007349] fails the full gate. No confirmatory or safety claim.
Independent saved-score audit passes exactly. OOF fold ordering and moment-selector differences
remain documented. Protocol a6033c0; checkpoints447b488 before evaluation.
