# Predictive-loss context adapter: failed development gate

Protocol/code freeze `ebac808`; final checkpoints committed `4514490` before development.
CPU training only: 768 independent paired-policy worlds, 1,536 episodes, 40 fixed epochs,
128 validation episodes, 47.062 seconds. Context/no-shift models11,176 parameters; ordinary12,601.
Evaluation:48 independent worlds ×5 policy cells,240 episodes,2,160 saved prediction arrays.
No additional paid call or confirmation. Prior $10.55 reservations and $3 reproduction reserve unchanged.

| Method | Expected NLL | Comparator minus context (nominal 95% task interval) |
|---|---:|---:|
| frozen | 0.568161 | 0.054260 [0.031455, 0.077064] |
| intercept | 0.571151 | 0.057250 [0.033737, 0.080762] |
| cv_mask | 0.555391 | 0.041489 [0.019722, 0.063256] |
| cv_parity | 0.504136 | -0.009766 [-0.020402, 0.000871] |
| cv_pooled | 0.501625 | -0.012276 [-0.021102, -0.003450] |
| moment | 0.469295 | -0.044606 [-0.055141, -0.034072] |
| context | 0.513901 | reference |
| ordinary | 0.566892 | 0.052991 [0.030107, 0.075875] |
| no_shift | 0.541084 | 0.027183 [0.014161, 0.040206] |

The context learner beats ordinary and no-shift neural controls, but loses to the centered-moment
correction by .044606 NLL and has .045750 mean null harm against the .01 limit. The gate fails.
Expected NLL averages paired policy cells within outcome worlds before uncertainty. Intervals are
developmental and nominal; they do not support a confirmatory multiple-comparison claim.

Independent audit reconstructs all saved expected/empirical scores and task intervals exactly.
Oracle reconstruction from saved float32 U differs by at most3.35e-8. Six boundary checks passed
using runpy (pytest is absent in the isolated CPU environment).

Limitations: sufficient always-observed U, four-bit/full-parity structural prior, engineered moments
in addition to a learned row encoder, limited target labels and non-identical optimization/capacity.
Only rate/block/value extremes exceed pretraining support;17/48 reversal cells have null source signal.
Pooled ridge penalizes the source intercept while exempting the final target-domain intercept.
No real-data, TFM, clinical, causal-policy-identification, safety or venue-readiness claim.

Reproduction with the pinned CPU environment (Torch2.8.0+cpu,NumPy2.3.5,sklearn1.8.0):

```powershell
& '../.venv-compiler-cpu/Scripts/python.exe' experiments/policy_adapter_v1/pilot.py train --out artifacts/runs/policy_reproduction_v1
& '../.venv-compiler-cpu/Scripts/python.exe' experiments/policy_adapter_v1/pilot.py evaluate --out artifacts/runs/policy_reproduction_v1
```

Do not overwrite existing outputs; clean reproduction remains a later readiness gate.
Next design requirements: `docs/POLICY_ADAPTER_NEXT_GATE.md`.
