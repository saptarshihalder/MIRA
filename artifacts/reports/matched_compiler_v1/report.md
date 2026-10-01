# Matched compiler development

Fixed final epochs/thresholds; every new seed is used development. Same synthetic teacher/tasks/optimizer/minibatches, different selector architectures. No TFM inference.

| Order | Gamma | Prototype | Linear | Raw DeepSets | Heuristic | Identity | Full parity L1 | CV compiler |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.64110 | 0.64110 | 0.64116 | 0.64132 | 0.64110 | 0.64450 | 0.64839 |
| 1 | 0.9 | 0.20963 | 0.20963 | 0.20963 | 0.20963 | 0.20963 | 0.21822 | 0.21914 |
| 2 | 0 | 0.64139 | 0.64139 | 0.64139 | 0.64139 | 0.64139 | 0.64217 | 0.64834 |
| 2 | 0.9 | 0.20767 | 0.63856 | 0.23974 | 0.20767 | 0.63856 | 0.21594 | 0.20767 |
| 3 | 0 | 0.64221 | 0.64221 | 0.64221 | 0.64221 | 0.64221 | 0.64112 | 0.64851 |
| 3 | 0.9 | 0.20722 | 0.64171 | 0.21388 | 0.20722 | 0.64171 | 0.21684 | 0.20722 |
| 4 | 0 | 0.64039 | 0.64039 | 0.64054 | 0.64039 | 0.64039 | 0.64482 | 0.64931 |
| 4 | 0.9 | 0.20410 | 0.20410 | 0.20410 | 0.20410 | 0.63957 | 0.21315 | 0.20410 |

Matrix teacher accuracy and active-parity-axis presence are evaluator diagnostics, not inference inputs. Selection accuracy is not downstream performance. All query targets are used only after the issued probability is saved. Failures remain in errors.json.
