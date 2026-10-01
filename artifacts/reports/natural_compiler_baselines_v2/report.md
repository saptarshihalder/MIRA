# Native CPU controls, three-fold repair

24 predictions complete; zero errors/warnings; $0 paid compute. Original frozen files/protocol unchanged. Logistic uses support-only three-fold group CV and C=(.01,.1,1,10); HistGB retains its fixed 200 iterations and 15 leaves. Native numeric values remain unchanged; identity appends all native mask bits.

| Dataset | Model | Native NLL | Identity NLL |
|---|---|---:|---:|
| hepatitis | logistic | 0.461546 | 0.404532 |
| hepatitis | histgb | 0.563026 | 0.563026 |
| horse_colic | logistic | 0.555609 | 0.560936 |
| horse_colic | histgb | 0.753356 | 0.753356 |

NLL weights the three folds by query count within each dataset (155 and353 queries). Descriptive naturally missing outcomes; no oracle/clinical claim. Canonical heldout partitions cover every row once; outer and inner source groups are isolated. Six query-label inversion replays produce identical native-logistic probabilities; learner rejects evaluator episode dictionaries.

v1_failed_manifest.json preserves the reproduced five-fold validator failure and tracebacks separately. split_audit.json, boundary_audit.json, prediction IDs/hashes, environment lock, full data/source hashes and protocol SHA are retained. Query labels enter scoring only after prediction NPZ files are saved.
