# CPU pilot audit — 1 October 2026

The complete supplied CPU matrix reproduces numerically. Its results support strong simple baselines on the stated synthetic generators; they do not establish TFM failure or an adapter contribution.

## Integrity and execution

All seven entries of the original handoff's SHA256 manifest match; that audit is preserved in `experiments/cpu_reproduction/execution_integrity.json`. The CPU pilot scripts were absent from that manifest; this audit separately hashed them and copied `headroom.py`, `check_math.py`, `tfm_mechanism.py`, and `requirements-cpu.txt` byte-for-byte into `experiments/cpu_reproduction/source`. Original files were preserved. For portable verification, the supplied CPU scripts, math checks, and full original results are bundled under `experiments/cpu_reproduction/reference/mira_pilot`, with an eight-file SHA256 manifest under `reference/SHA256.json`. `integrity.json` verifies that bundled subset.

Source review found no network calls, paid services, credential access, destructive file operations, or dynamic execution in the CPU path. Output writes were confined to the reproduction folder. Importing the TFM transformation helper did not instantiate or run any TFM.

Ran the recorded command with 12 development and 48 evaluation tasks per family, 2,048 queries, and evaluation seeds 30000–30047. It completed successfully in 116.84 seconds, producing 31,680 raw rows and 24 summary cells; stderr was empty. The original recorded runtime was 92.20 seconds.

## Environment and numerical agreement

The isolated environment matches the recorded Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, and threadpoolctl 3.6.0. It uses Windows 11 build 26200 on AMD64, OpenBLAS 0.3.30, and one BLAS/OpenMP thread. `execution_environment.json` preserves the execution record. The original platform, BLAS builds, CPU, and transitive package versions are not supplied, so exact environment equivalence cannot be claimed. The rerun locks joblib 1.6.0 and cloudpickle 3.1.2 as well as the specified packages.

All raw matrix keys match. Maximum absolute raw NLL difference is `3.785594060445874e-11`; 41 of 31,680 rows differ by more than `1e-12`. Method selections are unchanged in every summary cell. Maximum absolute summary difference is `2.220446049250313e-16`. Independently recomputed development choices, paired oracle gaps, and Monte Carlo standard errors match exactly.

At 32 fresh target labels, reproduced selected-simple excess NLL is approximately 0.00741 ± 0.00304 SE for label-only, 0.00488 ± 0.00174 for joint-pair, 0.00729 ± 0.00275 for value-dependent, and 0.04248 ± 0.01312 for sparse-pair. All four 128-label gaps remain below 0.001.

## Math and information access

The supplied numerical checks passed: fixed-share mixture identity error `1.7763568394002505e-15`, positive minimum path-bound slack 0.332995, exact posterior error `2.220446049250313e-16`, and finite-prior error `1.1102230246251565e-16`. These are numerical checks of the stated identities, not a proof of an unimplemented delayed-feedback method.

An independent audit exhaustively enumerated all masks in each family, checking likelihood normalization, positive probabilities, and the posterior formula. For each of four families, complementing every query label and replacing every query oracle probability left all 21 non-oracle predictor outputs bitwise unchanged. Flipping context labels changed the sparse mixture, confirming that the perturbation audit exercises an actual learned prediction path. Changing task-family structural metadata left generic predictors unchanged; the explicitly privileged finite-prior reference was excluded from that check.

All five TFM input transforms were invariant to query-label/oracle perturbations. A separate constructed case verified that imputed query values depend only on context means and use zero when a context column is entirely missing. This certifies the reviewed preprocessing helpers, not unexecuted TFM package internals. `mechanism_oracle` intentionally uses the known query posterior as an evaluation reference.

## Interpretation and artifacts

The generator-informed sparse mixture includes the true interaction basis and gamma 0.8 in its prior. The base knows the exact full-data posterior. Fresh-target contexts reveal the shift boundary; rolling-context method choice uses family and time-since-shift cells defined by the evaluator. This remains a diagnostic envelope, not a deployable online selector. Gamma-zero controls, real data, TFM/XGBoost inference, and delayed-feedback online experiments were not part of the supplied CPU matrix.

The original seeds were reused only for reproduction and cannot serve as untouched final evaluation for a subsequently developed method. No paid compute was consumed.

All executable checks, raw outputs, logs, lockfile, environment, and comparisons are under `experiments/cpu_reproduction/`; `comparison.json` and `leakage_and_posterior_checks.json` contain the detailed evidence. Portable verification scripts were rerun after packaging; the full matrix did not need another rerun.
