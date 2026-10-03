# Reproduce the current trained candidate

Read `docs/CROSSFIT_FINDINGS.md`, `CROSSFIT_PROTOCOL.md` and `CROSSFIT_RECOVERY.md`. The synthetic robustness gate passes; the full neural-specific gate fails because the scalar/query comparator remains statistically unresolved. This is not submission readiness.

The actual Modal training protocol was committed as f4dcb47 before dispatch. Its SHA256 is276095220d847cf279373d84d6c220a8f82dc36d4d1c14763adf2859676f438c. All three checkpoints were saved after500 CUDA updates. The paid evaluation failed at an ordinary Platt control; all models and132 GPU prediction files survived. CPU recovery preserves the same convex calibration objective and every planned case. The original failed run is archived with the recovered results; it remains counted in the ledger.

From the repository root, run locally without cloud charges:

```powershell
../.venv-compiler-cpu/Scripts/python.exe -m unittest discover -s experiments/crossfit_v1 -p test*.py
../.venv-compiler-cpu/Scripts/python.exe experiments/crossfit_v1/audit.py --folder artifacts/reports/crossfit_v1_gpu
```

The environment uses Torch2.8, NumPy2.3.5, scikit-learn1.8 and SciPy. The audit regenerates observed data and label identities, recomputes all scores directly, restores every trained checkpoint, replays all neural predictions/branch decisions, and independently fits the guarded logistic comparator. Its output audit timestamp/runtime is not a training event. Recreating the original training is a new paid run and requires a new reservation; the existing launcher intentionally refuses retries.

Three grouped unit tests cover query/verification isolation, nonzero gradients in both learned modules, inner-objective descent, hidden-value invariance and exact frozen-logit fallback. Numeric support-objective descent and finite-sample observed no harm do not guarantee future test risk. No real-data result for this candidate or new confirmation is claimed.
