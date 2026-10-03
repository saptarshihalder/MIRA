# Recursive JEPA engineering prototype

This is a working trained prototype, not a passed scientific gate. Read `docs/RECURSIVE_JEPA_PROTOCOL.md`. The exact pre-run commit is252c348; protocol SHA256 is96258a1fb109381b0e7662850e39753e4c9731a14ef150be3f401e58575f4f1f.

The three tied refinements use labeled-support class/residual memories and nonlinear row features. Backpropagation through support-head updates teaches the encoder/refiner using meta-training query labels. An explicitly updated, frozen EMA teacher supplies observed-only, support-conditioned representation targets. Prediction accepts no query labels. Fixed step sizes, bounded corrections and an available frozen intervention do not establish a learned safety gate.

Actual pilot: NVIDIA A100-SXM4-40GB, three controls64 updates each, 6,880 trainable parameters each (the supervised-only auxiliary predictor receives no gradient), 8,448 stored parameters including teacher. Same optimizer steps do not match recurrent FLOPs or intermediate supervision. All scores and checks are diagnostic; no validation selection, confirmation, native-data efficacy or novelty claim.

Tracked results include every prediction file and checkpoint, so verification needs no cloud training or payment. From the repository root, using the environment with Torch2.8, NumPy2.3.5, SciPy and scikit-learn1.8:

```powershell
../.venv-compiler-cpu/Scripts/python.exe -m unittest discover -s experiments/recursive_jepa_v1 -p test*.py
../.venv-compiler-cpu/Scripts/python.exe experiments/recursive_jepa_v1/audit_result.py --out artifacts/reports/recursive_jepa_v1_gpu --audit-output artifacts/manifests/recursive_jepa_curated_replay.json
```

The auditor recomputes all96 NLL/Brier arrays, regenerates observed values/masks/labels/row identities, reconstructs frozen logits, restores all teacher/online parameters, tests hidden-value invariance and exact frozen intervention, and replays48 predictions on CPU. It does not rerun training, certify representation robustness, or prove model provenance beyond the saved source/config/state identities and diagnostics.

The full model's JEPA MSE drops sharply while embeddings have low spread. Neural variants perform similarly; strong support logistic is much better on sign flip. Next work must examine per-task discriminative geometry and nonlinear feature structure, then freeze a compute-matched scientific development gate. Do not use confirmation seeds98000-98019 or automatically relaunch this paid pilot.

The isolated CPU smoke uses `run.py --device cpu --smoke --out <new-directory>`, two updates and separate engineering seeds. GPU dispatch is intentionally protected by a consumed ledger reservation; `launch.py` refuses repeat calls. Modal provider reports may lag; retain the $.80 provision. The $20 cap and $3 reproduction reserve apply across the entire project.
