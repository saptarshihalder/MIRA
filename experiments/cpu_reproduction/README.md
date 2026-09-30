# CPU reproduction and audit

This folder is self-contained. The supplied source files are copied byte-for-byte into `source/`. The original CPU scripts, math checks, and full results are bundled in `reference/mira_pilot/`, with SHA256 hashes in `reference/SHA256.json`. Original files outside this repository were preserved. No paid services or model checkpoints are used.

Executed on 1 October 2026 with Python 3.12.14 and the supplied core package pins. Full transitive versions are recorded in `requirements-lock.txt`; the initial execution machine, BLAS and one-thread configuration are in `execution_environment.json`. The original full-handoff integrity audit is preserved in `execution_integrity.json`. `environment.json` and `integrity.json` record the latest portable verification.

Use Python 3.12.14. From this folder in PowerShell, create an environment and verify existing results without rerunning the full experiment:

```powershell
python -m venv .venv
& '.venv/Scripts/python.exe' -m pip install -r requirements-lock.txt
& '.venv/Scripts/python.exe' audit_checks.py
& '.venv/Scripts/python.exe' source/check_math.py
& '.venv/Scripts/python.exe' compare_results.py
```

To rerun the full matrix, use `& '.venv/Scripts/python.exe' source/headroom.py --dev-tasks 12 --eval-tasks 48 --queries 2048 --eval-offset 30000 --out results` and then run `compare_results.py` again. On macOS/Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python` and omit PowerShell's `&`.

`integrity.json` checks every bundled original against the input manifest and confirms source-copy identity. `leakage_and_posterior_checks.json` records perturbation checks and exhaustive finite-mask posterior calculations. `comparison.json` compares the full raw result matrix, independently recomputes method choices and paired Monte Carlo standard errors, and records output hashes. `run.log` and `run.stderr.log` preserve the original reproduction output. The virtual environment is excluded from Git.

The original evaluation seeds are deliberately reused for reproduction. These results remain exploratory and must not be represented as untouched final evaluation for a method developed after seeing the handoff. The selected method in each family/protocol/budget is an evaluator-indexed diagnostic envelope. No TFM or XGBoost inference was executed by this audit.
