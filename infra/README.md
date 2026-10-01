# Bounded Modal development

The local dispatch environment needs `modal==1.6.0`, `numpy==2.3.5` and
`scipy==1.17.0`: matrix validation imports the same CLI as the experiment.
GPU model dependencies remain pinned in `modal_mechanism.py`. Credentials are
local and excluded from the repository.

From the repository root, dispatch one named entry and inspect its manifest
and errors before starting another:

```powershell
$env:PYTHONUTF8='1'
python -m modal run infra/modal_mechanism.py --run-id day3 --matrix configs/day3_matrix.json --entry gaussian_imputed_controls
python scripts/report_mechanism.py --out artifacts/runs/day3_gaussian_imputed_controls --report-out artifacts/reports/day3_gaussian_imputed_controls
```

Run IDs cannot be reused. The wrapper reserves $0.50 under an OS lock before
calling Modal, retains failed/crashed reservations, and preserves $3 of the
$26 cap for reproduction. Each invocation has a fresh remote output directory,
one T4 container, a 900-second limit, and no retries. Saved runtime costs are
estimates; provider billing can lag. Cloud results, including partial failures,
are extracted to ignored `artifacts/runs/`; verified reports and provenance are
tracked. Confirmation is rejected by this development catalog.

The first Day 3 dispatch stopped during local validation because this local
environment lacked NumPy; installing the two validation dependencies resolved
it before any GPU call or ledger reservation.
