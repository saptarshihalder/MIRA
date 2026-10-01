# MIRA

Controlled experiments on missingness representation in frozen tabular predictors.

**Status:** research in progress. The analytic CPU pilot has been reproduced, and 1,296 TFM/tree development cells have completed. A frozen pairwise hypothesis passed on 20 fresh tasks; real-covariate evaluation remains pending; no adapter or clinical result is claimed.

Start with [STATUS.md](STATUS.md), [SPRINT.md](SPRINT.md) and [decision.md](decision.md). The manuscript is `paper/main.tex`, edited in Codex's built-in LaTeX panel; compilation currently has an environment error. The original repository was empty when cloned on October 1, 2026; this checkout is the implementation workspace.

## Local verification

Use a fresh Python 3.12 environment and install this package with its test extras. Exact successful GPU environments are written with each run rather than pretending that a bootstrap is a complete lock.

```powershell
python -m pip install -e ".[test]"
python -m pytest tests -q
python scripts/run_mechanism.py --help
```

## Bounded Modal smoke

The Modal CLI uses existing local credentials. Keep credentials outside Git. From this directory, with an environment containing `modal==1.6.0`:

```powershell
$env:PYTHONUTF8='1'
python -m modal run infra/modal_mechanism.py --run-id smoke_tabpfn_v2 --model tabpfn:v2
```

Every run needs a unique ID. Smoke calls use the preserved pilot, reserve $0.50 of the $26 cap (initial historical smoke reservations were $0.35), use one T4 and expire after 900 seconds. See `configs/budget.json`. Results are extracted to `artifacts/runs/<run_id>`; logs, failed calls and cost estimates remain in the ledger. GPU prediction files and checkpoints are ignored by Git; publish only a curated anonymous reproduction bundle after final review.

The supplied task notes are proposed specifications and evidence. Current user instructions govern authorization and the deadline. This project reports uncertainty and limitations; readiness depends on executed experiments.

Development reports: `artifacts/reports/development_gaussian/`, `artifacts/reports/development_collision/`, and `artifacts/reports/preprocessing_audit.md`. Plot regeneration: `python scripts/plot_development.py` (requires matplotlib). Raw GPU predictions remain local under ignored `artifacts/runs/`; their hashes, exact configs and source identities are preserved with each run. Package them for final artifact review once the scientific protocol is frozen.

Day 1–3 checkpoint: [development summary](artifacts/reports/day3_summary.md), [data audit](artifacts/reports/day3_data_audit.md), [readiness audit](artifacts/reports/day3_readiness.md). Local dispatch requirements and named catalog usage: [infra/README.md](infra/README.md). Regenerate the summary with `python scripts/summarize_day3.py` after verifying the individual saved-run reports.

## Confirmed scoped result

A protocol committed before evaluation completed 360 confirmation cells. On 20 fresh Gaussian pairwise-mask tasks, TabPFN v2's indicator gain is **0.10389 nats** [0.07969,0.12809]; actual versus same-width shuffled gain is **0.09499** [0.06162,0.12837]. These 97.5% paired intervals cover the two prespecified primary contrasts with Bonferroni familywise coverage of at least 95%. Scope: 256 labels, named checkpoint, specified synthetic generator. No real-data or adapter claim.

![Fresh-task confirmation](artifacts/figures/confirmation_pairwise.png)

[Protocol](configs/confirmation_pairwise_v1.json), [decision and audit](artifacts/reports/confirmation_pairwise_v1/decision.md), [full report](artifacts/reports/confirmation_pairwise_v1/report.md). Regenerate reports with confidence .975, then run `python scripts/analyze_confirmation.py`; raw predictions must be present locally.
