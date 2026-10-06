# Lifted cavity network

Learned expectation propagation over a target-plus-nuisance latent for in-context sensor fusion with missing
sensors. Findings: `docs/LIFTED_CAVITY_FINDINGS.md`. Frozen confirmation: `docs/LIFTED_CAVITY_CONFIRMATION.md`.
Plan: `docs/OCT10_PLAN.md`.

| File | Role |
|---|---|
| `family.py` | Vectorised copy of `cavity_site.generate` (same law, any sensor count), exact oracle, population Gaussian, closed forms (repo ridge, complete-case ridge, EM-Gaussian) |
| `fa.py`, `anchors.py` | Support-only supervised factor analysis; numpy reference and batched torch versions |
| `models.py` | `LiftedCavity` (K nuisance dims, cavity on/off), `AnchorMLP` control, `closed_form`, wrapper for the repo's anchored classes |
| `data.py` | Task pools with precomputed anchors; source minibatches (≤1 missing sensor); evaluation batches over all k-missing masks |
| `train.py` | Shared recipe for every learned model (60k-task pool, 20k AdamW steps) |
| `evaluate.py` | Build panels with references (`panel`), score a trained run (`model`) |
| `collect.py` | Per-panel table with gap closed and paired task-level intervals |
| `oracle_audit.py` | Exact oracle and closed forms on the repository's closed cavity panels |
| `score_repo_panel.py` | Score new models on those panels as stored |
| `airq.py`, `finetune_airq.py`, `collect_airq.py` | UCI Air Quality weekly episodes, source-week fine-tuning, test-week table |
| `confirm.py` | Endpoints of the frozen confirmation protocol |
| `make_report.py`, `make_figure.py` | Tables and figure for the findings document |

CPU only. `torch>=2.3`, `numpy`, `scipy`, `pandas`, `matplotlib`. Unit checks: `python -m pytest tests/test_lifted_cavity.py -q`.

The Air Quality CSV is not committed. Download it from UCI (dataset 360, CC BY 4.0) and check its sha256 against the
value printed by `airq.py build` (development used `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`).
