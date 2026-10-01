# Fixed six-dataset real-covariate panel

Prepared from the official UCI API and official standardized CSV downloads on 1 October 2026. The six datasets were selected before model evaluation in `configs/real_covariate_panel.md`. No model scores were inspected during preparation.

The exact official CSV SHA256s, positive labels, dataset DOIs, citations, and source pages are frozen in `configs/panel_sources.json`. Every official dataset page was checked for its CC BY 4.0 license. Raw CSVs, official API responses, and page snapshots remain locally under `artifacts/data/raw_uci/`, which is ignored by Git. Their hashes are recorded in each dataset manifest. Citations and license URLs are retained in the committed manifests under `artifacts/manifests/panel_data/`.

Canonical arrays are committed under `artifacts/data/real_panel/<slug>/dataset.npz`; keys are `X`, `y`, `row_ids`, `group_ids`, `native_mask`, and `multiplicity`. IDs and labels are excluded from X. Heart retains only age, resting blood pressure, serum cholesterol, maximum heart rate, and oldpeak. Numerical zeros are preserved. No native missing values were found.

Exact selected-X/label duplicates are removed with multiplicities and raw-row memberships recorded in `row_maps.json`. Conflicting-label rows with identical X remain and share a group. Supplied source IDs are preserved and linked to groups; only WDBC provides an ID column. Sonar object identities and Spambase collection groups are unavailable, so duplicate grouping cannot establish full acquisition independence.

`splits.json` freezes all five `StratifiedGroupKFold` partitions using seed `61000+UCI_id`. Each support has 128 stratified rows sampled with seed `62000+100*UCI_id+fold`. Queries are held-out rows, with stratified capping at 1,024 using seed `63000+100*UCI_id+fold`; all actual query folds are below that cap. Identical-X/source groups never cross training/held-out boundaries. Datasets are aggregation units; folds have overlapping training pools.

Cleaned row counts: WDBC 569, Ionosphere 350, Banknote 1,348, Spambase 4,210, Sonar 208, Heart 270. Spambase retains three conflicting-label groups. `summary.json` records all fold sizes.

Preparation used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and scikit-learn 1.8.0. From the repository root, verify the bundled canonical data without network access:

```powershell
python -m pytest tests/test_panel_data.py -q
```

Use the recorded environment for re-preparation. With the original cached raw snapshots, `python scripts/prepare_real_panel.py` replays in place and refuses any differing frozen outputs. A fresh checkout should download to separate ignored outputs, because official HTML/API snapshots can change while the pinned CSV bytes remain identical:

```powershell
python scripts/prepare_real_panel.py --data-root artifacts/runs/panel_reproduction/data --manifest-root artifacts/runs/panel_reproduction/manifests
```

Compare reproduced `dataset.npz`, `splits.json`, and `row_maps.json` hashes against the frozen per-dataset manifests. `mira.panel_data.load_panel(...)` returns copies with read-only arrays; native values and masks are kept separate from subsequently imposed missingness. The second preparation pass reproduced all saved bytes. Tests verify artifact hashes, canonical IDs, duplicate handling, immutable arrays, declared labels/features, and all 30 group-isolated partitions.
