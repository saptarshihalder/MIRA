# Real-covariate panel v1 — predeclared 1 October 2026

Status: completed and independently verified, 1,440/1,440 cells under the frozen `real_panel_v1.json` protocol. The six datasets were selected before downloads or scores; the complete protocol, source and data hashes were committed as `dbda963` before evaluation. Official UCI metadata, CC BY 4.0 citations and downloaded CSV SHA256s are preserved in `panel_sources.json` and `artifacts/manifests/panel_data/`.

| Dataset / official source | Rows | Retained features | Positive label |
|---|---:|---:|---|
| [Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30; discard ID | M |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | 351 | 34 | g |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | 1,372 | 4 | 1 |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | 4,601 | 57 | 1 |
| [Sonar Mines vs. Rocks](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) | 208 | 60 | M |
| [Statlog Heart](https://archive.ics.uci.edu/dataset/145/statlog+heart) | 270 | 5 continuous only | 2 |

Heart retains `age`, `rest-bp`, `serum-chol`, `max-heart-rate`, `oldpeak`; its other eight features include categorical/binary fields and are excluded. Listed sizes precede cleaning. Spambase metadata was additionally verified through the [official UCI API](https://archive.ics.uci.edu/api/dataset?id=94).

## Splits and duplicates

Freeze one canonical row order. Remove exact duplicate `(selected X, y)` rows, record multiplicities, and group all identical selected-feature vectors together, including conflicting-label rows. Preserve provided subject/object identifiers for grouping when available, then exclude identifiers from predictors. Use `StratifiedGroupKFold(5, shuffle=True, random_state=61000+UCI_id)`; each held-out fold is a query partition. Select 128 support rows from training folds by a deterministic stratified sampler seeded `62000+100*UCI_id+fold`; optionally compare nested 32-row contexts only after pricing. Cap queries at 1,024 with a seeded stratified subsample. Save all row/group IDs. No panel tuning: synthetic development determines model settings.

Sonar includes repeated views of objects; Spambase has personalized collection effects. Duplicate grouping cannot recover unavailable acquisition/source groups. Report this limitation; treat datasets, rather than overlapping folds or query rows, as aggregation units.

## Imposed observation process

Keep raw values/labels and original mask `M_native` immutable. UCI reports no native missing values in these datasets; verify on ingestion, preserve numerical zeros, and log deviations. Store `M_imposed` separately; model input uses their union.

For each fold, choose one outcome-associated active column by seeded RNG without inspecting performance; it need not be predictive before masking. Use `r in {0.1,0.5}`, `gamma in {0,0.8}`, sign `+1`, and
`P(M_imposed,j=1 | y)=r*[1+gamma*(2*y-1)]`
on that column; other columns use independent Bernoulli(r). Record realized pooled/class-conditional rates. Labels are available only to the evaluator's retrospective mask generator, never query predictor inputs. This is an imposed association, not a realistic acquisition-policy claim.

Every method receives identical support labels, query IDs, values, and realized masks. Compare native, actual indicators, and shuffled indicators; price context-imputed controls subsequently. Hold complete-data splits/class prevalence fixed. Report empirical log-loss/Brier/AUROC; no real-data oracle is available. The final frozen design executes all five folds, four models (TabPFN v2, TabICLv2, XGBoost, generic logistic), two rates, two gammas and three representations: 1,440 cells. The preliminary one-fold pricing suggestion was superseded before scores by the bounded complete design. Report every executed cell and any budget truncation; never replace datasets after scores.

After exact duplicate removal, row counts are 569, 350, 1,348, 4,210, 208 and 270 (6,955 total). All six native masks are empty. Logistic selects C using support-only three-fold StratifiedGroupKFold with imputation/scaling inside the pipeline. Fold losses are query-weighted within datasets; dataset-balanced estimates use six dataset effects and descriptive, unadjusted nominal 95% t intervals. Saved predictions must pass `scripts/report_real_panel.py` before interpretation.
