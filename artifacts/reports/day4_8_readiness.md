# Day 4–8 independent readiness audit — 1 October 2026

**Engineering gates pass for the completed panel. Scientific scope remains narrow; keep the adapter inactive.** This audit used saved artifacts and CPU arithmetic only, with no paid calls, browser visits, source changes, or manuscript edits.

## Independent verification

The frozen protocol SHA256 is `e2058edaf9cb246a691f98659157e573ad94b74c237a53c33b505c711c45fcac`. Its sidecar and every frozen source/input hash match. All **1,440 cells across 120 saved episodes** are complete, without substitutions or recorded warnings: six fixed datasets × five folds × two rates × two gammas × four models × three views. Canonical row counts are 569/350/1,348/4,210/208/270.

Independently reconstructed each episode from canonical values, labels, row/group identities, frozen splits, and exact RNG seeds/mask law. Verified 128 support labels, disjoint support/query groups, native/imposed mask unions, finite probabilities, saved prediction IDs, full unique cell matrix, and preserved hashes. Recomputed empirical log-loss (probabilities clipped to [1e−6,1−1e−6]), Brier, and AUROC; maximum score discrepancy is **1.11e−16**.

For each dataset, concatenated per-query paired losses across five folds, equivalent to query-weighted fold differences. Equally averaged the **six dataset effects**, then computed nominal t intervals with five degrees of freedom. All 48 aggregate means/intervals agree with the report within **5.55e−17**. Folds and repeated conditions were not treated as independent replicates.

Code review verified the predictor/evaluator boundary, group-isolated support-only logistic C selection, Pipeline preprocessing, saved C/CV scores/warnings, and failure preservation. Independently ran 16 relevant ingestion/runner/baseline tests, all passing. Root separately reports broader passing suites.

## Findings that constrain claims

Native-minus-actual-indicator empirical NLL gains at gamma .8 (positive favors indicators):

| Dataset | TabPFN r=.1 | TabPFN r=.5 | TabICL r=.1 | TabICL r=.5 |
|---|---:|---:|---:|---:|
| WDBC | 0.00791 | -0.00309 | 0.00064 | 0.00089 |
| ionosphere | -0.00089 | 0.00371 | 0.00907 | -0.00931 |
| Banknote | 0.03744 | -0.00186 | 0.02074 | -0.01011 |
| spambase | -0.01056 | 0.00423 | -0.01096 | -0.00112 |
| sonar | -0.00337 | 0.00776 | -0.02036 | -0.00697 |
| Heart | 0.02733 | 0.01216 | 0.00292 | 0.00158 |

Dataset-balanced TabPFN gains are **.009644 [−.010141,.029428]** at r=.1 and **.003818 [−.002215,.009851]** at r=.5. TabICL native gains are near zero/negative, with both intervals crossing zero. TabICL's r=.1 actual-versus-shuffled gain is **.013325 [.002281,.024369]** nominally; disclose this control, but its native gain is only .000341. All intervals are unadjusted descriptive summaries of a small, fixed, nonrepresentative panel.

Mean-imputed logistic gains **.171407 [.074178,.268635]** at r=.5/gamma .8, while gamma-zero indicators worsen NLL by **.033732 [.002858,.064606]**. XGBoost native and actual-indicator probability arrays are bit-identical throughout; shuffled indicators alter predictions. These findings do not support an unconditional indicator recommendation or a universal frozen-model deficiency.

No dataset has a context-constant outcome-associated active column. **Zero finite active-column values equal the fitted imputation mean** in support or query across all episodes. Ionosphere has an inactive constant second column; Spambase has inactive context-constant columns {3,21,33,37,46} in some episodes (zero-based indices). They account for approximately 2.94%/1.97% overall observed support-mean matches. Other datasets have none. Diagnostics use float32 inputs and scikit-learn 1.8.0's support-fitted SimpleImputer reference, without instrumenting subsequent transforms. The panel does not reproduce the deliberate erasure of an informative active mask.

Saved-task L1 interaction logistic reaches expected NLL .193364, oracle gap .008930, at gamma .9 and descriptively beats TabPFN indicators. This post-confirmation analysis uses engineered interactions; known-beta references and generator-informed mixtures remain separately labeled. No useful adapter headroom has been established.

## Final gates

- [x] Frozen identity, grouped splits, learner isolation, tests, all requested cells, probabilities, warnings, and six-unit aggregation verified.
- [x] Complete controls and null/negative effects retained; no oracle claimed for real covariates.
- [x] Root reconciles reserved compute/provider billing and preserves final provenance (root-owned closure below).
- [x] Root verifies that manuscript/decision accurately limit claims to tested versions, retrospective synthetic outcome association, fixed datasets, and descriptive uncertainty.

Unknown acquisition groups remain a limitation. The mask law gives class-conditional probabilities .02/.18 at r=.1 and .1/.9 at r=.5; pooled rates depend on prevalence. This is a controlled intervention on real covariates, not evidence about natural missingness, causal acquisition policies, clinical use, online shift, or acceptance.

## Root closure after independent audit

Billing/provenance reconciliation saved: 19 terminal ledger entries (18 complete, one historical failure), reservations $9.05, provider app report $0.50322492 with lag/image-attribution caveats. The $3 reserve remains inside $26. Canonical data and all frozen hashes remain unchanged. Manuscript/decision/STATUS now reflect the actual full panel, all controls and negative outcomes. Figure regenerated from verified JSON and visually checked. Native compilation of the final same source failed with the preserved environment diagnostic; PDF readiness remains open. These closure checks are root-owned, not a claim of independent compiler or provider-cost audit.
