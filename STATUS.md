# MIRA status — October 1, 2026

User target: NeurIPS-quality draft by October 20; token conservation; $26 total compute cap.

## Completed

- Cloned the user's empty GitHub repository into `MIRA/`; working branch `codex/mira-research`.
- Preserved and hash-verified handoff in workspace `reference/handoff`. Supplied and bundled PDFs have identical extracted text despite different binary hashes.
- Reproduced full CPU pilot with Python 3.12.14 and exact core dependency pins: 31,680 raw result rows, 24 summary cells, 116.84 seconds. Maximum summary discrepancy 2.22e-16, raw NLL discrepancy 3.79e-11. Original hardware/BLAS not known.
- Math and predictor-information checks pass: query labels/oracles cannot alter the 21 non-oracle predictors; context labels can. Reports and exact commands are in workspace `experiments/cpu_reproduction/` and `artifacts/reports/cpu_audit.md`.
- Verified live released-package APIs: TabPFN 9.0.0 and TabICL 2.2.0. Modal uses Torch 2.8.0 and records full transitive locks.
- Successful Modal T4 inference smoke for TabPFN v2 and TabICL v2, each 6 model/representation cells, one development seed, gamma 0/.8. Checkpoint hashes and raw predictions saved under `artifacts/runs/`. No adapter trained.
- TabPFN v2 smoke active duration 16.14 seconds; first provider-reported app cost $0.00364287 (billing may lag). This differs from the $0.35 conservative reservation; ledger reserves failed and successful attempts against the cap.
- Sprint calendar and budget documented; early manuscript source opened in native editor.
- Typed experiment runner and saved-prediction reporter completed; 29 tests pass. Support/query row IDs are saved and verified disjoint, evaluator targets are separated from predictor inputs, and resume checks hashes.
- Completed and independently regenerated two 135-cell development matrices (Gaussian and exact-zero collision). Each contains 3 seeds × 5 gamma values × 3 models × 3 modes. Exact code/dependency/checkpoint identities and predictions are saved.
- Exact TabICL 2.2.0 preprocessing audit and all 15 paired-input checks completed. Native collision predictions are identical across gamma within each task seed.
- Six-dataset real-covariate panel predeclared in `configs/real_covariate_panel.md`; no data downloaded yet.
- Generated and visually checked `artifacts/figures/development_representation.png` and SVG from verified summaries.
- Provider-reported MIRA app charges total $0.05438541 at the latest retrieval; this may lag and image-build attribution is not separately audited. Conservative ledger reservations total $2.05 across five calls, including the failed checkpoint attempt. $26 cap and $3 reproduction reserve remain in force.

## Current evidence

At gamma .8 and seed 40000, TabPFN v2 expected log-loss was .316867 native, .312093 with indicators, .313516 with shuffled indicators; oracle .302073. TabICL v2 was .312419/.312220/.310282 respectively. Gamma zero indicator gains were negative in both models. These are engineering observations from one task, not evidence of a repeatable gain. Both native models captured most informative-mask signal here.

Gaussian observed nuisance values versus imputed constants may let a model recover masks indirectly. Test deliberate value/imputation collisions and compare preprocessing pathways before interpreting the native/indicator difference.

Collision development established a narrow deterministic preprocessing failure: TabICL mean-imputes native nuisance columns to constant zeros, then drops them. Its indicator gain at gamma .9 is .42735 nats (three-task exploratory 95% interval [.40787,.44682]). Gaussian gain is .00056 [-.00339,.00451]. TabPFN v2 does not show this failure; its collision indicator gain is -.00548 [-.00951,-.00145]. XGBoost native/indicator predictions match. These are development-only results under a deliberately degenerate control, not enough for a NeurIPS-quality final claim.

## Next work

1. TabPFN v3.5 access smoke failed: requires one-time license acceptance and a token from https://ux.priorlabs.ai. No substitute was made. Continue accessible models until user restores this access.
2. Run matched imputed/indicator controls and extend development to partial value collisions/quantization, interaction families and lower missingness. Preserve exact-zero and Gaussian matched controls. Do not rerun completed matrices without a concrete need.
3. Freeze confirmation hypotheses/configuration after development; keep seeds 60000–60019 unused until then. No hypothesis has been confirmed yet.
4. Implement ingestion/evaluation for the predeclared small real-covariate panel, with group/duplicate splits and separate native/imposed masks; evaluate cost before expanding.
5. Keep paper tied to saved evidence and compile source in native editor.

## Access and continuation

Modal CLI authenticated successfully using existing local credentials. BrowserOS Neo initially refused connections. Launched its installed BrowserClaw executable in the background, then connected to MCP successfully and verified a signed-in ChatGPT page. An independent critique was prepared in a research tab, but no assistant answer was obtained; the prompt stayed in the composer. Do not claim a completed Pro critique. Temporary RPC transport/debug files are outside the repository under workspace `tmp/`; reconnect through exposed BrowserOS tools on a later run. No email sent or manuscript submitted. Daily autonomous continuation is authorized and scheduled at 10:00 IST through October 20, automation ID `mira-research-sprint`. Repository code, reports, portable CPU reproduction and early manuscript were pushed to GitHub branch `codex/mira-research`. Raw GPU runs remain local and are ignored by Git.

The manuscript source `paper/main.tex` is opened in the native editor. Compilation failed with an environment diagnostic, `Unable to find standard directories for platform`; no source error was reported. No terminal TeX installation is available. Preserve the source and retry native compilation on a later session; do not claim a verified PDF.

Draft layout is standalone single-column, 10pt, 5.5-inch text width. Exact official NeurIPS style and checklist integration remain a final readiness gate; official 2026 template is https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip. Do not describe today's draft as submission ready.

Accounting correction: the first three smoke runtime JSON files label `cpu_vcpu: 2` and estimate one physical core. Modal `cpu=2` requests two physical cores. Their saved raw estimates are underestimates; use provider billing or the corrected two-core rate in budget.json. Reservations of $0.35 per smoke remain conservative. Future calls bound CPU and memory explicitly.
