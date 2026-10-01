# Day 1–3 evidence readiness — 1 October 2026

**Final execution assessment; appended results supersede the provisional pending items below.** The original engineering gates are substantially met for accessible TabPFN v2/TabICL v2. There is a specific preprocessing finding; no final scientific hypothesis has been confirmed. Continue the mechanism study and defer substantial adapter training.

| Gate | Evidence on disk | Assessment |
|---|---|---|
| Preserve/reproduce CPU pilot | `cpu_audit.md`: 31,680 raw rows, 24 summary cells; maximum summary difference 2.22e-16; checks of normalization, posterior, and predictor information access | Pass within recorded core dependency versions; original hardware/BLAS unknown |
| Execute actual TFM smoke | Successful native/actual/shuffled inference at gamma 0/.8 for two frozen backbones; environments, checkpoint hashes, predictions saved | Pass for named accessible versions; TabPFN 3.5 remains access-blocked |
| Run development matrix | Gaussian and exact-zero runs each contain 135 cells: 3 seeds × 5 gammas × 3 models × 3 modes; summaries regenerated from saved probabilities | Pass for one label-associated mechanism and default context/missingness settings |
| Verify the representation intervention | `preprocessing_audit.md`: all 15 Gaussian/collision pairs match U, labels, masks, oracle, base, row IDs; exact released TabICL source inspected | Pass for the deliberate construction, not general prior causality |
| Quantify effects honestly | Paired task-level effects and exploratory intervals recorded; raw probabilities retained | Development evidence only; three independent task seeds per gamma are a small uncertainty sample |
| Measure throughput/cost | Runtime records and budget ledger exist; status reports provider charges and conservative reservations separately | Pass for sizing further development; provider billing may lag |
| Decide scientific branch | `decision.md` chooses mechanism study and defers adapter; confirmation seeds 60000–60019 reserved | Pass as a development decision; final scope still needs the pending controls |

The strongest current evidence is version-specific: TabICL 2.2.0's numeric path mean-imputes observed zeros and NaNs identically, then removes the constant nuisance columns. Its collision/native predictions are bit-identical across gamma within each seed. At gamma .9, its native-minus-indicator effect is .42735 nats versus .00056 for Gaussian values. TabPFN v2 has no corresponding failure; XGBoost native/actual-indicator predictions match. This supports preprocessing localization under exact collisions, not a claim that all TFMs ignore masks.

The quoted intervals use only three task seeds and multiple exploratory comparisons without multiplicity correction. Do not interpret 135 cells, 1,024 query rows, gamma sweeps, or the two matched distributions as 135 independent experimental tasks. The full-oracle reference is privileged; compare neural headroom against equally informed simple learners before activating a model branch.

Before freezing confirmation, complete matched externally imputed controls, test partial/near-value collisions and lower missingness, and inspect actual-versus-shuffled differences. Preserve null results and the same-information/feature-width distinction. The exact-zero example is intentionally degenerate; a broadly useful paper needs evidence beyond that case and a specific contribution beyond standard indicator advice. The six-dataset panel is predeclared but unexecuted. A current-backbone claim must remain limited while 3.5 access is blocked.

Independent Pro critique was attempted but not obtained (`pro_critique.md`). Final audit should incorporate the pending controls, test results, final configuration freeze, and updated ledger before marking Day 3 complete. NeurIPS-quality manuscript readiness additionally requires untouched confirmation, real-covariate evidence, completed write-up, and verified compilation; it is not established by this engineering gate.

## Bibliography metadata check

Checked all seven entries in `paper/main.tex` against their linked primary records; manuscript left unchanged by this audit. The correction identified was replacing the corporate author `Prior Labs` with **B. Jäger et al.** and using the exact title **TabPFN-3.5: Technical Report** ([arXiv record](https://arxiv.org/abs/2609.17895)); the root agent reports applying it. Retain `v1` when citing the inspected HTML version; the current abstract record is v2.

The remaining abbreviated author entries, titles, venues/years, and identifiers match:

- [TabICLv2](https://proceedings.mlr.press/v306/qu26a.html): Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan; ICML 2026, PMLR 306:102477–102555.
- [DAMS](https://proceedings.mlr.press/v206/zhou23b.html): **Helen Zhou, Sivaraman Balakrishnan, Zachary Lipton**; AISTATS 2023, PMLR 206:9577–9606. `H. Zhou` is correct.
- [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484): Patrick Rockenschaub et al.; arXiv:2406.16484, 2024.
- [TabPFN Nature paper](https://www.nature.com/articles/s41586-024-08328-6): Noah Hollmann et al.; Nature **637:319–326**, 2025. Adding volume/pages would complete the otherwise correct entry.
- [Structural Missingness](https://arxiv.org/abs/2601.18500): **Chen Liang** et al.; arXiv:2601.18500, 2026. `C. Liang` is correct. If expanded, the current record spells collaborators **Yutong Zhao** and **Shenghang Zhou**; do not inherit differing search-index spellings.
- [TFM-Retouche](https://arxiv.org/abs/2605.06047): **Duong Nguyen, Mohammed Jawhar, Nicolas Chesneau**; arXiv:2605.06047, 2026. The manuscript's v1 citation matches the inspected version; v2 also exists.

## Completed imputation controls; other Day 3 jobs pending

Both new reports verify 108/108 cells: `day3_gaussian_imputed_controls` and `day3_zero_collision_imputed_controls`. Each uses three development task seeds, gamma 0/.5/.9, three models, and four representations. At gamma .9, the exact-zero **externally imputed-to-indicator** gains are:

| Model | Gain, nats | Exploratory 95% paired interval |
|---|---:|---|
| TabICL v2 | .450211 | [.430579, .469843] |
| TabPFN v2 | .446769 | [.434278, .459260] |
| XGBoost | .420492 | [.395389, .445594] |

These large effects demonstrate standard information loss when explicit imputation collapses observed zeros and missing values. They are not evidence that all three models lose masks on their native input paths. Native TabPFN/XGBoost already handle missingness on this construction; TabICL's native preprocessing makes that same collapse internally. Independent comparison of saved probabilities found native and externally imputed TabICL predictions bit-identical in all nine exact-zero task pairs. Gaussian pairs differ slightly (maximum probability delta 8.68e-5), so do not assert exact equality there.

The Gaussian gamma-.9 imputation gains are small: .003648 for TabICL, .003951 for TabPFN, and -.018290 for XGBoost, each with an interval containing zero. Gamma-zero exact-zero XGBoost indicators hurt by .083500 nats; a blanket recommendation to append indicators is unsupported. Three-task intervals remain development evidence.

At gamma .9, unrelated-column stress effects have intervals containing zero for all three models. The manifests append **16 independent columns**, whereas actual/shuffled masks in the original matrices append eight. This stress control therefore does not itself isolate identical-width effects; retain the original same-width shuffled-mask comparison and avoid claiming feature-count explanations have been universally excluded.

Await the partial collisions, paired quantization, interaction/low-missingness controls, final tests, and updated ledger before the final Day 3 assessment. Confirmation remains untouched.

## Expanded development interpretation; low-missingness run pending

The interaction report verifies **324 cells across four families**, including the label-only control. At gamma .9, TabPFN v2's native-to-indicator effect is .076537 nats for pairwise masks (exploratory interval [.045546, .107528]) and .031594 for value-dependent masks ([.009325, .053862]). The corresponding **shuffled-to-actual-indicator** effects are .074358 ([.034854, .113863]) and .016489 ([.000461, .032516]). Pairwise performance therefore offers the clearest development candidate beyond the matched-width shuffled control. The value-dependent shuffled contrast is smaller and its lower bound is close to zero. These are selected exploratory findings, not confirmed hypotheses or evidence of prior causality.

Sparse-pair native effects have large means but very imprecise three-task intervals: TabPFN .340151 [−.040183, .720484], TabICL .253719 [−.287700, .795138]. Do not rank these as demonstrated gains. TabICL pairwise/value-dependent effects also include zero; native/actual-indicator XGBoost outputs match. Evidence remains model- and family-dependent.

The paired partial-zero variants do not reproduce the exact-zero TabICL failure. At gamma .9, TabICL native effects are .000658 for approximately 10% observed zeros and .001107 for approximately 50%, both with intervals containing zero. Quantized values yield a small .003407 native effect ([.002450, .004365]), but the same-width shuffled contrast is .002175 [−.003202, .007552]. Thus this quantized result has not established a gain attributable specifically to true masks beyond that control.

Use **partial zero point masses** and **quantized values** as the scientific descriptions. The data audit finds no finite observed values exactly equal to fitted support means in the partial-zero runs; the quantized equality fraction is approximately 2.9%. A raw zero is not automatically an imputation collision. These figures use the auditor's support-only mean reference and do not instrument each third-party encoder's arithmetic. Legacy run-directory names containing `collision` should not determine manuscript terminology.

The saved-data audit confirms the Gaussian/exact-zero pair and all three partial/quantized pairings on U, labels, masks, oracle/base, and row IDs. Root reports 60 tests passing. Low-missingness evidence, the final ledger/configuration freeze, and final audit are still pending. The adapter remains unjustified by these representation findings: they do not show headroom beyond equally informed simple learners. Reserved confirmation seeds remain unused.

## Final execution assessment

All seven additional runs are complete: 1,026 new cells, 1,296 total development cells (12 smoke cells excluded). All 60 local tests pass. The executable saved-data audit passed 7/7 runs with zero errors, matching source/environment identities and exact paired observable/target/ID arrays. Reports were regenerated from saved predictions. No confirmation seeds were used.

The low-rate panel adds 243 cells. At gamma .9, TabPFN single-mask gain is .03329 [.00697,.05960], pairwise approximately zero, and value-dependent .03935 [.02040,.05831]; TabICL value-dependent .04167 [.00894,.07439]. Baseline rate changes oracle headroom; these are exploratory three-task intervals. Complete condition tables are in day3_summary.md/json.

**Day 1–3 execution complete for accessible checkpoints.** Freeze an interaction-focused confirmation design next. Adapter gate remains inactive; paper-quality readiness is pending independent confirmation, real-covariate evaluation, current-checkpoint access assessment, official formatting and verified compilation. Reservations $5.55 retain the $3 reproduction reserve within $26; provider-reported app costs $0.22198885 may lag. No verified Pro critique was obtained. An agent usage limit prevented another final agent turn; root ran the saved audit directly.
