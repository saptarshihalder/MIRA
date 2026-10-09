# Protocol v4 launch audit — October 7, 2026

October9 latest execution amendment: see LIFTED_V4_TIME_AMENDMENT.md. L40S source smoke completed178.240s; its20,520.962s estimate fails the original15,900s cap. A separate350-minute main extension is frozen before fitting, with scientific settings unchanged. USD13.25 reserved; completed calls are covered by historical3.50 (gross3.03766887), plus3 protected, total19.75. Earlier running-state notes below are historical. Main launch state is recorded in STATUS.md and the compute ledger.

Imported code/protocols: lifted-cavity b614708c2959b49bdf60aff62058ab62597480a4; merge 8138ec6. Frozen panel branch: 799f473459c99e8decbb39ff9924c86a90cc8dc7. No v4 job had started (user confirmation).

## Operational corrections before execution

- Pinned code/data revisions; immutable source, panel and package fingerprints at the results root. Changed identities require a new root. Never adopt historical unbound checkpoints automatically.
- A Linux process-group wall-time limit, one session lock, an append-only runtime log, and zero automatic retries. Initial free Colab T4 session: at most 45 minutes for data preparation and source-only smoke. No paid purchase or Modal spending.
- Smoke uses seed90401 synthetic fixtures and seed90402 Beijing SOURCE episodes. No confirmation-panel scoring in smoke. No interim endpoint calculations after seed1. Checkpoints are hashed before scoring.
- Validate every expected metric, condition, task count and mask count; reject NaN/Inf and truncated arrays. Limited TabPFN outputs have a distinct partial namespace. Confirmation requires seeds1,2,3.
- No change to architectures, training recipe, endpoint margins, target transforms, or model selection. The separately frozen sign-averaging repair is not silently applied to these models.

The dequantization amendment followed inspection of target distributions and some closed-form outputs. Report this chronology explicitly; do not describe the entire v4 dataset as untouched before every design decision. Equal-week pooling is the prescribed real-data estimand; training seeds are averaged within episodes, not treated as independent datasets.

## Manuscript corrections

Limit scalar-expert impossibility to the theorem's mask-independent setting. Limit rank necessity to the generic-rank static linear-Gaussian assumptions. The context-conditioned scalar ablation is empirical evidence, not a consequence of Proposition1. Replaced the mirror style with the official NeurIPS2026 style; see paper/lifted_cavity/STYLE_PROVENANCE.md.

Native compilation failed before typesetting: Windows sandbox helper setup refresh error. Existing PDF is stale relative to these edits and its new layout is unverified. No venue-readiness claim.

## Validation and remaining work

Initial targeted suite: 14 tests passed in40.57 seconds (11 model tests,3 launch-validation tests). GPU smoke/throughput, TabPFN weight access, exact device resume, all v4 training and endpoints remain pending until their logs exist. T4 allocation is available in the active Colab account; its UI says no subscription and zero compute units. Free allocation is not proof of Pro access. Do not count engineering checks as scientific improvement.

## First T4 execution and density repair

First bounded GPU session ran382.798 seconds: both2.1M models completed500 updates, checkpoint recovery and20-step source fine-tuning. TabPFNv2 downloaded and produced predictions, but its density call failed because raw-unit bar borders were float64 while target/logits were float32. No confirmation panels were scored. Evidence: artifacts/reports/lifted_v4_launch/first_smoke_diagnostics.txt. Repair e1b1d9d casts density logits/targets to the criterion border dtype, retaining raw target units. Four launch guard tests now pass, including this regression. The free runtime later disconnected from inactivity. A manually initiated corrected source-only session on October8 has a separate Drive root and45-minute cap; this is not an automatic retry or efficacy result.

## Corrected GPU interface result - October8

PASS on TeslaT4, torch2.11.0+cu130: TabPFN9.1.0/v2,8estimators,random_state0. Four source tasks at16 sensors, seed90401;16 inference calls. Runtime24.028 seconds,8 validated arrays,456 finite NLL/squared-error cells. The saved zip was recovered locally and panel completeness/source SHA256 checked independently. No confirmation or real-data scoring. Drive remount failed, so this isolated probe used temporary storage; it did not repeat learned-model training. Full v4 remains pending. Before external scoring, also retain the actual pretrained weight identity, not only its package/model-version name.

## October 9: A100 outcome and L40S source-only timing probe

A100 source smoke PASS: 179.694 seconds. Its main execution estimate was 45,329.76 seconds, exceeding the fixed 15,900-second child cap; the throughput gate FAILS. Full v4 main has NOT launched. The original bootstrap failure remains recorded; the successful A100 source smoke does not erase it.

The separately identified L40S source-only smoke is running: call `fc-01M4GVXA5GHNNP9XXH3SYM99GY`, app `ap-cLaTyuFLt3p4wRYFiWI2eJ`, $1 reserved. It adds a bounded scoring profile of 128 repeated tasks made from the four original source fixtures. This measures scoring startup/throughput, not independent efficacy. Frozen source e1b1d9d, panel commit 799f473, models, recipe and margins are unchanged; no automatic retries. Main remains conditional on successful smoke with the same immutable identity and a conservative estimate fitting 265 minutes.

Actual downloaded TabPFN v2 weight SHA256: `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Historical gross $2.82766545 is within the retained $3.50 provision. Historical provision + new smoke $1 + conditional main $11.50 + protected reproduction $3 = $19 under the $20 cap; provisions are not final invoices.

The native compiler still fails. Official portable Tectonic compiles the existing lifted paper to nine main pages/28 total; fifteen targeted tests pass and 854 numeric macros are unchanged. These are engineering/document checks, with no new v4 efficacy or publication-readiness claim.
