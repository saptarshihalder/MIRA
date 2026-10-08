# Protocol v4 launch audit — October 7, 2026

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
