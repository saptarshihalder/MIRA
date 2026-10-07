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
