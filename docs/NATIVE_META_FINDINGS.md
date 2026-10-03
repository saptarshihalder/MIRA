# Native episodic family transfer: completed October 4

The authorized next direction was executed on real native data. Preparation protocol commit8fb0b76 and a subsequent140-file source/input freeze preceded fitting. Raw replay checks all122 prepared files: feature error0, maximum CPU/GPU logit difference6.68e-6 and training frozen probability difference6.56e-7. Historical CUDA-trained XGBoost3.4.1 backbones remain fixed; Python3.12 replayed them on CPU. The new adapter training runtime is PyTorch2.8 CPU, with no new cloud calls, backbone refits or charges.

Train on both BRFSS endpoints and develop on roads, then reverse. Both BRFSS endpoints stay together because they share respondents.96 real2024 source episodes supply128 support/512 outer-training query labels each; target2025 queries are scoring-only. Four original source-selected naturally incomplete features preserve actual masks; no synthetic outcomes or missingness. Each held-out target still has its historical50,000-label source backbone,128 source-context labels and128 target support labels: dataset-family separation applies to learned adapter weights, not to all target-domain supervision. All26 used2025 groups from split171001 are evaluated with three initialization seeds. Three370-parameter models,500 updates each, two directions and three seeds produce18 checkpoints and78cells. Save traces/checkpoints before scoring; no selection or retries.

## Result: native development gate fails

| Task | Frozen NLL | Query model | Shared-mode control | Support-only head | Target Platt |
|---|---:|---:|---:|---:|---:|
| brfss_asthma3 | 0.393272 | 0.393272 | 0.393272 | 0.393272 | 0.394279 |
| brfss_diabete4 | 0.332329 | 0.332329 | 0.332329 | 0.332329 | 0.333962 |
| road_ksi | 0.613194 | 0.613031 | 0.613115 | 0.613034 | 0.604898 |

BRFSS predictions return the frozen model for every initialization; main acceptance0/60branches on each endpoint. Sub-nanoscopic frozen/query NLL differences are numerical precision, not improvement. Roads gain over frozen is only.000163, versus.000084 over shared-mode and.000003 over support-only, below the fixed.001 gate. Main roads NLL by initialization:.613033/.613024/.613038; its9/36 branch acceptances have no demonstrated nonlinear-specific utility. Target Platt is substantially better at.604898. All three initializations are retained; none is selected.

Means first average the three equally weighted initializations and then groups within each task (balanced grid). Initializations/groups are not independent datasets; BRFSS endpoints share one family. No fabricated dataset-level interval or fresh-confirmation claim. Historical within-task global calibration/linear references have additional target-family native meta-query exposure and are explicitly unmatched references. Strongest road references and all regressions remain in report.json.

## Guard diagnosis and scientific decision

A post-development diagnostic scores fixed unguarded candidates on the SAME used query labels, without changing the deployed rule. Main unguarded NLL is.395003 vs frozen.393272 on asthma and.333998 vs.332329 on diabetes: rejection prevents harm. Roads unguarded main.612886 nearly equals support-only.612889, while Platt.604898 remains much better. Removing verification or changing its threshold is not a justified fix. Native source-family training corrects the earlier synthetic-width mismatch but still does not establish useful cross-family neural transfer. This narrows the failure mechanism without proving every native architecture must fail.

Close this finite pilot. Preserve the original replicated synthetic query-head evidence as scoped positive evidence; do not promote this candidate or claim JMLR/NeurIPS readiness. Before another trained architecture, use an UNUSED dataset/cohort to establish conditional residual headroom over calibrated tree/linear predictors, then freeze native meta-training and matched controls around that real failure mode. Do not spend more GPU budget on equivalent closed variants or use any examined query labels for selection/confirmation. Synthetic confirmation98000-98019 remains untouched. Candidate UCI unused-patient claims require an access/boundary audit first and are not treated as confirmed unused data.

## Audit, cost and reproduction

Independent auditor verifies all18checkpoint/78cell identities, all scores and controls, exact checkpoint predictions/decisions and the failed gate; maximum saved-score, checkpoint and control error0. Raw IDs, source feature selection and family separation are audited. All weights, training traces, native inputs, prediction probabilities, individual initialization outcomes and guard diagnostics are tracked. See docs/NATIVE_META_REPRODUCTION.md and native_audit.py (--folder supports separate reproduced outputs). Auditing source was recorded separately after the training freeze; it is not falsely marked pretraining-frozen. Total training/evaluation372.36 CPU seconds. A Python3.11 package-version mismatch was resolved by using existing Python3.12 for preparation; no cloud call failed or retried. Provisions16.45 plus3 protected leave.55 under20; provider final invoices remain unverified. Same manuscript/source editor updated; PDF compiler environment remains unresolved.
