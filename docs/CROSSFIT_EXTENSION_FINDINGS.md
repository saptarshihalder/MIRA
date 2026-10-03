# Architecture attribution and transfer: October 4

Protocol/source commit f10bc89 preceded all new training/evaluation. One new support-only-head control trained for 500 CPU updates with 370 parameters, identical initialization, episodes, batches, optimizer, loss and verification rule. Its shared gate uses fitting-support averages; verification/query inputs cannot influence that gate. Four boundary/gradient tests pass. Checkpoint bc178a243eb9da294358a22d387f828a39dbd6c71f60315d7b491d689ed30e78. Existing main and other checkpoints remain fixed. No cloud calls or cost.

## Fixed independent synthetic development

20 new seeds 700001-700020, two widths, four regimes: 160 cells. Sign-flip main gain over capacity-matched support-only head is **.014318 [.002958,.025678]**, with 97.5% paired t interval as specified for the two primary comparisons. This replicates the used-panel .014631 [.004557,.024704] (95%) contrast and supports the narrow query-conditioning hypothesis. Parameter count and training exposures are matched; CPU/GPU execution differs. However gain over the task-conditioned shared filter is **.004074 [-.001861,.010009]** (97.5%), so the joint architecture gate FAILS. These are development results, not final confirmation; the family correction covers the two declared sign-flip contrasts, not every exploratory comparison or candidate tried historically.

Robustness gate PASSES again. Sign-flip gain over frozen .052614 [.039142,.066086] (95%); gain over guarded logistic .031557 [.019827,.043288]. No-shift gain .048559 [.035687,.061430]. Ignorable harm is now .000237, upper95 .000733 (below .001), rather than zero: verification is fallible. Nonlinear gain .000106 [-.000116,.000327]. Full-support logistic remains more accurate on sign flips, so universal superiority is false. Confirmation98000-98019 remains untouched.

## Native transfer: failed descriptive gate

Apply all frozen checkpoints to all78 saved development episodes: three previous seeds across every retained BRFSS asthma/diabetes and road KSI group. Same source128/target128 labels and frozen CUDA-trained XGBoost. Four source-selected standardized native features and exact masks recovered from all15 parity codes; source-fit/context/support/query identities checked disjoint. No native fine-tuning or feature reselection. Width4 and absence of synthetic always-observed anchors are extrapolations beyond training widths6/10.

| Task | Frozen NLL | Main | Support-only head | Target Platt |
|---|---:|---:|---:|---:|
| BRFSS asthma | .395842 | .396002 | .395842 | .398324 |
| BRFSS diabetes | .333466 | .333550 | .333527 | .335693 |
| Road KSI | .611319 | .610819 | .609267 | .603197 |

Main slightly harms both BRFSS endpoints and improves roads by .000500, where support-only gating and ordinary calibration do better. Its accepted branch counts are2/60,1/60 and2/36, respectively. Rejection of most updates is not evidence of useful native adaptation. Means average groups/seeds within task; seeds share rows/groups and BRFSS endpoints share the survey, so no independent dataset interval is claimed. Historical global calibration/linear controls use extra meta-training labels and are disclosed as stronger, unmatched-label references. All baseline results remain in native.json.

## Audit and reproduction

100 frozen file hashes,2,454 probability arrays,1,190 checkpoint replays and all recorded decisions pass; maximum independent score/replay error0. All238 extension cells and160 original-panel support-control predictions are retained. Exact native public-data input files are archived in original_native_inputs.zip; the zip contains repository-relative paths. Extract into the repository for full source-boundary reproduction; do not overwrite unrelated inputs. Independent audit: `python experiments/crossfit_v1/audit_extension.py` with NumPy, SciPy, scikit-learn and PyTorch CPU. Original training/generator sources and hash manifests remain fixed. Runner refuses to overwrite an existing result directory. Training traces/checkpoints persist before evaluation.

## Decision and next evidence gate

A narrow trained query-conditioning contribution now has replicated synthetic evidence. Joint architecture superiority and real-data utility are NOT established; JMLR/NeurIPS readiness remains unproven. Close this fixed batch. Do not extend seeds, tune the verification threshold on these queries or repeat the same frozen transfer hoping for a win.

The specific next hypothesis is whether source-only meta-training on heterogeneous native missingness episodes removes the synthetic-to-native mismatch. Before any training: freeze dataset-disjoint development/confirmation cohorts, a label-balanced representation, matched query-head/shared-head/calibration training, three initialization seeds, one finite compute bound, and full regret/coverage reporting. The current native panel is used development; no retrospective designation as confirmation. Retain the current fixed-checkpoint results as external failure evidence. This proposal is unexecuted, needs appropriate unused data and a defensible protocol, and is not a safety or novelty guarantee. Reconcile provider charges before another paid phase; provisions16.45 plus protected3 leave.55 under20.
