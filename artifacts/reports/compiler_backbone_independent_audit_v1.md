# Independent compiler backbone audit

TabPFN v2 CPU: pass. All162 predictions cover18 episodes ×9 views; no saved errors/warnings. Independently checked every probability/data SHA, probability range/length, empirical NLL/Brier,108 synthetic expected NLLs, frozen source/data/selector hashes, checkpoint SHA and dependency lock. Recalculated score drift: 0.0.

Canonical rows, source groups, mask/NaN patterns, selected columns and all episode arrays match reconstructed frozen inputs. Nonfloating arrays match exactly; floating tolerance is1e-12, largest drift 0.0. All162 saved coordinate maps are invertible.

Trained/heuristic maps match in 17/18 episodes; probabilities match exactly in 17/18. Three-seed order-averaged identity-minus-trained expected NLL means: gamma0 0.00000000, gamma.9 0.08587015. Negative effects are retained in JSON.

TabICL GPU audit pending. This is development, with three synthetic seed units and two descriptive natural datasets. CPU/GPU equivalence, confirmation, causal pretraining intervention, novelty and venue readiness are not established. No paid calls or source edits.
