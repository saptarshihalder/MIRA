# Trained MIRA-Compiler CPU prototype

6092 learned parameters over a fixed17-dimensional permutation-invariant moment encoding. Trained on2048 synthetic tasks for80 fixed epochs;256 validation and256 held-out development tasks. Synthetic active/gamma teacher is privileged pretraining supervision, excluded from target-task inference. No TFM called and no paid compute.

Teacher basis recovery: raw82.42%, confidence-fallback78.12%, fixed moment heuristic78.12%, identity46.09%. These are teacher classification metrics, not query log-loss or calibrated safety. Fallback ties the heuristic; a useful model advantage remains unestablished. Allseeds are now used development. Failed Windows dtype attempt was retained separately under trained_compiler_v0; fixed int64 targets produced this checkpoint.

One additional test verifies support-row permutation invariance, restricted input type and forced identity fallback/invertibility. Checkpoint reloading reproduces held-out probabilities within numerical tolerance. Config/checkpoint/source identities, weights, predictions and package lock are saved. Neural architecture is a baseline prototype, not established methodological novelty.

Next: train equally informed baselines with matched synthetic pretraining budgets; test selected maps through frozen backbones using actual query log-loss; record uncertainty/no-signal harms; freeze fresh confirmation only if development passes. No broad empirical or venue-acceptance claim.
