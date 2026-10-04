# Real-observation acquisition benchmark â€” October 4, 2026

The frozen trained-model benchmark fails both development gates. It establishes useful acquisition relative to stopping in these settings, but does not establish a new architecture or an advantage over the strongest controls. Code and eight input hashes were committed before fitting. No protocol or model was changed after scoring.

| Model / policy | Red risk + cost | White risk + cost |
|---|---:|---:|
| Masked-label mixture, depth two | .609431 | .591657 |
| Joint-likelihood mixture, depth two | **.607092** | .587787 |
| Masked-label mixture, greedy | .609431 | .591657 |
| Masked-label mixture, static subset | .612446 | .598382 |
| Masked-label mixture, stop | .687677 | .636958 |
| Masked tree, depth two | .634386 | **.584910** |
| Masked tree, greedy | .627860 | .585041 |
| Masked tree, static subset | .611853 | .590635 |

Entries average three initializations and twelve fixed price/availability scenarios. Lower is better. The masked-label mixture improves over its no-acquisition control by 11.38%/7.11%, but loses to the best control by .002339/.006748. Its depth-two and greedy means coincide. The prespecified .005 superiority margin and per-seed requirements both fail. This does not identify why lookahead failed, prove acquisition is unnecessary, or justify retuning these scenarios.

## What was actually trained and measured

UCI Wine Quality contains 6,497 rows and 5,318 exact feature-vector groups across red/white domains. Global hashing holds exact duplicates, including two cross-color groups, in one partition. Fit/validation/development counts are 978/303/318 red and 2911/996/991 white. Producer/batch identities are unavailable. Fit-only terciles quantize the eleven measurements; quality >=6 is the binary label. This is one dataset family with dependent domains and repeated scenario evaluations, not independent external confirmation.

Twelve 1,088-parameter categorical-mixture checkpoints receive 500 gradient updates each; six masked histogram-boosting models receive 100 iterations each. Validation calibrates predictions on a fixed temperature grid. All eighteen models, temperatures and state predictions are hash-locked before any development scoring. The 576 score cells use actual held-out rows, with paths revealing only requested values and at most two features. Both unavailable and already measured actions are forbidden. Tree policies use their own calibrated entropy and shared learned feature transitions, avoiding the unfair comparison of tree predictions against the mixture's label posterior. Training compute is recorded separately; tree and mixture budgets are not identical.

Acquisition costs and independent column billing are simulated. No clinical, causal, physical assay-saving or deployment claim follows. The comparison has not reproduced BRiG-AFA, NM-PPG, L2M or other current systems. It cannot establish state-of-the-art superiority even if its local gate had passed. The model is an established mixture with a masked-label objective; no novelty is claimed.

The complete bounded run took 23.516 CPU seconds; cloud calls and charges were zero. Existing $16.45 provisions and the $3 reproduction reserve remain unchanged under the $20 cap. Preserve all checkpoints and negative comparisons. This pilot is closed; further model work requires a substantive hypothesis rather than more depth or renamed planning components.

## Reproduction

Attribution: Cortez, Cerdeira, Almeida, Matos and Reis (2009), [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), DOI 10.24432/C56S3T, CC BY 4.0. `artifacts/manifests/wine_acquisition_data_audit.json` pins official raw downloads and split rules.

Use Python 3.11.9, NumPy 2.3.5, Torch 2.8.0+cpu, scikit-learn 1.8.0 and SciPy 1.17.0; training and audit use one CPU thread. From the repository, `python experiments/wine_acquisition_restore_inputs.py` restores missing official inputs and verifies reconstructed NPZ byte hashes without modifying the frozen audit. `python experiments/wine_acquisition_audit.py` replays saved evidence. The original runner refuses to overwrite its output directory; use a separate checkout/output directory for an intentional refit and compare against the saved artifacts. Fresh-checkout network restoration has not been exercised; local reconstruction has matched both frozen NPZ hashes.

Replay audit passed: all 576 cells (376,992 repeated row-score uses), eighteen fitted models, 529 state predictions per model, every policy decision and calibration choice, raw grouping and fit-only bin thresholds. Eight frozen input and thirty pre-scoring artifact hashes match. Full replay took 90.907 CPU seconds. Supplementary runtime/CSV manifest handling was repaired in the auditor only; no scientific rerun or source/model change occurred. Audit provenance distinguishes the full replay from the subsequent sixteen-summary-value check.
