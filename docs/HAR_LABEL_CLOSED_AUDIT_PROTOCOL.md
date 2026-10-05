# HAR identity audit, label closed — October 5, 2026

Freeze this protocol and script before downloading. Official UCI dataset 240, DOI 10.24432/C54S4K, CC BY 4.0: Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013). Verify the original 30-person HAR archive; do not substitute the updated postural-transition dataset.

Download at most 90 MB from the official archive, timeout 90 seconds, no automatic retries. Hash and retain the archive locally, excluding it from Git. Read only filenames, README, feature schema, subject IDs and feature matrices. Do not open activity-label members or fit any model. Reading the archive's compressed bytes for transport/hash is not outcome parsing; record the distinction. Track exactly which members are opened.

Verify 7,352 train and 2,947 test rows, 561 features, integer subject IDs in 1–30, 21/9 subjects, no train/test subject intersection, finite feature values and identity-row alignment. Hash canonical float64 feature rows and report exact duplicate intersections within/across subject allocations. Audit subject IDs as grouping fields, never predictor features. Array alignment can be checked by row counts but cannot reconstruct undocumented temporal identities.

Allocate the original 21 training subjects by ascending SHA256 of `mira-har-source-v1:<subject>`: first 14 source-fit, remaining 7 source-development. Preserve all 9 original test subjects with labels closed. This allocates eligible subjects, not a full modeling/evaluation protocol. Whole-subject splits prevent sensor-window overlap across subject allocations under the documented collection process; they do not make windows within a person independent. Do not make same-subject support/query independence or clinical/general-population claims.

This dataset has no native missing values. Any later masking/cost experiment must be explicitly synthetic and freeze its masks, costs and task definition. No training or evaluation-label release follows automatically from eligibility. The new mechanism must first clear its novelty, source-development and matched-control gates. External scoring remains conditional; feature/identity inspection alone is not confirmation evidence.

Repository access history: registry and official metadata only before this audit according to saved records. Searches cannot exclude undocumented access or inclusion in pretraining. Record raw download as a new access event without changing the old registry's historical bytes.
