# Lifted cavity networks: review code and score exports

This archive contains frozen model/training/scoring source, v2–v4 protocols,
preserved paper score arrays, closed-form reference arrays and cluster keys,
checkpoint identities, independent endpoint verification and the selected-cell
CPU reproduction report. Both passing and failing comparisons are included.
Source and panel revision identifiers in the manifest identify content.

Raw sensor observations, pickled panels, model weights, provider/account records,
the manuscript, its third-party style file and repository metadata are excluded.
JSON exports remove absolute host paths and normalize formatting; numeric arrays
are unchanged. Original and exported metadata hashes are recorded. This supports
auditing and regenerating the reported numbers, not independent training or full
inference reproduction. Automatic identity scans do not guarantee anonymity.

## Regenerate the paper's numeric material

Use Python 3.12 with torch, NumPy and matplotlib. `requirements-v4.txt` lists the
direct scientific runtime pins; plotting also requires matplotlib. The observed
93-distribution environment and GPU/driver record are retained in
`provenance/v4_environment.json` and `requirements-v4-observed.txt`. They were read
from the active continuation after source fitting began; they are not a launch
timestamp, wheel-hash lock or independently rebuilt environment. There is no
automatic dependency installation, network access, training or cloud launch.

```text
python reproduce_paper.py
python -m pytest tests/test_lifted_cavity.py tests/test_lifted_gauge.py -q
```

The first command verifies every exported payload hash, independently recomputes
all five E8–E12 endpoints from the 35 score archives and week identities, and creates a new
`regenerated` directory with macros, tables and figures. An existing output
directory is refused. The unit tests generate their own synthetic fixtures and
small checkpoints; they do not repeat the scientific comparison.

## Scientific scope

The v4 comparison uses three fixed source-training seeds and final checkpoints,
with nominal endpoint-wise intervals conditional on those fits. The real-data
target is log concentration during contemporaneous within-week adaptation.
Adjacent-week independence and training-randomness uncertainty are not resolved.
LCT is target-source-fine-tuned whereas the pretrained comparator is in context
only; that comparison cannot isolate architecture. Dequantization followed
inspection of target distributions and some closed-form results. Earlier real
transformer failures and the original factor-sign issue remain part of the
record. Content hashes do not provide independent execution timestamps.

## Data, pretrained weights and rights

Panel provenance records retain dataset attribution and hashes. Retrieve raw
data only from its attributed sources and observe their terms. Pretrained TabPFN
weights are not redistributed; their fixed identity is in the verification
report, and use requires their own applicable terms. Never deserialize untrusted
pickled artifacts as a verification step.

This is an authorized review export, not a grant of an open-source license.
No distribution or reuse license is invented for the authors' code or third-party
materials. Authors must select any public reuse terms and perform contextual
anonymity and attribution review before external submission.
