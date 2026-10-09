# Lifted cavity supplementary code — incomplete local draft

This code-only draft contains pinned model, generator, training, evaluation and
endpoint code; two model test modules; frozen v2–v4 and source-only gauge protocols;
the execution-time amendment; and panel hashes/recorded attribution. The gauge
repair is a separate source-only diagnostic, not the frozen v4 model. Scientific
source files are copied byte-for-byte from revision
`e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9`; panel metadata comes from
`799f473459c99e8decbb39ff9924c86a90cc8dc7`. Revision IDs identify content only.

No results, datasets, panels, weights, manuscript/style files, notebooks, provider
launchers, Git directory, author records or private handoff materials are included.
This is neither complete reproduction nor a submission-ready/public release.

## Build and verify locally

Use Python 3.10+ and Git with both frozen revisions already available in the
research checkout. The builder uses only the Python standard library, makes no
network requests and does not import scientific code or deserialize artifacts.

```text
python infra/build_lifted_supplement.py --self-test
python infra/build_lifted_supplement.py --out lifted_supplement_draft.zip
python infra/build_lifted_supplement.py --verify lifted_supplement_draft.zip
```

The destination directory must exist; existing archives are never overwritten.
The explicit allowlist rejects links, binary/large files and common personal-path,
email and credential patterns. ZIP ordering, timestamps and permissions are fixed;
identical inputs produce identical bytes. `MANIFEST.json` records each payload's
size and SHA-256; the command prints the archive hash. Verification reads bounded
text without extraction or execution. Hashes detect accidental changes, not
authenticity when an archive and its manifest are both replaced. Automated scans
do not establish anonymity; contextual manual review remains necessary.

## Limited code checks

In a fresh environment, Python 3.12 matches the declared v4 execution runtime.
`requirements-v4.txt` records its direct package pins, not every transitive package,
CUDA/driver or test-tool version. Installing dependencies requires network access
and is separate from packaging.

```text
python -m pip install -r requirements-v4.txt
python -m pytest tests/test_lifted_cavity.py tests/test_lifted_gauge.py -q
```

These tests generate synthetic inputs locally; the resume test creates and reads
its own checkpoints. Passing tests establishes only the tested model contracts.
For recipes, read the frozen protocol files and `colab_v4.py`; do not run the full
driver as a package check. V4 confirmation requires all three seeds and all frozen
comparisons. The time amendment changes execution limits only. GPU execution is
not promised to be bitwise reproducible.

## Open release requirements

Final scores, complete checkpoints, an environment/driver record, raw-data
retrieval verification, independent full reproduction and manuscript consistency
remain outside this draft. The panel provenance files record dataset attribution
and hashes; no panel is loaded by the builder. Existing scientific readers use
`torch.load(..., weights_only=False)` for panels/resume state. Such files can execute
pickle code: only use locally generated or independently trusted artifacts after
hash/provenance checks; never open unknown downloaded checkpoints as validation.

No repository or supplied handoff distribution license was found. User-provided
attachments do not grant redistribution rights; no license is invented here.
Third-party manuscript/style files, pretrained weights and raw data are excluded.
Recorded dataset licenses do not license this code. Rights-holder approval of code
redistribution and a manual identity/attribution audit are required before any
external submission or public release. Preserve required third-party attribution
when finalizing the package.
