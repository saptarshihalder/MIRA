# Complete-result integration

The continuation must finish with exit 0 before collection. No partial contrasts
or score-based recipe changes are permitted. The prepared software tests use
fabricated inputs and do not establish actual scientific reproduction.

From the repository, use the Modal environment for collection:

```powershell
& '../.venv-modal/Scripts/python.exe' infra/collect_lifted_v4_results.py --collect
```

Use the compiler CPU environment for the remaining commands:

```powershell
& '../.venv-compiler-cpu/Scripts/python.exe' infra/verify_lifted_v4_results.py --root artifacts/runs/lifted_v4_final --panels artifacts/inputs/lifted_v4_panels
& '../.venv-compiler-cpu/Scripts/python.exe' infra/replay_lifted_v4_predictions.py --root artifacts/runs/lifted_v4_final --panels artifacts/inputs/lifted_v4_panels
& '../.venv-compiler-cpu/Scripts/python.exe' infra/integrate_lifted_v4_verified.py --root artifacts/runs/lifted_v4_final --panels artifacts/inputs/lifted_v4_panels
```

Inspect the new stage manifest and complete verification/replay reports, then:

```powershell
& '../.venv-compiler-cpu/Scripts/python.exe' infra/integrate_lifted_v4_verified.py --root artifacts/runs/lifted_v4_final --panels artifacts/inputs/lifted_v4_panels --merge
```

Integration re-runs independent full verification and requires the fixed CPU
replay: all 24 checkpoints, 30 panel/checkpoint pairs and 408 unique arrays, with
unchanged input hashes and the originally fixed tolerances. It exports the five
verified panels' reference arrays and cluster keys for paper regeneration. No
model weights or raw observations enter the tracked paper-report directory.
Existing differing files abort the merge before any target write; identical
files may be reused. Unexpected interruption can leave an incomplete new stage
or partial merge; preserve it and investigate rather than overwriting evidence.

TabPFN metadata is named `train_v4.json` to preserve any earlier comparator
metadata. The paper generator prefers this version for v4 hardware/package
macros. All endpoint outcomes, including failures, must be reported. Regenerate
tables and numbers only after this complete-result barrier. Rebuild and visually
review the same manuscript source, audit new numerical claims, then create the
review supplement and verify its inventory and anonymity separately.

After successful replay, also run
`infra/summarize_lifted_v4_seed_sensitivity.py` with the same `--root` and
`--panels`. Report all three point gains for every endpoint. This is an added
descriptive audit, not a replacement for the fixed aggregate decisions;
see LIFTED_V4_SEED_AUDIT_PLAN.md. The review package requires its hash binding.

The research report metadata preserves provenance and may contain historical
machine paths. It is not automatically an anonymous submission supplement.
Hashes bind local files and checks; they do not supply independent cryptographic
execution timestamps, prove benchmark independence or reproduce training.

## October 11 one-shot completion worker

infra/finish_lifted_v4_evidence.py is already running locally, waiting on the existing read-only GPU observer. The hash-frozen plan is artifacts/manifests/lifted_v4_completion_worker_plan_20261011.json. Inspect artifacts/runs/lifted_v4_delivery/completion_result.json and worker_started.json before attempting any stage; do not duplicate it. It makes no GPU launch, recipe change, automatic retry, manuscript edit or Git push. Failed-upstream and duplicate-start guards were exercised with temporary inputs; actual scientific completion remains pending.

On complete upstream exit0 it executes collection, full independent verification, fixed selected-cell replay, staging, explicit conflict-preserving merge, all-seed descriptive audit, review export and extracted regeneration/tests. All tools are checked against recorded hashes before each child. Failed stages preserve outputs and stop; they must not be bypassed. A successful result is evidence-ready-manuscript-pending, requiring interpretation of every endpoint, same-source manuscript regeneration/compilation/layout review, final billing reconciliation and authorized Git publication.
