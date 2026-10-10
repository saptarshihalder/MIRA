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
