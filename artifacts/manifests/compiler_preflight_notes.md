# Compiler freeze audit

The uncommitted preflight candidate (`compiler_preflight_aborted_v1.json`, SHA ceede16d0f3203949c3afbd2480a0761c1cb705e3a7f4f6cf2035a8bf110cf34) was never executed remotely or charged. A real Modal SDK import exposed duplicate entrypoint registration on the reused app. The final wrapper uses the separate `mira-mask-compiler` app, with the same pinned image, cached backbone weights and append-only global ledger.

The executable freeze is `configs/trained_compiler_validation_v1.json`, SHA 7d0f7719304763ce7d5d9d40a575aa6fdeaeffa7e47ac1ad9493b6f349624ed4. Eight runner/wrapper tests pass; the local real SDK import passes. Source/data/checkpoint hashes are rechecked before reservation. This freeze precedes both new GPU runs. The earlier preflight candidate is audit history, not an executable protocol.

The 512-task CPU development comparison was already known before this freeze; the GPU screen is development, not confirmation. All 324 planned predictions, controls and failures must be retained. Initial paid ceiling $1, total project ceiling $26, protected reproduction reserve $3; failures count and retries are disabled.
