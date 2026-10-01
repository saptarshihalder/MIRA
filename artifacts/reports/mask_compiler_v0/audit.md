# Independent reviews

Three agents reviewed theory, baseline/leakage and the sprint design, without paid calls.

Theory: invertibility preserves information but not arbitrary marginals; oracle-aligned maps use unavailable active-set knowledge; restricted-logit feasibility is not TFM efficacy or causal pretraining attribution. All 16 masks and exact sign/oracle signal checks pass.

Executable audit: all 480 saved probability vectors valid; allprediction SHA 256s match; independently recomputed expectedNLLs agree exactly. Three compilerCV reconstructions agree exactly. No query/oracle selection leak found. Search costs 109fits/compiler versus 10fits/single-basis baseline. Full-parity L1/L2 are required controls. Different feature widths/search budgets and nonnested contexts are disclosed.

Executed sparse liblinear used no solver seed; independent CV replays differed up to 7.34e-5. CV clipping conventions also differed, without effect in sampled cases. Preserve outputs and executed-source snapshots. Future baseline code uses an explicit solver seed and shared clipping; new results require a new output directory. Fractional parity-input validation was repaired after execution without changing valid binary computation; executed source SHA/snapshot retained. Two targeted tests passed.

Plan: novelty is unestablished; parity/feature engineering precedents matter. Draft paid pilot requires committed runnable freeze, nested support-only selection guard, matched-search baselines and $1 total ceiling. Fresh confirmation, contemporary access and external/natural-missingness evaluation remain pending. No acceptance guarantee.
