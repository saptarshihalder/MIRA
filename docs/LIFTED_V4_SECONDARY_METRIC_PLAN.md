# V4 descriptive accuracy and calibration audit — October 11

Declared before complete local v4 contrasts are available, after panels and some checkpoints exist. This is an added descriptive audit, not preregistration or a new success gate. Preserve all original endpoints and report every condition/model, including adverse results.

After complete verification and matched fixed CPU replay, run:

    .venv-compiler-cpu Python infra/summarize_lifted_v4_secondary_metrics.py --root artifacts/runs/lifted_v4_final --panels artifacts/inputs/lifted_v4_panels

The output has 60 rows: three models; H1/H3 with k=0,1,2,3; each real pollutant and their pooled panel with e=0,3,6. For PFN-L/LCT-L report NLL, RMSE, central-90% coverage and CRPS. TabPFN exports only NLL and squared error; do not infer its other metrics. Average individual-model score arrays over all three fits, then equally over tasks or weeks. RMSE is sqrt(mean squared error), not ensemble error or mean-seed RMSE. The pooled real panel weights weeks equally and episodes equally within each week, preserving the original estimand.

Require exact recovery of all five frozen aggregate NLL gains, unchanged input hash bindings, and no overwrite. There are no new intervals, model selection or thresholds. Existing tests verify complete-input/replay boundaries; hand-computed engineering assertions checked unequal-week weighting, squared-error aggregation, mask averaging, unavailable metrics and invalid scores. These are software checks, not empirical findings. Actual secondary metrics remain pending.
