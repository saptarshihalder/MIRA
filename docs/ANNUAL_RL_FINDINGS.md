# Adaptive RL correction results — October 4, 2026

The user-requested finite RL pilot is complete. All three task gates FAIL. RL improves over the calibrated tree corrector on every task, but loses to the strongest conventional baseline on every task. These conclusions concern this specific contextual-bandit model and training budget, not all RL methods.

| Task | Frozen | Platt | Additive | Tree | RL | Exact PG | Supervised actor |
|---|---:|---:|---:|---:|---:|---:|---:|
| brfss_asthma3 | 0.393748 | 0.393274 | 0.395012 | 0.398039 | 0.394884 | 0.394895 | 0.396769 |
| brfss_diabete4 | 0.322880 | 0.325008 | 0.324776 | 0.332555 | 0.330176 | 0.329964 | 0.330401 |
| road_ksi | 0.551698 | 0.553190 | 0.553864 | 0.557583 | 0.553420 | 0.553435 | 0.552020 |

The nonlinear actor chooses among seven logit corrections using one-step REINFORCE, trained only from support labels. It has 19,847 parameters for BRFSS and 12,167 for roads. Three initializations, 300 updates per model, 15 new groups and three matched neural objectives produce 135 saved checkpoints. Actions are averaged into a deterministic predictive mixture. Support-only shrinkage accepts a nonzero RL correction in 8/15 asthma, 6/15 diabetes and 7/15 road seed/group cells. All rejected and unguarded outcomes are retained.

The RL gains over its own 384-label calibration anchor are only about .000023/.000031/.000021 NLL for asthma/diabetes/roads. Exact full-information policy gradients perform almost identically; tiny differences do not prove equivalence. The supervised actor wins over RL on roads but still loses to frozen prediction. Sampled policy optimization did not establish a useful new advantage. Training curves are archived; a fixed 300-step budget is not a convergence guarantee.

Fairness limits: each method sees the same 512 total support labels, but neural models use 384 for fitting and 128 for selection, while conventional CV baselines refit on all 512. Calibration-anchor estimation contributes to differences. The three neural methods have identical architecture, initialization, sampled fit batches, update count, regularization and selection choices. Full support labels permit exact action rewards, so this is not a problem where bandit-only feedback is necessary. No claim of RL novelty or universal failure.

Data: five subsequent eligible 2025 groups per task, disjoint from the previous annual screen; historical 2024 CUDA backbones fixed. There are 23,040 endpoint-row uses, 8,008 unique BRFSS records and 7,680 road records. BRFSS endpoints overlap; no independent dataset-level CI or confirmation claim. Raw features/labels had prior preprocessing exposure. All evaluated rows now become used development. Source-only vocabularies/scales, support-fitted anchors and hashes were frozen before policy fitting/scoring.

Audits pass: all 135 models reload, 405 neural scores and 60 conventional score arrays reproduce exactly; support selection, 45 matched initializations, raw-input joins, identity separation and convex-anchor stationarity check out. A NumPy integer JSON-serialization error in the post hoc auditor was repaired without changing fits, predictions, seeds, selection or gates. No experimental retry. Neural fitting took 74.265 seconds; baseline fitting 10.328 seconds on CPU, excluding preparation/audit. Zero cloud calls/charges; conservative provisions remain $16.45 plus $3 protected, leaving $.55.

Close annual_rl_v1. Do not add seeds or change settings on these used groups until the result becomes positive. Original synthetic evidence remains scoped; native RL advantage and JMLR/NeurIPS readiness remain unproven. The separately documented acquisition-model proposal remains conditional on novelty/data audit. Protected synthetic confirmation 98000–98019 is unchanged.

Reproduction: use Python 3.11/Torch 2.8 CPU for rl_run.py and rl_audit.py; Python 3.12/XGBoost 3.4.1 for preparation and conventional baselines. Add experiments/crossfit_v1 to the module path, import rl_run, set OUT to a new directory and call main() without changing frozen source. For conventional baselines import annual_headroom and override INPUT to annual_rl_inputs, OUT to a new baseline directory, and FREEZE to annual_rl_freeze.json before run(). The baseline auditor supports the same module globals; the RL auditor expects the baseline report/audit path shown in its source. All raw-data prerequisites are pinned in the freeze. Do not overwrite tracked outputs.

Final packaging and deterministic reproduction: see [RL completion review](RL_COMPLETION.md).
