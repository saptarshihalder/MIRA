# Trained MIRA-Compiler — working plan, updated October 3, 2026

## Day14 outcome supersedes unperformed-gate text below

All matched-training, native-data and direct-backbone development gates executed. See `docs/DAY_8_14_AUDIT.md` and `artifacts/reports/trained_compiler_backbone_v1/`. Trained compiler improves over uncompiled indicators on synthetic multibit tasks (.085870/.061096 mean gains for TabPFN/TabICL), but fixed heuristic and equally informed controls explain the gain. All six native-data compiler/identity predictions per backbone coincide. **Do not advance this candidate to fresh confirmation or claim a useful new learned method.**

Next CPU-only candidate must train on actual predictive utility with support-held-out training labels, rather than privileged active-parity recovery. Include class imbalance/acquisition blocks/observed-value interactions, compare centered moments and calibrated fixed selection, and match all meta-training/teacher and target-label budgets. Test whether a learned residual or utility ranker offers improvement beyond these controls before designing another paid gate. This is pending research, not a promised model or acceptance guarantee.

Initial two-call/$1 pilot exhausted with two preserved pre-inference failures. User's later GPU instruction authorized one separate $.50 repair, which completed162 T4 predictions. No further attempts in either phase. Global reservations10.55, cap26, reserve3. CPU TabPFN completed162 predictions; Colab notebook is prepared but unexecuted/unverified. All current seeds74000–74002 and90000–90063 are now development, while98000–98019 remain unused.

## Primary candidate

MIRA-Compiler is a small, permutation-invariant support-set encoder receiving numerical observations, observable masks, and support labels. It predicts one of **12 preverified invertible GF(2) mask bases**, including identity. Append z=Am mod 2 while retaining numerical values/NaNs and identical label budgets. Identity is the fallback for unsupported dimensions, failure, or a predeclared confidence threshold.

Synthetic meta-training may use explicitly privileged active-column/gamma teacher information **only during pretraining**. At target-task inference, basis selection uses support data alone: no query labels, active-column identity, gamma, oracle probabilities, or generator metadata. Disjoint meta-training, development, and fresh evaluation tasks are mandatory.

The falsifiable hypothesis is improved frozen-predictor query log-loss versus raw indicators, matched random bases, and equally informed learned/classical baselines on new multibit tasks. This is a candidate method, with no TFM efficacy or novelty guarantee. Parity features, feature engineering, and Bayesian information equivalence are established; invertibility alone is not a contribution. Compare against the existing [primary parity reference](https://arxiv.org/abs/2402.04248).

## Completed evidence remains fixed

Existing confirmation/panel evidence, configs, and hashes remain unchanged. V0 CPU checks completed 80 development tasks/480 saved predictions, including sparse full-parity controls; two tests pass, with $0 paid. Restricted-logistic gains, mixed sparse-baseline comparisons, and zero-signal harm do not validate the trained compiler or TFMs. Support sizes 32/128 were not nested; compiler selection used 12× more CV fits than each single-basis baseline. Preserve these limitations and `artifacts/reports/mask_compiler_v0/interpretation.md`.

The existing `mask_compiler_gate_v1.json` is a historical draft 120-cell pilot; GPU integration and confirmation remain unexecuted. Freeze a runnable trained design before paid inference.

## Prioritized Day 9–20 gates

1. **Days 9–11: CPU prototype.** Verify inverse/round-trip recovery and matched inputs, then train the support encoder on synthetic tasks. Freeze architecture, twelve bases, teacher targets, optimizer, task counts, seeds, fallback, and selection budget. Evaluate on disjoint development tasks. Stop if teacher metadata leaks into inference or recovery fails.
2. **Days 12–14: fair baselines and direct TFM validation.** Include existing support-CV/L1 interaction/parity and XGBoost baselines, raw/shuffled indicators, random bases, and learned baselines receiving the same synthetic pretraining tasks, teacher access, labels, and comparable optimization/parameter budgets. Record actual costs. Directly test exact pinned frozen TabPFN/TabICL checkpoints; require a contemporary-checkpoint comparison for broad claims. Log inaccessible checkpoints without substitution. Stop expansion if equally informed baselines explain gains.
3. **Days 15–17: fresh confirmation.** Only after development passes, freeze one hypothesis, useful-effect threshold, primary contrast, multiplicity policy, and aggregation unit. Use tasks disjoint from all previous runs, including 60000–60019. Stop a method claim if direct TFM gains miss the gate.
4. **Days 18–20: reproduction and writing.** Audit saved predictions, leakage, provenance, and claims; reproduce within reserve. Failed gates become an explicit negative result, not a promised breakthrough.

Initial paid ceiling: **$1**; total project ceiling: **$26**, protecting the **$3 reproduction reserve**. Separate reservations from invoices, disable retries, and stop before exceeding either bound.

Actual CPU prototype: fixed17-dimensional support-moment encoding plus learned6092-parameter MLP (not learned raw-row encoder),2048 training tasks,80 epochs. See `artifacts/reports/trained_compiler_v1/`; teacher recovery/raw82.42%, confidence fallback78.12%, heuristic78.12%. No downstream TFM claim. New trained protocol must supersede the historical draft before paid calls.
