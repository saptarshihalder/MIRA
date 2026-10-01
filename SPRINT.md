# MIRA: October 1–20, 2026

Objective authorized by the user: a rigorous NeurIPS-quality manuscript by October 20. This is a draft deadline, not a NeurIPS 2026 main-track submission deadline (May 6). Total project compute ceiling: $26; conserve tokens with saved state, short updates, bounded experiments and compact reports.

## Scientific decision

Primary track: controlled missingness representation study. Ask when a frozen tabular predictor differs between native missing values and the same data with explicit mask columns. Indicators are deterministic functions of native NaNs; preprocessing can change what reaches the network. An observed difference does not establish a causal pretraining-prior explanation.

Conditional track: MIRA-Shift. Activate only after development demonstrates useful remaining headroom against strong simple corrections and refreshed-context TFMs with the same labels. Existing CPU results do not justify large neural training. A negative finding stays in the record; manuscript scope follows evidence.

## Calendar and evidence gates

| Dates | Work | Required exit evidence |
|---|---|---|
| Oct 1–3 | Verify handoff, reproduce CPU, TFM compatibility/throughput, development matrix | Hashes, tests, predictions, exact checkpoints, paired effects and measured cost; branch decision |
| Oct 4–6 | Freeze hypotheses/controls and confirmation configs; implement conditional adapter only if justified | Read-only confirmation design; no selection using confirmation outcomes |
| Oct 7–11 | Fresh-seed confirmation and a small predeclared real-covariate panel | Independent task-level uncertainty, accessible data provenance, negative results retained |
| Oct 12–14 | Preprocessing/width/label-budget/backbone ablations | At least two backbones and contemporary checkpoint when accessible; error logs if inaccessible |
| Oct 15–17 | Write evidence-backed manuscript, figures and related-work comparison | Each number traced to predictions; claims scoped to actual tested conditions |
| Oct 18–20 | Clean-command reproduction, anonymous artifact review and final editing | Verified manuscript, code/data/config manifests, compute report, limitations, submission-readiness assessment |

## Minimum design

- Development starts at task seed 40000; confirmation reserves 60000–60019.
- Initial models: explicit TabPFN v2, TabICL v2, XGBoost. Contemporary TabPFN checkpoint added only after access and compatibility are verified.
- Initial matrix: gamma 0/.25/.5/.75/.9, 256 support rows, 1024 query rows, 4 ensembles, 3 development seeds. Pilot throughput smoke uses gamma 0/.8 and one task seed; .8 is a recorded engineering deviation, not confirmation.
- Modes: native, native+indicators, native+shuffled indicators; context-only mean imputation with/without indicators. Shuffling controls width/marginals; it does not fully isolate architecture effects.
- Extend families to pairwise and value-dependent masks, sparse pairs, lower missingness and context budgets. Freeze expansion choices after development, before confirmation.
- Primary effect: paired task-level expected log-loss(native) minus expected log-loss(indicators). Also empirical log-loss, shuffled-control differences and raw oracle gaps. Normalize only above denominator 1e-4; preserve negative effects.
- Query rows are not independent tasks. No query labels, oracle probabilities or generator parameters enter model inputs or preprocessing. Save context/query IDs and actual predictions.
- Clinical extension is optional and lower priority than controlled replication. No clinical usefulness claim without a valid patient split and outcome-release protocol.

## Budget allocation (ceilings, not bills)

| Purpose | USD |
|---|---:|
| Setup, compatibility and development | 6 |
| Confirmation | 8 |
| Real-covariate panel | 5 |
| Essential ablations | 4 |
| Clean reproduction reserve | 3 |

Modal calls must reserve cost in `artifacts/manifests/compute_ledger.json` before execution, have finite timeouts, one GPU container and no automatic retries. Include failed calls, image builds and provider usage when available. The user-reported remaining $26 is a spending ceiling; credit balance is not yet independently verified. No subscription purchase, additional funds, manuscript submission or messages to third parties are needed for this phase.

Day 3 reallocates the internal ceilings to cover seven conservative $0.50 development reservations. The total remains $26, including a $3 reproduction reserve; reservations are larger than measured active-compute charges and must not be described as invoices.

## Continuation

Read `STATUS.md` first, inspect ledger and existing outputs, then execute the next unresolved gate. Reuse results and exact dependencies. Update status/decision after meaningful evidence changes. Do not rerun confirmation after development changes without explicitly recording that it is no longer untouched. Preserve source attachments under the workspace `reference/handoff` directory.
