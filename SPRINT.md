# MIRA: October 1–20, 2026

Objective authorized by the user: a rigorous NeurIPS-quality manuscript by October 20. This is a draft deadline, not a NeurIPS 2026 main-track submission deadline (May 6). Total project compute ceiling: $26; conserve tokens with saved state, short updates, bounded experiments and compact reports.

## Scientific decision

Primary track: a trained-model contribution evaluated against strong simple controls. The 961-parameter uncertainty mixture improves over synthetic moments, but its calibrated-guard advantage is uncertain. Actual Modal T4 native Diabetes training is complete: source XGBoost and 400 CUDA updates each for fine-tuned/scratch neural and linear corrections. Native pretraining does not help; scratch neural and linear models perform similarly. Read docs/POLICY_ADAPTER_NEXT_GATE.md before a successor. Compare calibration and converged controls before broader query-dependent learning across native datasets. Historical representation evidence does not establish a causal pretraining-prior effect.

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

October3 checkpoint supersedes pending Day9–14 text below: matched learning, 3,584 CPU predictions, native-data provenance, 24 full-feature controls, 900 null supports and all324 backbone predictions are complete. Read `docs/DAY_8_14_AUDIT.md` and the current STATUS checkpoint. Two pre-inference GPU failures remain charged conservatively; user-authorized separate T4 repair completed162 predictions. Reservations10.55; provider MIRA app report0.54356618 (lag/attribution caveats). Both initial pilot and single repair are closed.

Day15–17: begin a CPU-only learner trained on predictive utility rather than privileged basis recovery. Include class imbalance, acquisition blocks, observed covariates and matched centered/calibrated heuristics. Freeze an actual objective and comparison budget; require improvement beyond simple controls before fresh confirmation or additional paid execution. Do not tune on the already-used two native datasets and call them external holdouts. Day18–20: fresh external evidence, reproduction, manuscript synthesis, official formatting/authorship and readiness review; current learned-method and native-compiler/PDF gates remain unmet.

Read `STATUS.md` first, inspect ledger and existing outputs, then execute the next unresolved gate. Reuse results and exact dependencies. Update status/decision after meaningful evidence changes. Do not rerun confirmation after development changes without explicitly recording that it is no longer untouched. Preserve source attachments under the workspace `reference/handoff` directory.

## Accelerated confirmation checkpoint

At the user's instruction to continue toward a concrete output, confirmation_pairwise_v1 was frozen and committed before execution. Its 360 cells passed the two prespecified primary contrasts on 20 fresh tasks. Seeds 60000–60019 have now been used for this scoped panel. Read the confirmation decision before choosing the next gate; preserve outcome-independent protocol changes and keep later baseline/real-covariate claims separate.

## Accelerated Day 4–8 checkpoint

Completed on October 1 at the user's request: strong support-trained baselines (240 CPU predictions), complete data/source/protocol freeze before scores (dbda963), evidence-based decision to leave the conditional adapter inactive, the previously frozen fresh-task pairwise confirmation (360 cells), and all 1,440 cells of the six-dataset/five-fold panel. Independent saved-probability/episode/aggregation audit passes; 76 local tests pass. Full details are in `artifacts/reports/day4_8_completion.md`.

There was no need to wait for calendar dates or repeat completed confirmation. The six-dataset results narrow transfer claims; the paper records all controls/negative effects. Day 9 onward concerns targeted label-budget/representation ablations, contemporary access, reproduction/anonymous packaging and manuscript readiness. Keep confirmation separation: previously used seeds are not untouched. Native compiler environment failure still prevents PDF verification; official formatting/authorship/final readiness remain open.

Reservations now total $9.05 across nineteen calls, including failures, with $3 reproduction reserve retained. Provider-reported app charges $0.50322492 may lag and image-build attribution remains unverified. Continue from saved evidence, not additional broad exploratory reruns.

## Authorized elevation extension (next priority)

Follow `docs/MASK_COMPILER_PLAN.md` and draft `configs/mask_compiler_gate_v1.json`. CPU groundwork is complete:80 exploratory tasks / 480 saved predictions, exhaustive reversible-mask checks and two targeted tests; no paid calls. Strong sparse baselines give mixed results and zero-signal harm survives, so there is no TFM/method claim yet. Seeds73000–73004 are used development.

Day9–11: audit prior art, support-only nested selection/guard, matched-search parity/interaction baselines and isolated TFM runner. Day12–14: freeze/push and run at most two $0.50 feasibility calls, using fresh development seeds74000–74002. Keep total $26 / $3 reproduction reserve. Day15–17: confirm only a passing, useful hypothesis using a separately frozen plan and unused seeds98000–98019; contemporary-checkpoint access and fresh external validation are required for broad claims. Day18–20: reproduce, audit, package and finish a paper whose claims follow evidence, including a negative extension if gates fail. No guarantee of venue acceptance. Prefer this bounded hypothesis test over further broad matrices.

## Trained-model direction supersedes study-only endpoint

The user requires a trained contribution. Primary candidate is MIRA-Compiler, a learned support-conditioned selector of reversible mask bases for a frozen tabular predictor. A6092-parameter CPU prototype is actually trained; saved evidence lives in `artifacts/reports/trained_compiler_v1/`. Rawteacher recovery82.42%, guard78.12% (ties the78.12% heuristic); this does not establish downstream efficacy. Privileged synthetic supervision is disclosed and never supplied at inference. Existing study remains supporting evidence. Substantial residual-adapter training remains conditional; small compiler pretraining is now authorized.

Next gates: matched-pretraining learned baselines and strong existing CPU/CV controls; frozen-backbone validation with actual query losses and null harms; commit runnable protocol/checkpoint hashes before at most$1 paid pilot; then conditional fresh confirmation and external evaluation. Novelty/acceptance remain unestablished. Use current named checkpoints honestly, preserve$3 reserve, and do not reuse trained/development seed ranges as untouched confirmation. Daily continuation now follows this primary direction.
