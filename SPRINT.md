# MIRA: October 1â€“20, 2026

Objective authorized by the user: a rigorous NeurIPS-quality manuscript by October 20. This is a draft deadline, not a NeurIPS 2026 main-track submission deadline (May 6). On October 3 the user reduced the total Modal compute ceiling to $20, superseding the earlier $26 ceiling. Conserve tokens with saved state, short updates, bounded experiments and compact reports.

## Scientific decision

Primary track: a trained-model contribution evaluated against strong simple controls. The 961-parameter uncertainty mixture improves over synthetic moments, but its calibrated-guard advantage is uncertain. Actual Modal T4 native Diabetes training is complete: source XGBoost and 400 CUDA updates each for fine-tuned/scratch neural and linear corrections. Native pretraining does not help; scratch neural and linear models perform similarly. Read docs/POLICY_ADAPTER_NEXT_GATE.md before a successor. Larger 2026-release A100 development and converged/calibration controls are complete; neural utility is small or matched by simple controls. Diagnose query-dependent learning on CPU before another paid protocol. Historical representation evidence does not establish a causal pretraining-prior effect.

Conditional track: MIRA-Shift. Activate only after development demonstrates useful remaining headroom against strong simple corrections and refreshed-context TFMs with the same labels. Existing CPU results do not justify large neural training. A negative finding stays in the record; manuscript scope follows evidence.

## Calendar and evidence gates

| Dates | Work | Required exit evidence |
|---|---|---|
| Oct 1â€“3 | Verify handoff, reproduce CPU, TFM compatibility/throughput, development matrix | Hashes, tests, predictions, exact checkpoints, paired effects and measured cost; branch decision |
| Oct 4â€“6 | Freeze hypotheses/controls and confirmation configs; implement conditional adapter only if justified | Read-only confirmation design; no selection using confirmation outcomes |
| Oct 7â€“11 | Fresh-seed confirmation and a small predeclared real-covariate panel | Independent task-level uncertainty, accessible data provenance, negative results retained |
| Oct 12â€“14 | Preprocessing/width/label-budget/backbone ablations | At least two backbones and contemporary checkpoint when accessible; error logs if inaccessible |
| Oct 15â€“17 | Write evidence-backed manuscript, figures and related-work comparison | Each number traced to predictions; claims scoped to actual tested conditions |
| Oct 18â€“20 | Clean-command reproduction, anonymous artifact review and final editing | Verified manuscript, code/data/config manifests, compute report, limitations, submission-readiness assessment |

## Minimum design

- Development starts at task seed 40000; confirmation reserves 60000â€“60019.
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
| Completed conservative reservations including failures | 16.95 |
| Clean reproduction reserve | 3 |
| Unallocated | 0.05 |

Modal calls must reserve cost in `artifacts/manifests/compute_ledger.json` before execution, have finite timeouts, one GPU container and no automatic retries. Include failed calls, image builds and provider usage when available. The latest user-authorized total ceiling is $20; credit balance is not yet independently verified. No subscription purchase, additional funds, manuscript submission or messages to third parties are needed for this phase.

Earlier internal allocations are historical. The current $20 ceiling includes the $3 reproduction reserve; reservations are larger than measured active-compute charges and must not be described as invoices. A deployed Modal app has no schedule or web endpoint and scales to zero containers after its bounded, manually invoked evaluation.

## Continuation

October3 checkpoint supersedes pending Day9â€“14 text below: matched learning, 3,584 CPU predictions, native-data provenance, 24 full-feature controls, 900 null supports and all324 backbone predictions are complete. Read `docs/DAY_8_14_AUDIT.md` and the current STATUS checkpoint. Two pre-inference GPU failures remain charged conservatively; user-authorized separate T4 repair completed162 predictions. Reservations10.55; provider MIRA app report0.54356618 (lag/attribution caveats). Both initial pilot and single repair are closed.

Day15â€“17: begin a CPU-only learner trained on predictive utility rather than privileged basis recovery. Include class imbalance, acquisition blocks, observed covariates and matched centered/calibrated heuristics. Freeze an actual objective and comparison budget; require improvement beyond simple controls before fresh confirmation or additional paid execution. Do not tune on the already-used two native datasets and call them external holdouts. Day18â€“20: fresh external evidence, reproduction, manuscript synthesis, official formatting/authorship and readiness review; current learned-method and native-compiler/PDF gates remain unmet.

Read `STATUS.md` first, inspect ledger and existing outputs, then execute the next unresolved gate. Reuse results and exact dependencies. Update status/decision after meaningful evidence changes. Do not rerun confirmation after development changes without explicitly recording that it is no longer untouched. Preserve source attachments under the workspace `reference/handoff` directory.

## Accelerated confirmation checkpoint

At the user's instruction to continue toward a concrete output, confirmation_pairwise_v1 was frozen and committed before execution. Its 360 cells passed the two prespecified primary contrasts on 20 fresh tasks. Seeds 60000â€“60019 have now been used for this scoped panel. Read the confirmation decision before choosing the next gate; preserve outcome-independent protocol changes and keep later baseline/real-covariate claims separate.

## Accelerated Day 4â€“8 checkpoint

Completed on October 1 at the user's request: strong support-trained baselines (240 CPU predictions), complete data/source/protocol freeze before scores (dbda963), evidence-based decision to leave the conditional adapter inactive, the previously frozen fresh-task pairwise confirmation (360 cells), and all 1,440 cells of the six-dataset/five-fold panel. Independent saved-probability/episode/aggregation audit passes; 76 local tests pass. Full details are in `artifacts/reports/day4_8_completion.md`.

There was no need to wait for calendar dates or repeat completed confirmation. The six-dataset results narrow transfer claims; the paper records all controls/negative effects. Day 9 onward concerns targeted label-budget/representation ablations, contemporary access, reproduction/anonymous packaging and manuscript readiness. Keep confirmation separation: previously used seeds are not untouched. Native compiler environment failure still prevents PDF verification; official formatting/authorship/final readiness remain open.

Reservations now total $9.05 across nineteen calls, including failures, with $3 reproduction reserve retained. Provider-reported app charges $0.50322492 may lag and image-build attribution remains unverified. Continue from saved evidence, not additional broad exploratory reruns.

## Authorized elevation extension (next priority)

Follow `docs/MASK_COMPILER_PLAN.md` and draft `configs/mask_compiler_gate_v1.json`. CPU groundwork is complete:80 exploratory tasks / 480 saved predictions, exhaustive reversible-mask checks and two targeted tests; no paid calls. Strong sparse baselines give mixed results and zero-signal harm survives, so there is no TFM/method claim yet. Seeds73000â€“73004 are used development.

Day9â€“11: audit prior art, support-only nested selection/guard, matched-search parity/interaction baselines and isolated TFM runner. Day12â€“14: freeze/push and run at most two $0.50 feasibility calls, using fresh development seeds74000â€“74002. Keep total $26 / $3 reproduction reserve. Day15â€“17: confirm only a passing, useful hypothesis using a separately frozen plan and unused seeds98000â€“98019; contemporary-checkpoint access and fresh external validation are required for broad claims. Day18â€“20: reproduce, audit, package and finish a paper whose claims follow evidence, including a negative extension if gates fail. No guarantee of venue acceptance. Prefer this bounded hypothesis test over further broad matrices.

## Trained-model direction supersedes study-only endpoint

The user requires a trained contribution. Primary candidate is MIRA-Compiler, a learned support-conditioned selector of reversible mask bases for a frozen tabular predictor. A6092-parameter CPU prototype is actually trained; saved evidence lives in `artifacts/reports/trained_compiler_v1/`. Rawteacher recovery82.42%, guard78.12% (ties the78.12% heuristic); this does not establish downstream efficacy. Privileged synthetic supervision is disclosed and never supplied at inference. Existing study remains supporting evidence. Substantial residual-adapter training remains conditional; small compiler pretraining is now authorized.

Next gates: matched-pretraining learned baselines and strong existing CPU/CV controls; frozen-backbone validation with actual query losses and null harms; commit runnable protocol/checkpoint hashes before at most$1 paid pilot; then conditional fresh confirmation and external evaluation. Novelty/acceptance remain unestablished. Use current named checkpoints honestly, preserve$3 reserve, and do not reuse trained/development seed ranges as untouched confirmation. Daily continuation now follows this primary direction.

## Trained successor update — October 3

Bridge v1 fails; v2 support-conditioned regularization has a real controlled shifted gain over support logistic 0.025002 [0.021761,0.028315] and target-only conditioning .003300 [.000862,.005632]. However it harms the ignorable null by .060856 [.050163,.070476] and nonlinear-shift NLL .619693 exceeds frozen .594366. The aggregate shifted gain does not establish robustness. Both gates fail; neither candidate advances to native confirmation.

Provisions $14.05 plus protected reproduction $3 within $20. Current next gate: docs/BRIDGE_NEXT_GATE.md. No completed phase is repeated; all negatives and confirmation separation persist.

## Recursive JEPA engineering checkpoint — October 3

Implemented and actually trained a 6,880-trainable-parameter, three-step tied recursive label-memory model on Modal NVIDIA A100-SXM4-40GB. Support-only residual/class memories drive nonlinear embeddings and bounded differentiable head updates; supervised meta-query loss backpropagates through every step. A support-conditioned EMA teacher predicts embeddings across nested observed-only missingness views. Query labels never enter inference or the teacher target. JEPA/VICReg/MAML motivate ingredients; novelty is unestablished.

Three controls completed64 CUDA updates each (full, supervised-only, one-step); runner13.835s/wrapper24.838s. Seven grouped unit tests and a separate two-update CPU engineering smoke pass. Independent audit verifies16 files/96 probability arrays and48 CPU checkpoint predictions (max error1.19e-7), all label boundaries, source logits, teacher states and exact frozen intervention. Saved raw predictions, all checkpoints, traces and audit are in artifacts/reports/recursive_jepa_v1_gpu. Protocol96258a1f was committed252c348 before the paid call. The app has zero active tasks, no schedule or endpoint, and no retries.

Full JEPA auxiliary MSE1.036854→.003989; student mean std .022198→.029807 stays far below variance floor1 and final recurrent query std is .015229 on the logged batch. This is low-spread/collapse-risk evidence, not robust label representation. Diagnostic NLL full/frozen: sign-flip .616819/.617999; nonlinear .592116/.593328; no-shift .600430/.601214; ignorable .646807/.647985. Supervised-only and one-step scores are almost identical; support logistic sign-flip .503530 is much stronger. Two correlated development worlds cannot establish safety, significance or neural novelty. Different recurrence/deep-supervision/auxiliary costs prevent a causal ablation claim. Gate is deliberately null, not passed. Earlier failed full gates remain failed.

Next: diagnose representation geometry and nonlinear label discrimination on CPU before another GPU protocol. Use per-world variance/effective rank, held-out support class separation and shuffled-label controls; do not solve a low-variance auxiliary merely by inflating embedding norm. Consider a learned field interaction/attention representation that preserves task-relevant cross-feature structure, then independently validated frozen fallback. Freeze a fresh computationally matched protocol and per-regime/null gate before scientific GPU expansion. Confirmation98000-98019 remains untouched; no native extension is activated.

Conservative provisions14.85 plus protected reproduction3 within20 leave2.15 unallocated. New $.80 retained; provider report has not yet posted matching rows, so charge is unavailable, not zero. Preserve source/editor; no email or submission.

## Spectral learned-operator checkpoint — October 3

Read docs/SPECTRAL_FINDINGS.md. A257-parameter learned mode filter and scalar control each completed500 CUDA updates on Modal A100; fresh development8seeds x2widths x4regimes. Neural selectivity improves sign-flip NLL over the trained scalar by.019314 [.004884,.033744] and nonlinear NLL over support logistic by.010659 [.004019,.017299]. However frozen wins on nonlinear and ignorable shifts; ignorable harm.013053 [.006387,.019719]. Full development gate FAILED. Intervals are exploratory and unadjusted. Support-objective descent holds under the stated convex bound and passed numerical checks; it is not test-risk safety. All384 score arrays/192 CPU predictions audit/replay pass. Protocol8383e139 committedf00fce3 before GPU. Artifacts/reports/spectral_v1_gpu includes full predictions/checkpoints.

JEPA diagnosis: solved head on learned pooled features worsens sign-flip NLL to.637938, versus original.616819 and random-solved.630923. The next architecture candidate is a cross-fitted label-response operator: trainable field interactions plus learned spectral gates informed by disjoint support gradient agreement. First test free CPU discrimination of harmful modes; then commit fresh label/compute-matched protocol. See SPECTRAL_FINDINGS for closest prior art, constraints and failure cases. No claim of unique novelty,100% safety or venue readiness; confirmation98000-98019 untouched.

Budget:15.65 provisions +3 protected reproduction leaves1.35 within20. Latest job complete/zero active tasks, no retry; provider charges may lag, full.80 retained. Existing native LaTeX compiler environment failure persists until a successful check. No email/submission.

## Support-verified query operator — October 4

Read docs/CROSSFIT_FINDINGS.md first. A370-parameter query-conditioned trained correction now PASSES the prespecified exploratory synthetic robustness gate on20fresh latent seeds/two widths/all four regimes. Sign-flip gain.045838 [.034527,.057149] over frozen; .032354 [.020077,.044631] over same-rule guarded logistic; .018036 [.008201,.027872] over no-query learner. All80ignorable branch decisions reject adaptation. Nonlinear harm.0000123 is small. However gain.006202 [-.002545,.014948] versus the114parameter scalar/query learner is uncertain; the stronger neural-specific gate FAILS. Ordinary full-support logistic still wins on sign flip while harming null cases. No universal safety, novelty, native utility or venue-readiness claim. Intervals are unadjusted exploratory; confirmation98000-98019 untouched.

Three500-update models trained on Modal CUDA/A100 allocation. Evaluation failed at the Platt numerical control after132cells; all checkpoints preserved. Committed CPU-only recovery solved the SAME convex calibration objective with gradient<8.33e-11; no retrain/remote retry/seed changes. Original GPU run stays FAILED in ledger. All160cells completed;1,600scores,960CPU prediction-array replays,960split decisions and640identity arrays audit pass. Original132GPU neural files agree within1.87e-9. Training traces/exact device-name log were lost at late evaluation abort, disclosed. Artifacts/reports/crossfit_v1_gpu includes original failed-run archive and recovered full predictions.

Next: resolve capacity/task-conditioned scalar and support-only-head controls, then freeze a finite independent replication and native-transfer protocol. Current learned field representation is handcrafted; optional learned interactions need evidence against matched features. See CROSSFIT_FINDINGS. Do not add seeds until significance or spend on another architecture without an unresolved-control rationale. Budget16.45 provisions +3 reproduction leaves.55 under20. Latest app tasks0; bills may lag. Same paper/editor updated; native PDF compiler environment remains unresolved. No email/submission.

