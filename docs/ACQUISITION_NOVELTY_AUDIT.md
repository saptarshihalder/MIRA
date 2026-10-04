# Acquisition novelty gate — October 4, 2026

Verdict: the broad proposed contribution fails the distinctness gate. Architecture-specific novelty remains unresolved; no successor training or external scoring is justified yet. This is a targeted methods review, not an exhaustive literature/code audit.

| Primary source inspected | Methods evidence | Consequence for MIRA |
|---|---|---|
| [BRiG-AFA, August 2026](https://arxiv.org/html/2608.02305v1), sections 3–4, equations 3–5 | Fits budget-conditioned terminal-risk networks backward from shorter-horizon targets; deployment sees observed values/mask/candidate/budget only. | Learning lookahead to exploit delayed feature value is already covered. A two-step-versus-one-step ablation alone cannot establish novelty. |
| [NM-PPG, May 2026](https://arxiv.org/html/2605.05511v1), sections 3.1–3.4 | Non-myopic acquisition with stop actions and measurement costs; pathwise gradients and straight-through rollouts address policy optimization. | Switching from sampled REINFORCE to differentiable acquisition is also prior art. |
| [L2M preprint v1](https://arxiv.org/html/2510.12624v1), sections 3.1 and 4.2, equations 1–5 | Defines retrospective-missingness constraints and blocked policies, trains with task context and targets greedy information gain. | Missingness-aware, support-conditioned acquisition is not by itself distinct. This inspected preprint must be compared with the final 2026 version before attributing every detail to that version. |
| [Sequential latent acquisition, 2020](https://arxiv.org/html/2011.00825v1), section 3.2 and experiments | Sequential latent belief representation supports acquisition/control policies under partial observations. | Recurrent latent transitions plus RL do not independently establish novelty. |

[BRiG-AFA repository](https://github.com/JIAORONG-FENG/BRiG-AFA) README exposes reproduction commands and an MIT license. [L2M repository](https://github.com/reAIM-Lab/Learning-To-Measure) README exposes predictor/policy pretraining commands and source/config directories. Repository landing pages were inspected; implementation files have NOT been audited or executed. Closest-baseline reproduction remains pending. No source metrics are presented as our results.

One possible substantive revision to audit, not a claim or implemented model: learn when to spend computation on deeper acquisition planning from support-estimated transition error, rather than always using two steps. It must beat fixed-depth methods at equal total prediction/measurement/compute budgets and survive direct adaptive-computation prior-art review. Simply adding an uncertainty threshold is insufficient. Do not train this unverified revision automatically.

Two delegated audits terminated immediately at the account usage limit and produced no evidence. The primary agent performed the recorded source/metadata reads directly. No paid experiment, new outcome evaluation or baseline reproduction occurred. October 7–12 development superiority and October 13–16 external validation remain pending their prerequisites; they are not completed milestones.
