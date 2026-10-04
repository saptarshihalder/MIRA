# Conditional next direction: trainable planning under constraint changes

Update: the October 4 prior-art check rejects novelty of this generic planner (CONSTRAINT_PRIOR_ART_CHECK.md). The subsequent Wine established-model benchmark fails its superiority gates (WINE_ACQUISITION_FINDINGS.md). The following is historical conditional planning, not a promoted architecture.

October 4, 2026. Proposal only; no successor has been trained or validated. The closed repair probe identifies a synthetic planning gap but rejects the tested learned solution. More architectural complexity alone does not justify another run.

## Concrete model hypothesis

Learn an observation-state encoder, conditional outcome transitions and terminal prediction risk from source fitting data. Unroll a finite planning layer that masks unavailable or already measured actions and decrements the remaining budget at every transition, including imagined future steps. Propagate terminal risk and expected measurement counts through these transitions; evaluate their sum with deployment prices. Include stopping in every state. Train prediction and decision losses end to end, with Bellman consistency as an ablation rather than an assumed improvement. At deployment freeze weights and supply only observed measurements, masks, prices and budget. No query labels or unacquired values enter the policy.

The proposed distinction to investigate is a learned representation whose future measurement-count predictions remain feasible under previously unseen combinations of constraint edits. Test whether structural propagation provides useful compositional transfer beyond ordinary masked planning. Feasibility constrains actions; it does not guarantee prediction accuracy, calibration, clinical safety or publication quality. Exact-source teachers are permitted only in a separately labelled diagnostic with equal access for controls; real experiments must learn from the same finite source labels.

## Prior art and the novelty burden

- [Successor features/GPI, NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html): reward decomposition and policy reuse are established.
- [Value Iteration Networks, NeurIPS 2016](https://arxiv.org/abs/1602.02867): differentiable planning is established.
- [Geometric policy composition, ICML 2022](https://proceedings.mlr.press/v162/thakoor22a.html): combining planning and policy improvement is established.
- [BRiG-AFA, August 2026](https://arxiv.org/html/2608.02305v1), [NM-PPG, May 2026](https://arxiv.org/html/2605.05511v1), and [L2M, 2026](https://proceedings.mlr.press/v306/kobayashi26a.html): non-myopic learned acquisition and related components overlap strongly.

These works support the ingredients, not the novelty or efficacy of this proposal. Audit their implementation and constraint-transfer assumptions before calling the architecture distinct. If it reduces to a standard masked value-iteration or successor-feature implementation, stop rather than rename it.

## Budgeted gates through October 16

1. October 4-6, prerequisite audit: write a component-by-component distinction against the closest methods. Audit candidate data versions, licenses, identities, label proxies and feature acquisition groups before any download/scoring. Costs must be measured/documented or explicitly simulated; simulated costs cannot establish clinical utility. The eight metadata candidates are not yet an external panel.
2. October 7-12, conditional development: freeze one protocol with source-only fitting, three seeds and finite cost/availability shifts. Compare equal-supervision and equal-search-budget direct masked planning, ordinary SF/GPI, a full-information upper reference, and applicable acquisition baselines. Ablate learned transitions, structural propagation and constraint conditioning. Fix primary risk-plus-cost, harm and runtime thresholds before fitting; no test-based threshold tuning. Inspect calibration and source-task harm. Use new development worlds; the failed repair cells stay used development.
3. October 13-16, conditional validation: only after development passes, lock checkpoints and score a separately frozen, identity-audited external panel once. Assess dataset-level variation only with enough independent families; report seed variation separately. Reproduce from raw inputs, preserve failures and audit label access. If prerequisites fail, record the failure and leave the milestone incomplete.

No cloud launch is presently justified. Modal provisions remain $16.45 plus $3 protected under the $20 cap, leaving $0.55 with invoices unreconciled. Reconcile actual provider charges before reserving any bounded GPU pilot; count failures and allow no automatic retries. Start with a CPU feasibility check only after the novelty/data prerequisites pass. Do not spend protected reproduction funds chasing a positive result.

Deliverable standard: a distinct trained mechanism with reproducible advantage over the strongest matched controls on independent real data. This remains an aspiration, not a result or acceptance guarantee.
