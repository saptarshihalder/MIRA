# Trained contribution and closest overlap

The candidate contribution is a support-calibrated, factor-augmented Gaussian
site layer for in-context sensor regression, instantiated both as a small
cavity-refinement network and as a transformer output layer. Information-form
aggregation, latent-variable products of experts, learned EP messages, set
attention and analytic initialization are established ingredients. The claim
is the particular trained construction and its measured behavior, not invention
of those ingredients or proof that no equivalent architecture exists.

| Prior work | Established overlap | Candidate distinction and remaining limit |
|---|---|---|
| Learned EP, Heess et al. 2013 | Learned, reusable inference operators and cavity information | Deployment-specific response calibration is inferred from incomplete labeled support; learned messages themselves are not novel. |
| MVAE, Wu and Goodman 2018 | Multivariate latent products of experts; omit absent modalities | Explicit target-plus-nuisance factor sites with a support-fitted analytic anchor; vector lifting and subset composition are not new alone. |
| NeuMiss | Gaussian conditioning with missing features and learned iterative computation | Episodic unknown sensor responses and variable-cardinality shared site heads; neither informative missingness nor general mask robustness is solved. |
| Set Transformer | Shared set processing and permutation-compatible interaction | A constrained Gaussian site output layer rather than set attention alone; guarantees depend on the anchor contract. |
| IFNet 2025 | RNN-learned fusion weights for correlated measurement noise within information filtering | The publisher describes temporal state estimation; the candidate targets labeled calibration support and contemporaneous regression. Full IFNet text was not inspected, so exhaustive separation is not established. |
| DIFNet, pinned arXiv v1 | Decentralized learned information fusion, correlated noise, changing node/topology settings | Section3.2 specifies N-dependent dense input/output sizes, with absent communication zero-filled; the candidate uses shared sensor heads without resizing them. This is an architectural distinction, not proof DIFNet cannot accommodate changing sensor counts. |

Primary sources: [learned EP](https://proceedings.neurips.cc/paper/2013/hash/1714726c817af50457d810aae9d27a2e-Abstract.html),
[MVAE](https://arxiv.org/abs/1802.05335),
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html),
[IFNet publisher](https://www.sciencedirect.com/science/article/pii/S1566253524005281),
[DIFNet full v1, section3.2](https://arxiv.org/html/2508.18854v1).
The first three records are cross-checked in LIFTED_REFERENCE_VERIFICATION.md;
the IFNet publisher abstract/section summaries and DIFNet architecture text were
read on October11. This is a focused audit, not exhaustive novelty clearance.

## Attribution of gains

V4 matches backbone scale, source-task/update budgets, seeds and fine-tuning
recipes. It does not separately isolate analytic initialization, lifted-site
parameterization and every other head difference. Earlier anchor-residual MLP
and static-site controls provide supporting evidence at a different scale;
they are not matched large-transformer component ablations. The restricted
static-site rank necessity proposition does not establish necessity of lifting
for arbitrary context-dependent scalar neural heads.

Unrun controls must remain unrun: an equally initialized scalar/contextual head,
a generic joint-Gaussian head, a larger-rank site model, and replication on a
third real domain. Do not add these after inspecting v4 as if they were frozen
confirmation. Their absence limits component-level and broad-generalization
claims even if the fixed v4 endpoints pass.
