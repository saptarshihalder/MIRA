# Bridge v1 A100 development

Main model: bridge; 20801 trained parameters, 800 CUDA updates. Gate passed: **False**.

Four validation and twelve development latent seeds; widths6/10/14 and four regimes. Average widths/regimes within seed before paired bootstrap. These are controlled, retrospectively label-dependent masks; no natural-data claim.

| Model | Sign flip NLL | Nonlinear shift | No shift | Ignorable shift |
|---|---:|---:|---:|---:|
| frozen | 0.603053 | 0.598463 | 0.599860 | 0.660759 |
| target_platt | 0.605348 | 0.601231 | 0.603152 | 0.661005 |
| support_logistic | 0.506724 | 0.637568 | 0.499201 | 0.711813 |
| bridge | 0.598199 | 0.600113 | 0.600790 | 0.658251 |
| generic_contextual | 0.599501 | 0.595255 | 0.595942 | 0.656956 |
| linear_residual | 0.602928 | 0.598512 | 0.599811 | 0.661156 |
| target_only | 0.597795 | 0.593840 | 0.594040 | 0.655697 |
| no_query | 0.602993 | 0.598427 | 0.599757 | 0.660635 |

Paired shifted gains (control minus primary):

- frozen: 0.001602 [-0.001330, 0.004660].
- target_platt: 0.004133 [0.000136, 0.007725].
- support_logistic: -0.027011 [-0.037520, -0.016530].
- generic_contextual: -0.001779 [-0.004448, 0.000908].
- linear_residual: 0.001563 [-0.001456, 0.004576].
- target_only: -0.003339 [-0.005855, -0.000681].
- no_query: 0.001553 [-0.001368, 0.004609].

Null harm (primary minus frozen):

- no_shift: 0.000930 [-0.002245, 0.003798].
- ignorable_shift: -0.002507 [-0.004642, -0.000187].

Independent audit: 1536 score arrays and all observed data/masks/labels/row boundaries; CPU replay on 120 checkpoint predictions. Full prediction archives are retained locally under ignored artifacts/runs; prediction_manifest.json identifies every file.

Reproduce: python experiments/bridge_v1/run.py --config configs/bridge_v1.json --out NEW_DIRECTORY --device cuda (or cpu). Protocol and source hashes are frozen; never overwrite an existing run.

Four-step v2 solver is not claimed converged; full support logistic is converged. Meta-regularization/optimization layers are established prior art. Venue readiness and universal safety remain unresolved.
