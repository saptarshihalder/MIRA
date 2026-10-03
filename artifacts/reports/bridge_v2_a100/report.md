# Bridge v2 A100 development

Main model: contextual_regularizer; 1857 trained parameters, 600 CUDA updates. Gate passed: **False**.

Four validation and twelve development latent seeds; widths6/10/14 and four regimes. Average widths/regimes within seed before paired bootstrap. These are controlled, retrospectively label-dependent masks; no natural-data claim.

| Model | Sign flip NLL | Nonlinear shift | No shift | Ignorable shift |
|---|---:|---:|---:|---:|
| frozen | 0.600367 | 0.594366 | 0.600610 | 0.661772 |
| target_platt | 0.601104 | 0.595543 | 0.602702 | 0.663152 |
| support_logistic | 0.491318 | 0.624173 | 0.496648 | 0.719699 |
| contextual_regularizer | 0.445794 | 0.619693 | 0.444034 | 0.722628 |
| global_regularizer | 0.495650 | 0.610864 | 0.498016 | 0.710704 |
| type_regularizer | 0.479474 | 0.606654 | 0.478402 | 0.711746 |
| target_only_regularizer | 0.458689 | 0.613399 | 0.456088 | 0.728806 |
| fixed_regularizer | 0.498337 | 0.608472 | 0.500623 | 0.707019 |
| global_grid_control | 0.498337 | 0.608472 | 0.500623 | 0.707019 |

Paired shifted gains (control minus primary):

- frozen: 0.064623 [0.057045, 0.072035].
- target_platt: 0.065580 [0.058766, 0.071919].
- support_logistic: 0.025002 [0.021761, 0.028315].
- fixed_regularizer: 0.020661 [0.017715, 0.023771].
- global_regularizer: 0.020513 [0.017650, 0.023685].
- type_regularizer: 0.010320 [0.006958, 0.013657].
- target_only_regularizer: 0.003300 [0.000862, 0.005632].
- global_grid_control: 0.020661 [0.017715, 0.023771].

Null harm (primary minus frozen):

- no_shift: -0.156577 [-0.167392, -0.146396].
- ignorable_shift: 0.060856 [0.050163, 0.070476].

Independent audit: 1728 score arrays and all observed data/masks/labels/row boundaries; CPU replay on 144 checkpoint predictions. Full prediction archives are retained locally under ignored artifacts/runs; prediction_manifest.json identifies every file.

Reproduce: python experiments/bridge_v2/run.py --config configs/bridge_v2.json --out NEW_DIRECTORY --device cuda (or cpu). Protocol and source hashes are frozen; never overwrite an existing run.

Four-step v2 solver is not claimed converged; full support logistic is converged. Meta-regularization/optimization layers are established prior art. Venue readiness and universal safety remain unresolved.
