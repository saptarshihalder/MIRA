# Lifted cavity network: confirmation protocol (frozen before any confirmation panel exists)

Committed October 6, 2026, before generating or scoring any panel listed here. Development results are in
`LIFTED_CAVITY_FINDINGS.md`. No model is retrained, selected or tuned after this commit, and every outcome is
reported, pass or fail. No retries.

## Fixed models

- Trained by the shared recipe in `experiments/lifted_cavity/train.py`: 60,000-task pool seed 5150, 20,000 updates,
  training seeds 1–3. Their checkpoint hashes are recorded at scoring time.
  - `lift1_s{1,2,3}` (primary)
  - `anchor_mlp_s{1,2,3}` (non-modular control)
  - `lift1_static_s{1,2,3}`, `lift0_s{1,2,3}` (ablations, descriptive)
  - `repo_cavity_fresh_s1` (repository architecture reference, descriptive)
- Closed forms: FA-Gaussian (primary closed-form control), EM-Gaussian, complete-case ridge, repository ridge.
- Privileged references: exact oracle, population joint Gaussian.

## Panels (seeds never used before this commit)

| ID | Command | Purpose |
|---|---|---|
| C1 | `evaluate.py panel --seed 20261101 --tasks 256` | main: 5 sensors, nonlinearity .4, 48 support rows |
| C2 | `evaluate.py panel --seed 20261102 --tasks 128 --sensors 8` | width transfer (trained on 5 sensors) |
| C3 | `evaluate.py panel --seed 20261103 --tasks 128 --nonlin 0.8` | stronger nonlinearity than training |
| R1 | `airq.py build --seed 7 --target "NO2(GT)"` | real data, new target. Same weekly episodes and 2004-10-01 split as the CO development run. Models fine-tuned on source weeks with the identical recipe (4,000 episodes, 2,000 steps, lr 3e-4, fine-tune seed 11) from `lift1_s1` and `anchor_mlp_s1`, plus `lift1` trained on source weeks only. |

## Endpoints

All gains are in nats of mean Gaussian NLL, so positive means `lift1` is better. Intervals are paired 95%
normal-approximation intervals over tasks (synthetic) or test weeks (real), computed on per-task NLL averaged over
training seeds.

- **E1 (primary, C1, two missing sensors).** Each of the three `lift1` seeds beats FA-Gaussian by ≥ .01. The
  seed-averaged interval's lower bound is > 0.
- **E2 (primary, C1, two missing sensors).** Seed-averaged `lift1` beats seed-averaged `anchor_mlp` by ≥ .01, with
  lower bound > 0.
- **E3 (C2, two missing sensors).** Seed-averaged `lift1` beats FA-Gaussian by ≥ .01, with lower bound > 0.
  `anchor_mlp` cannot run on 8 sensors; this is reported, not scored.
- **E4 (C3, two missing sensors).** Seed-averaged `lift1` beats seed-averaged `anchor_mlp` by ≥ .01, with lower
  bound > 0.
- **R1 (real, two missing sensors, 23 future weeks).** Fine-tuned `lift1` beats EM-Gaussian, with week-level lower
  bound > 0.

Secondary, descriptive only: all other k, ablations, coverage, and gap closed relative to the oracle.

## Results (scored after the protocol commit; all endpoints evaluated once)

| Endpoint | Panel | Comparison at two missing sensors | Gain [95% CI] | Result |
|---|---|---|---:|---|
| E1 | C1, 256 tasks | `lift1` vs FA-Gaussian; per seed +.0740 / +.0739 / +.0742 | +.074 [+.064, +.084] | PASS |
| E2 | C1, 256 tasks | `lift1` vs `anchor_mlp` | +.016 [+.012, +.020] | PASS |
| E3 | C2, 8 sensors | `lift1` vs FA-Gaussian | +.115 [+.097, +.132] | PASS |
| E4 | C3, nonlinearity .8 | `lift1` vs `anchor_mlp` | +.028 [+.020, +.036] | PASS |
| R1 | Air Quality NO2(GT), 23 future weeks | fine-tuned `lift1` vs EM-Gaussian; 17/23 weeks better | +.112 [+.050, +.173] | PASS |

Context, two missing sensors (mean NLL):

| Panel | Oracle (privileged) | `lift1` (3 seeds) | `anchor_mlp` (3 seeds) | FA-Gaussian | EM-Gaussian | Repo ridge |
|---|---:|---:|---:|---:|---:|---:|
| C1 | −.053 | .072 | .087 | .146 | .158 | .391 |
| C2 (8 sensors) | −.575 | −.420 | n/a | −.305 | −.176 | .170 |
| C3 (nonlinearity .8) | −.108 | .190 | .219 | .327 | .295 | .481 |

On NO2, fine-tuned `anchor_mlp` gains −.004 [−.050, +.042] over EM-Gaussian. `lift1` trained on source weeks
without synthetic pretraining gains +.100 [+.052, +.148].

**What the E2 margin means.** It clears .01 but is modest: on C1, the modular architecture's advantage over an
equally trained residual MLP is about a fifth of its advantage over the closed form. The larger separations are in
width transfer, where the MLP cannot run at all, and on real data, where the MLP gains nothing.

Raw scores, checkpoint hashes and the evaluation script output: `artifacts/reports/lifted_cavity_v1/confirmation/`.
