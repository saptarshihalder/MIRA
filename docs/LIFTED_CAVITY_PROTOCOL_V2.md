# Lifted cavity network: protocol v2 (frozen before any v2 panel exists)

Committed October 6, 2026, before generating or scoring any panel or test split listed here. Protocol v1
(`LIFTED_CAVITY_CONFIRMATION.md`) passed. v2 adds the baselines a reviewer will demand, a privileged
Bayes-optimal ceiling, wider and harder shift panels, and a real dataset with natural missingness. No model listed
here is retrained, selected or tuned after this commit. Every endpoint is reported once, pass or fail.

## Models (fixed)

**Synthetic-trained learned models.** All use the same task family and training masks (at most one missing query
sensor).

| Model | Description | Training |
|---|---|---|
| `lift1_s{1,2,3}` | Primary | Unchanged since v1 |
| `lift2_s1`, `lift1_static_s{1,2,3}`, `lift0_s{1,2,3}`, `repo_cavity_fresh_s1` | Ablations and repository architecture | Unchanged |
| `anchor_mlp_s{1,2,3}` | Non-modular control | Unchanged; runs only with 5 sensors |
| `pfn_s1` | TabPFN-v2-style cell-token transformer (`pfn.py`, 202,562 parameters) | Fresh tasks every step, same training masks; 30,000 steps of 16 tasks × 16 queries; final checkpoint, no selection |
| `pfnall_s1` | Same transformer | Every query mask (0–4 missing, uniform); descriptive only |

**Support-only closed forms.**
- FA-Gaussian (one factor).
- EM-Gaussian.
- NL-FA: cubic sensor curves with curvature ridge λ = 100, clamped extrapolation and a one-factor residual (`nlfa.py`).
  λ and the degree were chosen on development panel 20261007 only.
- Complete-case Bayesian linear regression (evidence-tuned).
- Complete-case ridge.
- The repository's mean-imputed ridge.

**Privileged references.**
- Exact oracle with true parameters.
- Population Gaussian.
- Bayes-optimal predictive under the true task prior, by HMC (`bop.py`). It sees exactly the support and query
  inputs; development panel 20261007 gave R-hat ≤ 1.1 on 99.95% of identifiable parameters.

## Synthetic panels (new seeds)

All panels have 20% support missingness. Masks are every pattern with k = 0–3 missing sensors, or 20 seeded
patterns when there are more (`evaluate2.banks`).

| ID | Seed | Tasks | Sensors | Nonlinearity | Support rows |
|---|---|---|---|---|---|
| F1 | 20261201 | 256 | 5 | .4 | 48 (main; HMC ceiling computed) |
| F2 | 20261202 | 128 | 8 | .4 | 48 |
| F3 | 20261203 | 128 | 16 | .4 | 48 |
| F4 | 20261204 | 128 | 5 | .8 | 48 |
| F5 | 20261205 | 128 | 5 | 0 | 48 |
| F6 | 20261206 | 128 | 5 | .4 | 16 |
| F7 | 20261207 | 128 | 5 | .4 | 96 |

## Real data

Built with `eval_real.py build --seed 2027`.

**Beijing multi-site.**
- Task: log PM2.5 at each of 12 stations from the other 11, using the natural missingness.
- Test: weekly episodes from 2016-01-01 to 2017-02-28. This period has never been scored.
- Fine-tuning: source episodes from 2013-03-01 to 2015-12-31.
- Conditions: natural masks, plus 3 or 6 extra dropped sensors.

**UCI Air Quality, CO and NO2.**
- Test: weeks after 2004-10-01. These weeks were examined in v1, so they are secondary evidence only.

**Fine-tuning.** The recipe in `finetune_real.py` is identical for every model: 4,000 source episodes, 2,000 steps,
lr 3e-4, seed 11. It is applied to:
- `lift1_s{1,2,3}`;
- `lift1` from initialisation, without synthetic pretraining;
- `lift1_static_s1` and `lift0_s1`;
- `pfn_s1`;
- `anchor_mlp_s{1,2,3}` (Air Quality only).

Zero-shot `lift1_s1` and `pfn_s1` are also scored.

The gas-sensor-drift dataset was examined in development and dropped. Its heavy-tailed sensor outliers made every
Gaussian method's NLL dominated by a few episodes (median-versus-mean gaps of over 100 nats), so it cannot separate
methods. This is reported, not hidden.

## Endpoints

Gains are in nats of mean Gaussian NLL; positive means `lift1` is better. Each endpoint uses a paired 95%
normal-approximation interval over tasks, with per-task NLL averaged over training seeds. On Beijing, intervals are
cluster-robust: differences are first averaged over the 12 stations within each week, then the interval is taken
over weeks. Pass requires gain ≥ .01 and a lower bound > 0.

| Endpoint | Panel | Comparison |
|---|---|---|
| E1 | F1, two missing sensors | `lift1` vs NL-FA (the strongest closed form on development) |
| E2 | F1, two missing sensors | `lift1` vs `pfn_s1` (matched training masks) |
| E3a | F3, 16 sensors, two missing | `lift1` vs FA-Gaussian |
| E3b | F3, 16 sensors, two missing | `lift1` vs `pfn_s1` |
| E4a | Beijing, natural masks | fine-tuned `lift1` vs complete-case Bayesian linear regression (the strongest closed form on development) |
| E4b | Beijing, natural masks | fine-tuned `lift1` vs fine-tuned `pfn_s1` |

Computed by `experiments/lifted_cavity/confirm2.py`.

**Secondary, descriptive.**
- Every other panel and k.
- The share of the achievable gap closed on F1, measured as (FA-Gaussian − method) / (FA-Gaussian − HMC ceiling).
- 90% coverage and CRPS for the Gaussian predictives.
- `pfnall_s1`.
- Ablations.
- Air Quality results.
