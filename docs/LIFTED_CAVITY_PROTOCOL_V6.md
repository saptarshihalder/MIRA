# Lifted cavity: protocol v6: the universal pool transformer on three new sensor networks

Committed October 11, 2026. At commit time:
- no model of this project has been fine-tuned on, or scored on, METR-LA, PEMS-BAY or Intel Lab;
- no v6 test panel exists;
- the only looks at these data were shapes, missing fractions and per-day sensor availability (used to place the
  Intel split; see below);
- no UPT checkpoint has been scored on anything. Source training of UPT s1–s3, LCT s3 and PFN s3 is running on a
  Colab L4 GPU; it uses the synthetic prior only.

## Model under test

**UPT** (`upt.py`): the LCT backbone and lifted-site head, plus a free Gaussian head and a gate on the target-column
token. The output is the linear pool

    p = π · p_site + (1 − π) · p_free

- **Initialization:** zero site head and gate bias +4. For every query, the NLL at initialization is within 0.018
  nats of the FA-Gaussian closed form (tested).
- **Training:** same source recipe as `lct_s1` and `pfn_s1`: 30,000 steps, 16 tasks × 16 queries, lr 1e-3, warm-up
  1,000, d = 64, 4 layers.
- **Seeds:** 1, 2 and 3; final checkpoint only.

## Comparators

- **Learned models**, all with the identical recipe: PFN, LCT and `lift1`, each with seeds 1, 2 and 3.
  - Fine-tuning uses `finetune_real.py` defaults: 4,000 source episodes, 2,000 steps, lr 3e-4, 64 × 8, seed 11.
  - Each model is fine-tuned separately per network.
- **Two-seed PFN linear pool:** fine-tuned PFN s1 and s2, mixture weight ½.
- **Training-free site pool (v5):** fine-tuned LCT s1 with PFN s1, linear pool.
- **Closed forms** written by `eval_real.py build`: FA-Gaussian, EM-Gaussian, BLR and NL-FA.
- **TabPFN v2 in context**, with native missing values and 8 estimators, as in v4.
- **LightGBM** with native missing values, fitted per episode on the 48 support rows. Defaults, except
  `min_child_samples=5`, 200 trees and learning rate 0.05.
- **MICE + BLR:** sklearn `IterativeImputer` (default settings, `random_state=0`) fitted on the support inputs,
  followed by Bayesian ridge regression.

## Networks (`realdata_v6.py`; panels built with `eval_real.py build --seed 6061`)

The task is in-context virtual sensing: predict one target sensor from the 11 sensors nearest to it by location.
- **Episodes:** one day of one target, with 48 support and 48 query rows drawn from the day's rows where the target
  is observed.
- **Split:** chronological (source, then test).
- **Conditions:** extra dropped sensors 0, 3 and 6.

| Network | Readings | Missing | Source | Test | Test targets | Test episodes |
|---|---|---|---|---|---|---|
| METR-LA | speed, 207 sensors, 5-min | zeros (natural) | 2012-03-01 → 05-15 | 05-16 → 06-27 | 24 (seeded) | 991 |
| PEMS-BAY | speed, 325 sensors, 5-min | none; 20% simulated support dropout | 2017-01-01 → 04-30 | 05-01 → 06-30 | 24 (seeded) | 1,464 |
| Intel Lab | temperature, 54 motes, 5-min means | natural; readings outside [0, 50] °C are missing | 2004-02-28 → 03-11 | 03-12 → 03-24 | all | 519 |

**Why the Intel split sits where it does.** Most motes stop reporting after March 23: the per-day count of motes with
at least 96 readings falls from 40 on March 21 to 2 on March 25. The split was placed using only this availability
count, before any model was run.

## Endpoints (`confirm6.py`)

**Common settings:**
- The extra = 0 condition.
- Per-episode NLL, pooled over the three networks.
- Learned models use the seed average of per-episode NLLs over seeds 1–3.
- Paired 95% intervals, cluster-robust over network-days.

Gains are in nats, positive when UPT is better.

| ID | Comparison | Pass condition |
|---|---|---|
| P1 | UPT vs fine-tuned PFN | gain ≥ .01 and lower bound > 0 |
| P2 | UPT vs two-seed PFN linear pool | lower bound > −.01 (UPT uses one model, the control two) |
| P3 | UPT vs TabPFN v2 in context | gain ≥ .01 and lower bound > 0 |
| P4 | UPT vs PFN on a fresh synthetic panel: seed 20261603, 128 tasks, 16 sensors, nonlinearity .4, 48 support rows, k = 2 (trained on 5 sensors) | gain ≥ .01 and lower bound > 0 |

**The headline claim** ("UPT keeps the lifted model's extrapolation and the transformer's real-data fit") needs P1,
P2 and P4 all to pass. P3 is reported separately.

**Secondary, descriptive:**
- each network separately;
- the extra = 3 and 6 conditions;
- every comparator;
- the gate's mean π per network;
- CRPS and 90% coverage;
- latency per query.

**Ablations (descriptive):**
- UPT with π fixed at ½;
- the log-pool head;
- the training-free pool;
- a fresh synthetic panel with 5 sensors (seed 20261601, 256 tasks) for the closed-form → oracle gap.

All endpoint outcomes, including failures, are reported. No hyperparameter, split or comparator changes after the
first v6 score.
