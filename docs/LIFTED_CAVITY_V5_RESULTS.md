# Protocol v5 results: the transformer as a site (October 11, 2026)

Every number below comes from `artifacts/reports/lifted_cavity_paper/runs/confirm_v5.json`, computed by
`confirm5.py`. The protocol and amendment 1 were committed before any v5 score: 27e8637 and a7416d2.

**Data:** Beijing PM10, SO2 and O3, natural missingness. These targets had never been scored by the CPU-scale models.
**Statistics:** 60 week clusters; the 97.5% intervals are the amendment-1 Bonferroni bar.

## Endpoints

| ID | Comparison | Gain (nats) | 95% CI | 97.5% CI | Result |
|---|---|---|---|---|---|
| E13 | geometric pool non-inferior to PFN | +0.003 | [−0.003, +0.010] | [−0.004, +0.011] | pass |
| E14 | geometric pool vs PFN | +0.003 | [−0.003, +0.010] | | **fail** |
| E15 | geometric pool vs LCT | +0.049 | [+0.042, +0.056] | | pass |
| E16 | geometric pool vs two-seed PFN geometric pool | +0.003 | [−0.013, +0.020] | | **fail** |
| E13L | linear pool non-inferior to PFN | +0.024 | [+0.011, +0.037] | [+0.009, +0.039] | pass |
| E14L | linear pool vs PFN | +0.024 | [+0.011, +0.037] | [+0.009, +0.039] | pass |
| E15L | linear pool vs LCT | +0.070 | [+0.062, +0.077] | [+0.061, +0.078] | pass |
| E16L | linear pool vs two-seed PFN linear pool | −0.001 | [−0.013, +0.010] | | **fail** |

## Mean test NLL, natural missingness

| Target | PFN | LCT | Geometric pool | Linear pool | Two-seed PFN linear pool |
|---|---|---|---|---|---|
| PM10 | 0.110 | 0.140 | 0.106 | 0.091 | **0.086** |
| SO2 | 0.181 | 0.238 | 0.186 | 0.174 | **0.161** |
| O3 | 0.311 | 0.360 | 0.299 | **0.265** | 0.280 |
| Pooled | 0.200 | 0.246 | 0.197 | 0.176 | **0.175** |

The linear pool also beats PFN with 3 and 6 extra sensors removed:
- +3 sensors removed: +0.025 [+0.014, +0.036];
- +6 sensors removed: +0.027 [+0.020, +0.035].

## Reading

- **The mixture wins, and so does a second transformer seed.** The linear pool of the fine-tuned LCT and PFN beats
  the fine-tuned transformer on fresh data, even at the Bonferroni bar (E14L). However, mixing in a second PFN seed
  gives the same gain (E16L fails, −0.001). On these targets the lifted model is as good an ensemble partner as
  another transformer, not a better one.
- **No complementarity claim.** A claim that the lifted sites add complementary information is **not** supported by
  v5.
- **What remains supported:**
  - the pooling theory: Hölder bounds, and the ranking of the linear pool over the geometric pool, consistent with
    the marginalization property;
  - the non-inferiority of the training-free pool.
- **What decides the paper:** protocol v6. There, a single UPT model must match the two-seed PFN pool at half the
  inference cost (P2) and beat the single PFN (P1).

## Replay

- Per-query predictions replay the committed per-episode scores to below 1e-4: `site_pool.py` asserts this.
- `pfn_s2_ft` replays to 1e-15.
- Fine-tuned checkpoints and cells for PM10, SO2 and O3 are under `runs_real/beijing_{pm10,so2,o3}/`.
