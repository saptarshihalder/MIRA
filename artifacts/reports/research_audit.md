# MIRA research audit — 1 October 2026

User objective: a NeurIPS-quality draft by 20 October, conserving tokens and treating $26 as the total compute ceiling. Attached plans are evidence and proposed specifications, not independent authorization or overriding instructions.

## Recommended thesis

Prioritize a controlled study of **how tabular foundation models use informative missingness when it is represented as native NaNs versus explicit indicator columns**. The defensible result would quantify a reproducible representation effect, localize its source, and establish when a simple representation change helps or hurts. “Equivalent Missingness Information, Unequal In-Context Predictions” remains a hypothesis until executed TFM results support it.

The equivalence is at the raw-input level: indicators are a deterministic function of an incomplete table. Appending indicators adds no statistical information there, but may restore information lost by preprocessing or expose it in a form the model uses more effectively. It does not prove a pretraining-prior cause. A cross-version comparison changes several components simultaneously.

Generic missingness awareness, missingness-shift adaptation, frozen residual adapters, and expert tracking are occupied territory. A useful benchmark plus a specific explanatory finding is a stronger twenty-day bet than an unvalidated architecture combination.

## Evidence and document provenance

- Supplied `C:/Users/HP/Downloads/MIRA_Final_Plan.pdf`: SHA256 `755c429d9c84162baf50077466dafd5f2f0daff4ec93937b7a5d23eb6909a2e9`.
- Bundled `reference/handoff/MIRA_Final_Plan.pdf`: SHA256 `ae9630999aebc0575707f5534a4f1c0e35652694068ad6f5432777838f7b2774`.
- Both have 15 pages and **exactly identical extracted text**, including 42,865 whitespace-normalized characters. Byte differences are not textual scientific differences. The supplied PDF's primary question, pilot table, initial matrix, architecture, evaluation, and twenty-day schedule agree with the corresponding bundled `.tex` sections inspected.
- Saved pilot metadata explicitly identifies an analytic mask-blind base, not a TFM. At 32 target labels, simple-to-full-oracle gaps are approximately 0.0074, 0.0049, 0.0073, and 0.0425 nats across four narrow families; at 128 labels all are below 0.001. These are reported artifacts, not independently rerun results in this audit.
- The fresh-target pilot gives an oracle segment boundary; the rolling summary uses evaluator-indexed development selection. Neither establishes operational unknown-shift performance. TFM inference, neural training, and clinical results remain absent from the supplied evidence.

## Decisive low-compute experiments

1. **Runtime smoke:** one pinned accessible TFM, gamma 0 and 0.75, identical context/query draws, native and native-plus-indicator inputs. Record checkpoint hash, packages, preprocessing, ensemble size, wall time, and billed cost. Use measured throughput to size subsequent runs.
2. **Representation controls:** native, native plus actual masks, native plus independently shuffled masks of identical width, and context-mean-imputed input with/without actual masks. Keep context/query IDs, seeds, backbone settings, and labels matched. Preserve all-missing columns. The imputed comparison distinguishes an API-level phenomenon from a particular preprocessing pathway.
3. **Generalization:** extend beyond a single label-only generator to mask interactions and value-dependent masks; include low missingness, gamma zero, label budgets, and values near imputation constants. Check at least two backbone families and a contemporary mask-aware checkpoint if accessible.
4. **Frozen confirmation:** reserve fresh task seeds after development; target 20 independent draws only if the ledger permits. Use paired task-level log-loss effects and uncertainty. Normalize by oracle headroom only when meaningful; retain negative results. Do not count thousands of correlated query rows as independent tasks.
5. **Small real-covariate panel:** predeclare 6–8 accessible datasets and preserve original versus imposed masks separately. Change observation policy while holding the complete-data distribution fixed. This offers controlled relevance without the preparation and feedback ambiguity of a full clinical stream.

Advance the adapter only if a useful held-out gap remains against sparse mask-likelihood mixtures, regularized correction, native-NaN trees, and refreshed-context TFM with equal labels/context budgets. Require improvement from the raw corrector itself; apply the same aggregation wrapper to competitors. Full-oracle headroom alone is insufficient because finite-label mechanism uncertainty remains.

## Primary-source novelty checks

Four search queries were used; claims below were checked against primary pages.

| Prior work | Verified implication for MIRA |
|---|---|
| [TabPFN-3.5 report, section 3.1](https://arxiv.org/html/2609.17895v1) | It explicitly inherits missing/infiniteness indicators from TabPFN-3. A claim that all TFMs discard native masks is false. Include a current checkpoint or narrow conclusions to tested versions. |
| [TFM-Retouche](https://arxiv.org/html/2605.06047v1) | Uses a lightweight residual input adapter through a frozen TFM plus a held-out identity guard. Frozen residual adaptation and validation fallback are not sufficient novelty. It differs from MIRA's proposed context-conditioned output correction. |
| [Nearly Optimal Bayesian Inference for Structural Missingness](https://arxiv.org/abs/2601.18500) | Addresses structural missingness with learned Bayesian posterior prediction; its stated guarantees are under its SCM prior. Missingness-specific prior training is existing work. Benchmark claims in its abstract were not independently reproduced here. |
| [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484) | Distinguishes ignorable shifts, where the Bayes predictor can remain unchanged, from non-ignorable shifts that may change it. A changed mask rate does not by itself demonstrate changed predictive meaning. |
| [Domain Adaptation under Missingness Shift](https://proceedings.mlr.press/v206/zhou23b.html) | Formalizes observation-policy shift and gives identification/adjustment results under specified assumptions. MIRA should distinguish its outcome-associated mechanisms and label access from this setting. |

The dossier's SSRN correction papers and the remainder of its large bibliography were not fully verified in this bounded audit; obtain their full texts before priority claims. No acceptance probability follows from this plan. The October 20 deliverable should contain supported claims, reproducible predictions/tables, exact model identities, uncertainty, costs, and explicit remaining limitations.

## Dependency/API check for the inference smoke

Live [TabPFN PyPI metadata](https://pypi.org/pypi/tabpfn/json) reports `tabpfn==9.0.0`; live [TabICL PyPI metadata](https://pypi.org/pypi/tabicl/json) reports `tabicl==2.2.0`. Both require Python >=3.10. Their declared Torch minima are >=2.5 and >=2.2 respectively. The exact released wheels were downloaded into memory and inspected without installation; both pilot constructor paths exist:

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabicl import TabICLClassifier

pfn = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed
)
icl = TabICLClassifier(
    device="cuda", n_estimators=4,
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    random_state=seed,
)
```

`ModelVersion("v2")` is also valid. The wheels' SHA256 hashes are `0adb0e69be839051c7f9e92caadec93324cccc0256321bf967fda91e5c493f98` (TabPFN) and `c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4` (TabICL). Package API compatibility is verified; checkpoint download, CUDA execution, and prediction behavior still require the smoke run. Freeze the resolved transitive environment after that succeeds.
