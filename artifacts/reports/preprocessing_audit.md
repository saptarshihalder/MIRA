# TabICL 2.2.0 preprocessing audit — 1 October 2026

The exact released numeric preprocessing path provides a specific explanation for the development result under deliberate zero/imputation collisions. This audit inspects source and existing saved predictions; it performs no new model inference, changes no runner code, and uses no confirmation seeds.

## Exact inspected distribution

Downloaded the [TabICL 2.2.0 released wheel](https://files.pythonhosted.org/packages/77/22/bbac2df399b8ed5a699bde38496ebb129a045c5d160fe52786727de932f3/tabicl-2.2.0-py3-none-any.whl) with `pip download --no-deps --only-binary=:all:`. Its SHA256 equals the digest in [versioned PyPI metadata](https://pypi.org/pypi/tabicl/2.2.0/json):

`c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4`

Exact source member hashes:

| Wheel member | SHA256 |
|---|---|
| `tabicl/_sklearn/preprocessing.py` | `2163a3bf6754127365a7aedac854d99a6f565545cc1ea7e63c68a596081b2976` |
| `tabicl/_sklearn/classifier.py` | `64f5ef7ef88a920a3f7a5f6e2d9d8dc08769ec343e931f910acc2d9c4eaa5534` |

The wheel, extracted members, and [provenance record](preprocessing_sources/provenance.json) are preserved locally. The [official repository](https://github.com/soda-inria/tabicl) is a source reference; this audit's identity is the released wheel, rather than an assumed match to its current branch. Both actual GPU environments record TabICL 2.2.0 and scikit-learn 1.8.0.

## Code path and implication

In the exact wheel's `classifier.py`, lines 503–504 fit `TransformToNumerical` on support inputs; line 729 transforms query inputs with that fitted encoder. In `preprocessing.py`, line 104 creates `SimpleImputer(keep_empty_features=True)`, and the ndarray branch assigns that imputer at line 127. This uses its default mean strategy without appended missing indicators. The class docstring's claim that ndarrays pass through unchanged is stale relative to this implementation.

`EnsembleGenerator.fit` then applies `UniqueFeatureFilter` at lines 1058–1060. The filter keeps only columns with more than one unique support value (lines 240–244), except for its fallback if every column would disappear. No original mask is supplied through this numeric preprocessing route.

For the constructed collision input, every finite nuisance value is exactly zero. Mean imputation maps both observed zeros and missing entries to zero; any entirely missing support column also receives the retained-column zero fallback. Thus differing nuisance mask patterns become identical numeric rows before the neural model. Constant filtering subsequently removes all eight nuisance columns, leaving U. Appended actual or shuffled mask columns remain binary and nonconstant. The source establishes this many-to-one transformation under the stated numeric construction; it does not establish a general deficiency of missingness-aware modeling or a pretraining-prior cause.

## Saved-artifact verification

Across all **15 task pairs** (three development seeds × five gamma values), Gaussian and collision NPZ files match exactly on `uc`, `yc`, `mc`, `uq`, `yq`, `mq`, `oracle`, `base`, and support/query row IDs. Their NaN patterns match; all finite collision nuisance values are zero; support/query IDs remain unique and disjoint. Source and environment hashes agree, and configurations differ only in `value_distribution`. Both runs completed all 135 requested cells with no cold-start cells. The [verification record](preprocessing_sources/paired_input_verification.json) preserves per-task checks and file hashes.

For each of seeds 40000–40002, saved collision/native TabICL query-probability arrays are **bit-identical across all five gamma values**. U and labels are paired across gamma; the source-derived collapsed native input is therefore identical. This prediction invariance supports the proposed localization without new inference or inspecting model internals.

## Development evidence and next control

Native-to-indicator expected-NLL gains for TabICL v2 are shown below. Intervals are paired 95% Student-t intervals over three independent tasks per gamma, with no multiplicity correction.

| Gamma | Gaussian gain | Collision gain | Collision interval |
|---:|---:|---:|---|
| 0 | -0.001662 | -0.001484 | [-0.008546, 0.005577] |
| 0.25 | 0.001790 | 0.021110 | [0.015095, 0.027125] |
| 0.5 | 0.003462 | 0.107083 | [0.096350, 0.117816] |
| 0.75 | -0.000468 | 0.269354 | [0.255583, 0.283124] |
| 0.9 | 0.000563 | 0.427346 | [0.407869, 0.446824] |

These are exploratory development findings in one synthetic family, not confirmation or broad real-data evidence. The effect is consistent with masks being erased by this preprocessing path under exact value collisions. Version comparisons alone cannot establish a causal prior explanation.

The decisive follow-up is the matched `imputed`/`imputed_indicators` pair and saved-probability comparison to `native`/`native_indicators`, initially on the same development seeds. Under exact-zero collisions, externally imputed native inputs should collapse identically to this encoder's outputs, while explicit masks retain the missingness signal. After these controls, freeze the hypothesis/configuration before using reserved confirmation seeds. Broader collision rates, near-mean values, other mechanism families, and real covariates remain necessary scope checks.
