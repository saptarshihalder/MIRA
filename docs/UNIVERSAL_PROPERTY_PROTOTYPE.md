# Universal properties applied to a cheap conditional fusion model

October 11, 2026. User-requested prototype; **untrained**, separate from the frozen v4 study and the dense covariance-transport candidate. No scientific panels, real labels, trained checkpoints or paid compute are used. This is a candidate construction, not a cleared novelty or publication-readiness claim.

## What the mathematical idea actually contributes

The linked [universal-property explanation](https://en.wikipedia.org/wiki/Universal_property) suggests defining a computation through its composition law. Here the concrete construction is the free commutative monoid of sensor-evidence atoms. For fixed model, support, anchor, query and component, assign sensor i the message

    T_i = (w_i w_i^T / psi_i,
           w_i u_i / psi_i,
           u_i^2 / psi_i + log(2 pi psi_i)).

There is a unique additive extension to finite multisets: T(E) = sum_i T_i. Hence disjoint chunks give the same statistics regardless of order or grouping. Add the Gaussian prior only once, in the decoder. Keep the scalar third term: omitting it would give incorrect mixture gates. The posterior itself is not an additive map, and no universal approximation or statistical superiority follows from this universal property.

Deletion is subtraction of a known included contribution in the ambient additive group. The implementation checks query-context identity and sensor-index inclusion. It rejects overlapping merges and foreign-context evidence. Its IDs refer to columns, not physical devices; two duplicated columns are not automatically recognized. Evidence must be produced by the same Query object, unmodified. Updating parameters/support invalidates caches. Arbitrary subtraction does not promise a valid Gaussian.

## Implemented architecture

Code: `experiments/compositional_mixture/model.py`, default **1,317 trainable parameters**, three mixture components, two added noise factors.

1. A shared small encoder pools observed support rows separately for each sensor. It uses support labels and invariant anchor statistics. Missing NaNs are removed before neural operations. There is no sensor ID, query label or pooling across query sensors.
2. For each sensor and component it learns response offset/slope, diagonal-noise scale, a scalar scale on the existing factor loading, and two extra factor loadings. Existing factor coordinates never enter a learned scalar network; D -> DQ for orthogonal Q preserves predictions. Unlike the prior dense transport proposal, no P-by-P exponential/eigendecomposition is required.
3. Each component defines a coherent joint Gaussian over target and sensors, conditional on fixed support/anchors. Integrate its small shared latent exactly, obtaining both the target conditional and the observed-sensor marginal likelihood.
4. Normalize those likelihoods with learned global component priors. This produces query-dependent mixture weights and a nonlinear predictive mean, while retaining a mixture density. Train using the actual mixture NLL, never a moment-matched Gaussian NLL.

For component c, centered latent z has identity prior. Its first coordinate is (y-my)/sqrt(vy). Compile x_i = location_ci + w_ci^T z + epsilon_ci, epsilon independent conditional on c,z, with positive diagonal noise. Set J_c = I + sum_i J_ci and h_c = sum_i h_ci. Then

    log p(x_observed | c, support)
      = -0.5 [sum_i scalar_ci + log det J_c - h_c^T J_c^-1 h_c].

Decode y from the first coordinate of N(J_c^-1 h_c, J_c^-1); weight components by their observed marginal likelihood. Correlated sensors share nuisance latents and must not be treated as independent target-only experts. Absence contributes no evidence: informative missingness is not modeled.

Readout cost is O(M P L^2 + M L^3), L=1+K+r; for fixed small ranks/components it grows linearly with sensor count P. Encoding support costs additional work linear in support rows and sensors. Anchors must already be supplied; their fitting costs are excluded. Chunk summaries retain O(M L^2) state per query. The CPU benchmark compares only identical probabilistic readouts; it is not a trained-model speed/quality comparison.

## Guarantees and limits

- Exact arithmetic: associative/commutative disjoint evidence accumulation, identity at empty evidence, known-subset removal, positive posterior variances, and orthogonal invariance of the original factors.
- Recompiling any sensor subset gives the same corresponding component rows, provided support rows, anchors and parameters stay fixed. The resulting finite joint mixtures agree under marginalization. Conditioning on fewer observed sensors can and normally does change the posterior. This is not a claim that arbitrary marginals determine a unique joint distribution.
- Support-row and sensor permutation invariance hold for this encoder with correspondingly permuted fixed anchors. Refitting FA on different subsets can change anchors and is outside these claims.
- Small distinct random output initialization avoids dead gradients in new factors and identical components. Initialization is deliberately **not exactly the FA solution**. An initialization-matched control is required.
- The conditional family is still a small mixture of Gaussian factor models; support-only encoders may miss dependence beyond the supplied anchor. Mixture collapse, finite-rank bias, fitting cost and real utility remain unresolved. Solves fail explicitly; no hidden jitter or retries.

## Prior work and novelty boundary

Primary sources checked October 11:

- [Deep Sets, Zaheer et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf): shared encoding and commutative pooling are established. The monoid interpretation is not a new neural architecture by itself.
- [Ghahramani and Hinton, Mixtures of Factor Analyzers (1996, revised 1997)](https://www.cs.utoronto.ca/~hinton/absps/tr-96-1.pdf): low-rank Gaussian mixtures are a direct antecedent and mandatory control. We do not claim their probabilistic family as new.
- [Bishop, Mixture Density Networks (1994)](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/): learned mixture densities and nonlinear prediction are established.
- [Conditional Neural Processes, Garnelo et al. (2018)](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf): support-conditioned prediction is established; the paper explicitly distinguishes it from fully consistent Bayesian conditioning.
- [The Gaussian Neural Process, Bruinsma et al.](https://arxiv.org/html/2101.03606): learned conditional covariance is established. Restriction consistency must be scoped to fixed support, not claimed as a new general stochastic-process theorem.

The hypothesis worth testing is whether this small support-amortized, factor-invariant, exactly composable mixture retains predictive quality while reducing compute and improving consistency under changing sensor sets. That combination is a design hypothesis; neither the universal property nor these software checks establish a NeurIPS-level novelty claim. A broader direct prior-art audit is still required.

## Finite next decision, before any fit

Finish and audit the existing v4 experiment first. Preserve its failures and all used-panel status. Before successor training, freeze a separate source-development protocol, data boundaries, checkpoint identities, all three seeds, budgets and stopping margins. Establish source-only residual headroom over strong calibrated joint-Gaussian and mixture-factor controls. Do not choose ranks, components or margins from used evaluation outcomes.

Mandatory comparisons: support-fit mixture of factor analyzers; neural mixture-density/set model; single-component version; current lifted model; and an initialization/data/update/search-budget-matched neural mixture with an unconstrained or mask-dependent head. Report parameter count, total training/inference cost including anchors, exact mixture NLL, accuracy and calibration across sensor counts. Include diagonal-only/no-extra-factor and fixed/uniform-posterior-gate ablations to isolate correlation and mixture gating. No claim that software consistency alone improves prediction.

No new paid calls until completed provider charges are reconciled and a worst-case reservation fits the USD20 cap while preserving USD3 reproduction reserve. No old panel becomes fresh confirmation. Only a source-development pass and a newly frozen real-data protocol can authorize fresh successor evaluation.

## Engineering results recorded October 11, 2026

The parent agent ran `../.venv-compiler-cpu/Scripts/python.exe -m pytest tests/test_compositional_mixture.py -q`: **7 structural tests passed in 12.17 seconds**. These constructed fixtures check dense Gaussian posterior/evidence and mixture-NLL agreement; disjoint merging, deletion and context/duplicate guards; fixed-anchor subset recompilation and permutations; factor rotations, masks, empty queries and the query-label boundary; finite nonzero gradients; a constructed nonlinear mixture mean; and explicit rejection of invalid masks and observed nonfinite values. Backpropagation in a structural fixture is not an optimizer step or evidence of successful learning. The default model has **1,317 trainable parameters**.

The saved report is `artifacts/reports/compositional_mixture/engineering_benchmark_20261011.json`. It compares the compositional calculation with the corresponding dense Gaussian component means, variances and log evidence on constructed random tensors. Every recorded width is shown below; times are medians of five samples after one warmup, using one CPU thread, float64, three components and latent width five (Python 3.11.9, PyTorch 2.8.0+cpu; fixture seed 20261118).

| Sensors | Compositional median (ms) | Dense median (ms) | Dense / compositional | Maximum absolute discrepancy |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1.1113 | 0.4799 | 0.432x | 1.4211e-14 |
| 128 | 1.7551 | 2.0262 | 1.154x | 5.6843e-14 |
| 512 | 2.6683 | 42.7035 | 16.004x | 2.2737e-13 |

At 32 sensors the compositional path is approximately **2.32 times slower**; at 128 it is approximately 1.15 times faster, and at 512 approximately **16 times faster**. These are local readout timings for this fixture and implementation. Anchor fitting, support encoding, training and end-to-end trained-model quality are excluded; this does not establish a general 16-times model speedup. The compositional timed call also produces mixture weights, whereas the dense reference returns component quantities.

There were **zero training steps and USD0 cloud cost** for this prototype. No scientific fit or evaluation panel was run. The primary-source mathematical audit above supports the Gaussian identities and identifies established antecedents; it provides **no novelty clearance, learning-success claim or publication-readiness evidence**. The read-only implementation audit found no concrete defects in likelihood normalizers, masking, factor gauge handling or subset consistency within the stated fixed-support/fixed-anchor scope.

`artifacts/manifests/compositional_mixture_prototype_20261011.json` records these results and SHA-256 hashes of the model, package initializer, benchmark, structural tests, saved benchmark report and this final document. The test count and duration are the parent agent's completed-run report; this documentation step did not rerun tests or benchmarks.
