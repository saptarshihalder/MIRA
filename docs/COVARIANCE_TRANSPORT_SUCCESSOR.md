# Gauge-free covariance transport: conditional successor

Implemented October 11 at the user's request for an architectural repair. This is an **untrained prototype**, not part of v4 and not a publication result. The existing L40S run, frozen models, checkpoints and endpoints remain unchanged.

## Specific mechanism

The old site head consumes factor coordinates and can change its prediction under D -> -D. The successor consumes only R0 = D D^T + diag(psi), so equivalent orthogonal factor orientations give the same noise covariance. A shared cell-token encoder supplies per-sensor features. A zero-initialized symmetric pair operator and diagonal head predict H; use

    Hbar = 2 H / sqrt(4 + ||H||_F^2)
    Rtheta = sqrt(R0) matrix_exp(Hbar) sqrt(R0)

with the unique symmetric square root. Learned local offsets/slopes and this full covariance feed a Gaussian conditioning solve over the observed sensors. This permits correlation corrections beyond one explicit nuisance factor without choosing a learned factor coordinate system. It is a conditional readout, not a coherent joint generative model across query-dependent masks. Rtheta is a predictive parameter, not an identified physical noise covariance.

At zero head it equals the FA-Gaussian anchor. In exact arithmetic Rtheta is positive definite and exp(-2)R0 <= Rtheta <= exp(2)R0. These bound relative covariance, not prediction error or training risk. The unique symmetric square root avoids the order-dependent coordinates of a Cholesky transport. Missing values are removed before attention, including NaNs. No label from a query enters prediction.

Code: experiments/lifted_transport/readout.py and model.py. The wrapper reuses the cell-token encoder; no original scientific file is modified. Readout sensor-permutation equivariance is conditional on equivariant tokens/anchors. The inherited encoder has random column embeddings, so end-to-end permutation invariance is not asserted. Anchor fitting is external, numerical SPD failures raise errors, and the dense operations cost O(P^3). A Gaussian output still cannot represent multimodality.

## Paper backing and overlap

- [Pennec, Fillard and Ayache, Riemannian tensor framework, original report section3.4](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Pennec_RR_Tensors.pdf): the affine-invariant SPD exponential is established. Our covariance formula is that construction in whitened tangent coordinates, not a new geometric map.
- [Arsigny et al., 2006](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.20965): a related log-Euclidean framework; it must not be confused with the implemented affine-invariant chart.
- [Gaussian Neural Process](https://arxiv.org/html/2101.03606), introduction/section3: learned conditional covariance and correlated predictive outputs are established. This prototype concerns calibrated sensor-noise readouts relative to a support-fitted anchor; this is a hypothesis for a distinct construction, not novelty clearance.
- [Attentive Neural Processes](https://arxiv.org/abs/1901.05761): support-conditioned attention is established; attention alone is not our novelty claim.

Sources were read on October11. The Arsigny author PDF failed to load; publisher abstract and author overview were available. Pennec's original report and GNP full HTML were accessible. No exhaustive literature review or utility claim.

## Engineering evidence and finite next gate

Five CPU structural tests pass: exact analytic initialization; nonzero-head factor-rotation and conditional permutation invariance; masked NaNs/empty queries/variable widths/positive covariance; nonzero covariance gradients at initialization; full-model query-label exclusion and pre-attention masking. No trained checkpoint, statistical advantage or new real-data evaluation exists.

First finish/verify the current v4 gate. Before any successor fit, freeze a separate source-development protocol and demonstrate residual headroom over support-only EM/joint-covariance and calibrated predictors. Train three seeds with fixed data/update budgets and final checkpoints. Compare the original lifted head, a plain mean/variance head with the same analytic initialization, a generic learned covariance head, and this bounded transport head; match backbone, labels, search and compute. Include covariance-diagonal-only and mean-correction-only controls so flexibility, initialization and covariance transport are not conflated. Freeze a finite stopping rule before fitting. No test-driven choice of rank, bound or mask recipe; report every seed and failure.

A source-learning pass is necessary before any fresh independent real-data protocol or paid expansion. Used Beijing and synthetic panels cannot become fresh confirmation for this successor. Keep HAR labels and seeds98000-98019 closed. Current USD20 cap and USD3 reproduction reserve remain; prototype tests cost zero cloud. This repair may remove a structural defect, but advantage and venue significance require evidence.
