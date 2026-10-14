# Two universal properties behind lifted sites and the transformer site

This note supports protocol v5 (`LIFTED_CAVITY_PROTOCOL_V5.md`). Both statements are checked numerically in
`tests/test_lifted_site_pool.py`.

## Notation

- **Sensors:** a sensor set S, with latent z = (y, h) ∈ R^L and L = 1 + K.
- **Natural-parameter monoid:** a Gaussian factor over z is a pair (Λ, η), with Λ symmetric. Under addition, these
  pairs form a commutative monoid G; multiplying factors adds natural parameters.
- **Masks:** a mask m ∈ {0,1}^S lists the observed sensors. Masks sit inside the free commutative monoid N^S.

## U1: lifted sites are the free construction on sensors

**Universal property** (Mac Lane 1998, the free commutative monoid). For every assignment s: S → G of one site per
sensor, there is exactly one monoid homomorphism F: N^S → G with F(e_j) = s(j). It is

    F(m) = Σ_j m_j s(j).

**Consequences for missing-sensor prediction.** Write the posterior as prior + F(m).
1. **Characterization.** Suppose a predictor's natural parameters are additive over disjoint masks:
   F(A ⊔ B) = F(A) + F(B). Then it *is* a product of per-sensor sites; no other form exists. The static lifted network
   (`lift1_static`, and FA-Gaussian) is of this type. The test checks F(A ⊔ B) = F(A) + F(B) − F(∅).
2. **Free extension.** Defining a site for a new sensor extends F uniquely. This is the structural reason for exact
   handling of unseen masks and for zero-shot width transfer (protocol v2 E3, v3 E6).
3. **Cost.** All 2^P masks follow from P site evaluations plus one L×L solve per mask.
4. **Limitation.** F(m) depends on x_j only through s(j). Interactions across sensors can enter only through the latent
   dimension. This matches the known limits of sum decompositions: Wagstaff et al. (2019; JMLR 2022) show that the
   latent width must grow with set size for universality. Proposition 3 of the paper (nuisance dimension ≥
   shared-noise rank) is the Gaussian counterpart.

**What the data say about the limitation.** Fine-tuned Beijing PM2.5 NLLs, natural missingness (committed cells):

| Model | NLL | Additive over masks? |
|---|---|---|
| Static sites (`lift1_static`) | 0.259 | exactly additive (U1) |
| Cavity re-linearisation (`lift1`, three seeds) | 0.205 | additivity broken by the cavity |
| LCT | 0.122 | sites read transformer tokens |
| Transformer (PFN) | 0.087 | not additive |

So the free construction buys extrapolation, and the non-additive part buys real-data fit. The question is how to add
the second without losing the first.

## U2: the transformer as a site (naturality, or external Bayesianity)

**Pooling operator.** For experts p_1, …, p_M with weights w,

    Pool_w(p_1, …, p_M) ∝ ∏ p_i^{w_i}.

For every likelihood factor ℓ (for example, a newly observed sensor's site), the following square commutes:

    (p_1, …, p_M) ── ·ℓ each ──▶ (p_1 ℓ, …, p_M ℓ)
          │ Pool_w                        │ Pool_w
          ▼                               ▼
         p̄ ─────────── ·ℓ ────────────▶ p̄ ℓ        (normalized)

It commutes if and only if Σ w_i = 1.
- **Uniqueness.** Genest (1984) proves that, under mild regularity, log-linear pools are the *only* pooling
  operators with this property ("externally Bayesian"). See also Genest & Zidek (1986).
- **In category terms,** pooling is natural with respect to Bayesian updating.

**Consequences.**
1. **Weights.** β_LCT + β_PFN = 1 in protocol v5 is forced by the property, not chosen ad hoc. With any other sum, a
   sensor site added after pooling would count with weight Σw ≠ 1. The test exhibits the broken square.
2. **Same object as before.** The pool of the LCT and the transformer is again a lifted posterior: the prior, the
   tempered sensor sites and one tempered *target* site from the transformer. This is the generalized product of
   experts (Hinton 2002; Cao & Fleet 2014) and a power-EP site (Minka 2004).
3. **Proposition 5 (Hölder).** For every query and every outcome y,

       NLL_pool(y) = (1−β) NLL_LCT(y) + β NLL_PFN(y) + log Z,   with Z ≤ 1.

   The slack −log Z is a Rényi/Bhattacharyya-type divergence between the experts. It is zero only when they agree.

## Cheap architecture this suggests (next; not trained here)

**Pool once, complete by sites.** By U2, a sensor site multiplied in after pooling acts on the pool exactly as on
each expert. U1 makes sites additive. So the pooled predictive is computed once at the natural mask. Every extra
dropped or added sensor is then handled in closed form, by subtracting or adding its lifted site in natural
parameters, which is the EP cavity operation. One transformer pass therefore serves all 2^P sub-masks.

This is exact for the site part. For the transformer part it is an approximation, whose error is measurable on the
+3/+6 dropped-sensor conditions. It must be tested under its own frozen protocol before any claim.

## References

- Cao, Y. and Fleet, D. J. (2014). Generalized product of experts for automatic and principled fusion of Gaussian
  process predictions. arXiv:1410.7827.
- Genest, C. (1984). A characterization theorem for externally Bayesian groups. *Annals of Statistics* 12(3),
  1100–1105.
- Genest, C. and Zidek, J. V. (1986). Combining probability distributions: a critique and an annotated bibliography.
  *Statistical Science* 1(1), 114–135.
- Heskes, T. (1998). Selecting weighting factors in logarithmic opinion pools. *NIPS 10*.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*
  14(8), 1771–1800.
- Mac Lane, S. (1998). *Categories for the Working Mathematician*, 2nd ed. Springer.
- Minka, T. (2004). Power EP. Microsoft Research Technical Report MSR-TR-2004-149.
- Wagstaff, E., Fuchs, F., Engelcke, M., Posner, I. and Osborne, M. A. (2019). On the limitations of representing
  functions on sets. *ICML*.
- Wagstaff, E., Fuchs, F. B., Engelcke, M., Osborne, M. A. and Posner, I. (2022). Universal approximation of functions
  on sets. *JMLR* 23(151).
- Zaheer, M. et al. (2017). Deep Sets. *NeurIPS*.
