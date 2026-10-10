October 11 bibliography update:42/42 bibliographic metadata entries verified, with explicit primary-catalog and contents evidence in LIFTED_REMAINING_REFERENCES.md; not full-text or novelty clearance.

# Bounded mathematical audit — 2026-10-10, amended-text recheck

Scope: the four propositions and their appendix proofs. The initial audit also read the bounded reference-verification memo. No model, data, experimental results, or external services were inspected. All four amended statements receive conditional PASS below; this is not a whole-paper readiness or independent formal-verification declaration.

## Proposition 1 (Scalar experts): PASS under the stated assumptions

main.tex:117,123–126 now explicitly supplies mutual independence, positive sensor-noise scales, nonzero target slopes, and equality for every reading. These resolve the initial audit's missing-hypothesis verdict. appendix.tex:7–19 correctly forces singleton likelihood sites and derives pair exactness iff $d_jd_k=0$. Positive noise makes the conditional covariance invertible; nonzero $a_j$ makes the forced precision positive.

The previously identified counterexample $u=y$, $a=d=(1,1)$, with independent unit-variance errors, is now excluded by mutual independence. Previously checked arithmetic remains consistent with the example: singleton coefficient $0.4993757803$, pair coefficient $-0.9588145565$. The conclusion concerns mask-independent Gaussian experts over the scalar target.

## Proposition 2 (Exactness): PASS under the stated assumptions

main.tex:148,164–166 now specifies an independent nonsingular latent prior, $v_y>0$, $\psi_j>0$, and mutually independent errors independent of the latent vector. These resolve the prior regularity omissions. appendix.tex:25–32 then gives the correct Bayes factorization and natural parameters for every fixed subset, including the empty subset.

The former example $A=0,e=y,x=y$ is excluded by error/latent independence. The distinction between fixed-subset inference and informative observation masks remains explicit and appropriate.

## Proposition 3 (Lifting dimension): PASS within the declared static linear-Gaussian class

main.tex:174–177 now states mutual latent/error independence, positive error variances, fixed $w_j,c_j$ and positive fixed $\psi_j$, independence of site parameters from readings/subsets, a proper nonsingular Gaussian prior, and equality for every reading and every subset of size at most two. These resolve the initial literal-scope objections. The positivity restriction correctly applies only to $\psi_j$, not to $w_j$ or $c_j$.

appendix.tex:41–46 validly identifies the sensor covariance from singleton/pair conditionals. The explicit contradiction assumption at appendix.tex:48 justifies the rank bound used in the disjoint-block cofactor argument at lines 49–54.

For history, the excluded dependent-error example used $a=\mathbf1$, $D=(.1,.2,.3,.4,.5)^\top$, independent standard Gaussian $e,v,y$, and $u=-D^\top e/2+\sqrt{1-\|D\|^2/4}\,v$. It satisfies the row-deletion condition but has $\operatorname{Cov}(Du+e)=I_5$; independence now excludes it. Necessity is not established outside the stated static class.

## Proposition 4 (Structural guarantees): PASS conditional on valid, equivariant anchors

main.tex:225–229 now requires fully equivariant anchor parameters and support statistics with consistent nuisance orientation, and uses the infimum of training risk. Both initial wording issues are resolved. appendix.tex:61–73 supports mask exclusion, conditional permutation invariance, and exact recovery of the anchor at zero corrections. It proves neither an attained risk minimum nor training convergence. The caveat at main.tex:233–236 appropriately preserves the original model's factor-orientation limitation.

No further mathematical correction was identified in this bounded recheck.
