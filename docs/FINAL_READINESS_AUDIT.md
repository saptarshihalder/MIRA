# Final scientific readiness audit — October 9, 2026

**Verdict:** credible bounded research contribution; strongest-main-track NeurIPS/JMLR readiness is not established. This is one independent, bounded audit, not exhaustive novelty clearance or new efficacy evidence.

## Distinction from prior work

Learned EP messages are established by Heess/Tarlow/Winn (2013), Eslami et al. (2014), and Jitkrittum et al. (2015), already cited. Their full methods were not newly inspected here; avoid claiming learned cavity refinement itself is novel. [Official 2013 proceedings record](https://papers.nips.cc/paper/2013).

[NeuMiss](https://arxiv.org/abs/2007.01627) already derives missing-pattern Bayes predictors and unrolls a Neumann approximation, including MAR and self-masking MNAR. The candidate distinction is cross-deployment adaptation from labeled calibration support with shared per-sensor parameters, rather than mask-aware Gaussian inference alone.

[DIFNet v1](https://arxiv.org/html/2508.18854v1) extends IFNet's learned correlated-noise fusion to decentralized sequential state-space filtering. Its introduction explicitly permits changing sensor nodes/topologies: calling this prior art fixed-sensor is inaccurate. Distinguish episodic unknown calibration, not broad fusion, modularity, or extensibility. IFNet full text was not independently obtained.

[Set Transformer](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf), §§1,3–4, already supports invariant set processing and interacting multiple outputs. Per-token/set-valued prediction is insufficient novelty. The narrower candidate is an anchor-initialized Gaussian information-site head over target plus nuisance, with its restricted rank necessity result. A matched flexible set head and direct covariance/precision head remain decisive controls. Exact linear-Gaussian factor fusion is classical; application plus useful experiments does not alone establish conceptual novelty.

## Decisive evidence and defects

- Historical synthetic evidence supports efficient structured prediction and transfer from 5 to 16 sensors against the tested 203k transformer. It does not establish superiority to pretrained TabPFN/TabICL or broadly competitive transformer training.
- Real superiority fails: v2 E4b is −0.118 nats; v3 E7b fails non-inferiority, −0.021 [−0.093,+0.052]. E7a varies by training seed. Natural missingness is sparse; most dropout is simulated.
- The original network is factor-sign sensitive. The import audit's per-cell replay fails; permutation tests on a precomputed anchor do not prove invariance after refitting. The separate sign-averaging repair passes a source gate only and must not inherit historical efficacy.
- HMC fixes response signs from finite support. A nonzero generating slope does not make sign recovery certain; this is a sign-conditioned approximation, not a certified Bayes-optimal ceiling. Gap percentages remain reference-relative.
- v4 GPU probes establish interfaces, not efficacy; full frozen endpoints and actual pretrained-weight identity remain pending. Dequantization followed partial data inspection. A fresh official-style compile produced nine main pages, 28 total, with no overfull boxes; scientific gaps remain.

## Corrections applied

Only prose changed in `paper/lifted_cavity/{main,abstract_results,experiments,experiments_setup,intro_results,limitations,appendix,audit,checklist}.tex`: conditional permutation guarantee, precise filtering distinction, and qualified HMC reference. Textual HMC labels were also corrected in `experiments/lifted_cavity/make_paper.py` and `paper/lifted_cavity/generated/main_table.tex`. Numbers/macros are untouched. No generator, models, GPU experiments, compiler, or commits were run.

Local citation check: 41 `.tex` files, 39 distinct cited keys, 42 bibliography entries; zero unresolved or duplicate bibliography keys. This checks identifiers, not source accuracy. Figure-legend source changed from “Bayes-optimal (HMC)” to “Sign-conditioned HMC.” Saved `figures/main.pdf`/`main.png` outputs were not regenerated; their label refresh remains pending.
