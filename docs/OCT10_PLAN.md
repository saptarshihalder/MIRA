# October 6–10 plan: one paper, the lifted cavity network

This plan supersedes the open-ended "next mechanism" search in `NEXT_TRAINED_MECHANISM_HYPOTHESIS.md` and
`OCTOBER10_FINALIZATION.md`. Read `LIFTED_CAVITY_FINDINGS.md`, then `LIFTED_CAVITY_CONFIRMATION.md`.

**Status (October 7).**
- Protocol v1: all five endpoints passed.
- Protocol v2: five of six passed. E4b failed: a fine-tuned TabPFN-v2-style transformer beats the lifted network on
  Beijing.
- The NeurIPS-format draft is compiled at `paper/lifted_cavity/main.pdf`; results are summarized in
  `LIFTED_CAVITY_RESULTS.md`.
- Protocol v3 (lifted sites as the transformer's output layer) is running.

**Decision needed first.** A parallel line, MIRA-Circuit, was built in another session on October 6. It is described
in the Claude project doc `claude/mira_circuit_sprint.md` and is not in this repository. MIRA-Circuit compiles Boolean
missingness circuits into one column for a frozen predictor. Its synthetic pilot gates pass, but its TabPFN/TabICL
and real-data steps have not run yet. This repository's earlier six-dataset panel found no real-data indicator
benefit. The lifted cavity network already has real-data confirmation.

Choose one main paper for October 10. This plan applies if it is the lifted cavity network. MIRA-Circuit's frozen
Colab steps (~2–4 GPU hours) can still run in parallel as a second paper; their results do not change this plan.

**Working title.** *Lifted Cavity Networks: In-Context Sensor Fusion with Missing Sensors and Shared Noise.*

**Claim.** A product of per-sensor Gaussian sites over a target-plus-nuisance latent improves on the strongest
support-only closed forms and on a larger non-modular network trained identically. The sites are:
- exact at initialisation for factor-structured noise;
- refined by learned, cavity-conditioned re-linearisation;
- trained across tasks.

The improvement holds for unseen missingness patterns, for more sensors than seen in training, and on real
multi-sensor data.

## Rules for the coding agent (paste into its instructions)

1. **Scope.**
   - Work only on this paper until October 10.
   - No new mechanisms or pilots.
   - Earlier probes go into one appendix table.
2. **Strong controls.**
   - Every comparison includes the support-only EM-Gaussian and FA-Gaussian closed forms.
   - Mean-imputed ridge never gates anything.
3. **Oracle headroom first.**
   - Synthetic tables report the exact oracle and the share of the closed-form→oracle gap closed.
   - Never train a variant whose oracle headroom over the strongest closed form is below the margin.
4. **Adequate learning.**
   - Amortised models use the shared recipe in `experiments/lifted_cavity/train.py`: ≥60k tasks, 20k steps.
   - One small pilot is not evidence against an idea.
5. **Audit discipline, compressed.**
   - One page of protocol per batch, committed before scoring.
   - Hashes and replays as before.
6. **Data boundaries.** Panels with seeds 20261007–20261012 and 20261101–20261103, plus the Air Quality CO and
   NO2 builds, are used data. Do not tune on them.

## Schedule

| Date | Work | Exit condition |
|---|---|---|
| Oct 6 (evening) | Merge branch `lifted-cavity` into `codex/mira-research`. Run `tests/test_lifted_cavity.py`. Reproduce `lift1` seed 1 on panel 20261007 (~13 min CPU). Download the official UCI Air Quality file and compare its hash. | Reproduction within 1e-3; hash checked (if it differs, rebuild and report) |
| Oct 7 | **Second real dataset with natural per-sensor missingness** (recommended: UCI Beijing Multi-Site Air Quality, dataset 501): predict one station's log PM2.5 from the other 11 stations. Weekly episodes, 48 labeled support hours, temporal split. Freeze a one-page protocol before fitting: identical fine-tuning recipe, `lift1` vs EM/FA-Gaussian and `anchor_mlp`, endpoint at the dataset's natural missingness. 11 inputs also tests width transfer on real data. | Endpoint reported, pass or fail |
| Oct 8 | Figures from `make_figure.py` and `make_report.py`. Related-work audit against learned EP messages (Heess 2013; Eslami 2014; Jitkrittum 2015), MVAE, NeuMiss, IFNet/DIFNet, KalmanNet and CNP/PFN. Make the distinctness statement precise. | Each claim mapped to one table cell |
| Oct 9 | Rewrite `paper/main.tex` from scratch: introduction; method with Propositions 1–2; synthetic, transfer and real-data experiments; limitations; appendix with the oracle audit and a one-table summary of the closed probes. Fix PDF compilation (tectonic or Overleaf). | Compiled PDF, every number traced to a file |
| Oct 10 | Clean-checkout reproduction of Tables A–D and the confirmation table, final readiness review, push. | Deliverable package |

## Venue realism

- **By October 10:** a complete, arXiv-ready draft is achievable.
- **Main track:**
  - NeurIPS 2026 has passed.
  - JMLR has long review and expects a mature body of work, so it is not a sprint target.
- **Realistic targets:** TMLR (rolling) once the second real dataset is in, or ICML 2027 (January deadline) with broader
  real-data evidence and the full novelty audit.

## If the second real dataset fails

Report it. The real-data claim then rests on one device (CO and NO2 targets), and the paper says so. Do not reopen
the mechanism search.
