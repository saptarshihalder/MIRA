# RL module completion review — October 4, 2026

**Complete as a bounded negative-result module; overall manuscript remains partial and is not submission-ready.**

Completed: frozen protocol and cohort boundaries; 135 checkpoints across three seeds; parameter/initialization/update-matched neural controls; strong conventional references; support-only selection; saved raw/selected predictions and failures; exact checkpoint replay and score audits; methods, results and limitations in the existing manuscript.

Final review validates 357 file hashes and the action-independent-baseline/expected-policy-gradient identity (maximum gradient difference 1.49e-8). One frozen REINFORCE fit was independently rerun from its original fitting inputs: every weight matches exactly, with no query labels supplied. This is a reproduction check, not another scientific trial. Commands below print a compact report without modifying the original results. The optional output directory must not already exist.

```powershell
python experiments/crossfit_v1/rl_review.py --retrain-one --output artifacts/reports/my_rl_review
```

Use the saved runtime versions in artifacts/reports/annual_rl_completion/environment.json. Verification uses tracked prepared inputs and models; rebuilding those inputs additionally requires the original public data pinned in annual_rl_freeze.json. Missing raw reconstruction inputs are explicitly reported. Complete all-model and baseline replay commands remain in ANNUAL_RL_FINDINGS.md. Exportable descriptive results are in annual_rl_completion/results.csv; seed ranges describe training variability, not population confidence intervals.

Post hoc diagnosis, without refitting: the RL policy assigns about 90–93% probability to zero correction. Its gain over its own calibration anchor is only 0.000021–0.000031 NLL. This supports the scoped finding of little useful learned correction; it does not establish convergence or prove all RL approaches ineffective. The 384 fitting/128 selection split versus conventional 512-label refits remains disclosed. No architecture-superiority or safety claim is warranted.

Still required for the requested venue target: a distinct useful trained-model contribution; replicated gains beyond strong controls; external unused evaluation; full novelty review; and successful PDF/layout verification. These are scientific and production gaps, not completed checkboxes. The acquisition-model proposal remains conditional on novelty/data audit. Do not retune this closed RL panel until positive.

Budget: zero new cloud cost. The reproduction fit used less than one CPU second, excluding verification startup. Modal provisions $16.45 plus protected $3 remain unchanged under the $20 cap. Native LaTeX compilation remains environment-blocked.
