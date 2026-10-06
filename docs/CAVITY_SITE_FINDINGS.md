# Trained cavity architecture: results and repair - October 6, 2026

**Both finite practical gates FAIL; no JMLR readiness or established novelty.** The core is a trained architecture, with frozen protocols and direct structural controls. Prior art includes [expectation propagation](https://tminka.github.io/papers/ep/) and [MVAE](https://arxiv.org/abs/1802.05335); neither cavity inference nor product-of-experts aggregation is claimed new. A support-conditioned learned site-revision contribution would require consistent advantage over the corresponding full-aggregate and static-site controls, plus stronger prior-art and real-data validation.

The first pilot trains twelve models on 512 new synthetic tasks: five nonlinear sensors with shared noise, 48 incomplete labeled support rows per task and source query masks deleting at most one sensor. Development uses 64 new tasks and all ten two-sensor deletions. All weights precede scoring. Site networks have 3,202 parameters, versus 8,514 for the ordinary MLP; equal nominal site parameter counts do not mean identical active degrees of freedom or FLOPs. Three seeds and 3,000 updates per model are retained.

All three cavity models collapse toward the prior even on source data: NLL about1.4158, prediction standard deviation .00018-.00072 and total precision1.007-1.009. Development cavity NLL about1.42948 loses substantially to the MLP and ridge. This diagnoses a source-learning failure, not general architectural impossibility. The batch is closed.

A separately frozen repair initializes Gaussian-moment-inspired sites from finite support statistics and exposes the full availability mask. All neural controls receive the same repaired ingredients; the larger MLP receives the initial site statistics too. The learned updater multiplies site precision and corrects its natural mean, replacing rather than accumulating sites. Cavity/full/static networks have3,362 nominal parameters; MLP9,154. This combined repair does not isolate initialization from explicit-mask effects. The old panel is never rescored: seed819202 supplies a fresh, adaptive development panel from the same family.

| Repaired model | Mean Gaussian NLL | Mean squared error | Mean nominal95% coverage |
|---|---:|---:|---:|
| Cavity | .415012 | .134259 | .918294 |
| Full aggregate | .410339 | .133328 | .915592 |
| Static sites | .422938 | .136447 | .922884 |
| Larger MLP | .473872 | .143260 | .895139 |
| Support ridge | .395915 | .126211 | .982813 |

Cavity gains over MLP are .032958/.060275/.083347 NLL for the three seeds (12.42% improvement in the mean). Gains over static are .008659/.003671/.011448. Against full aggregate they are .009983/-.004137/-.019867, and against ridge -.007418/-.015897/-.033975. Every predeclared practical gate fails. Even the favorable first full-aggregate contrast is below .01. Do not lower the margin, select seeds, call coverage calibrated, or attribute a causal correlation effect from this panel alone.

Across both runs:24 neural checkpoints,72,000 updates,276.078 seconds wall time on CPU, zero cloud calls/cost. Separate per-sensor audit implementations replay798,720 prediction rows and16,640 task/mask cells; maximum prediction differences1.19e-7 and9.54e-7. Audit durations5.328 and4.968 seconds. Source/context regeneration and all score/gate/manifest checks pass. No HAR labels, old panels or protected confirmation seeds were opened. Frozen sources and all failures remain retained.

**Close both configurations.** The next substantive issue is whether a learned joint conditional sensor model can represent residual correlations that independent positive-precision sites miss. This is an untested hypothesis with Gaussian-process, conditional-neural-process and multimodal latent-variable precedents, not a newly established contribution. Require a precise distinction and source-only headroom over an appropriate joint-covariance predictor before another neural fit. Actual real-data advantage remains mandatory. The October10 deadline does not change this evidence standard.

Reproduction: clean pre-fit checkout for each protocol, then its experiment script; audit scripts are added in the results commit. Existing outputs are refused to prevent overwrites. In the current checkout, run `experiments/cavity_site_audit.py` and `experiments/anchored_cavity_audit.py`. No GPU claim is made for these local CPU pilots.
