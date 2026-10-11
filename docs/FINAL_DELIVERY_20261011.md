# October 11 research delivery

**Not submission-ready for NeurIPS/JMLR.** This is the preserved research draft, implementation and audit record. There is no submission or acceptance claim.

## Completed and retained

- The fixed three-seed L40S continuation completed with exit0 in 7,647.417 seconds. All 24 learned checkpoint files and the 35 expected score archives were collected locally, together with the fixed pretrained binary. Collection is not independent verification.
- The original packaging failure and the local observer/worker timeout are preserved. The timeout occurred before collection; a read-only provider query recovered the completion status. The same hash-frozen evidence tools then performed collection and stopped at verification.
- The paper retains its historical v1-v3 results, including real-data transformer losses, seed sensitivity, the original factor-sign replay discrepancy, and the dequantization/protocol limitations. All 854 historical numerical macros are unchanged.
- The same `paper/lifted_cavity/main.tex` builds to nine main pages, references starting on page10, 30 pages total. Abstract, limitations and appendix disclose the failed v4 audit. Same-source Tectonic succeeds; native isolated compilation still cannot resolve the local style. No replacement editor/tab was opened.
- Gross September+October Modal metering is USD10.95127410, including failed work. All listed provider task counts are zero. USD12 covers completed gross plus contingency and USD3 remains reserved for reproduction: USD15 provisioned within the USD20 cap. Final invoices remain open; credits do not make the compute free.

## What blocks the new results

The frozen driver checks fine-tuning completion before a fresh fit, but does not repeat `ft_done()` after the subprocess returns successfully. Since that helper also writes the checkpoint-identity sidecar, fresh fine-tunes can reach scoring without their required pre-score hash binding.

All twelve seed2/3 fine-tunes lack that sidecar in the collected bundle. A direct remote listing confirms absence for one representative directory; a broader read-only listing hit provider rate limits and was not retried. Seed1's six sidecars exist in the final bundle but were absent from its original recovery snapshot: later existence is not evidence of pre-score creation.

The independent verifier correctly stops **before endpoint arithmetic**, leaving E8-E12 undisclosed by this audit. CPU prediction replay, integration, seed/secondary-metric analyses and final review ZIP were not run. There is no basis to call these endpoints passed or failed. Source and scores remain frozen; no backfilled checkpoint identities, new fitting, altered thresholds or relaxed verifier were used.

The current development driver now binds completed fine-tunes before scoring and refuses already-scored unbound runs. Four mocked orchestration tests pass. This is a prospective engineering fix; the executed source archive and scientific history remain unchanged.

Key records:

- `artifacts/manifests/lifted_v4_checkpoint_binding_failure_20261011.json`: root cause, current byte hashes, independent audit and limits.
- `artifacts/reports/lifted_v4_launch/final_strict_verification_failure_20261011.json`: exact verifier failure.
- `artifacts/manifests/lifted_v4_local_handoff_recovery_20261011.json`: observer timeout recovery without training retry.
- `artifacts/manifests/lifted_v4_completed_budget_20261011.json`: completed-cost reconciliation.
- Local full bundle: `artifacts/runs/lifted_v4_final/`; original source/panels remain under `artifacts/inputs/`.

## New architectural work, separate from paper evidence

The universal-property suggestion led to a 1,317-parameter support-conditioned Gaussian-mixture prototype, with exact composition of evidence and consistent sensor-subset restriction for fixed anchors. Seven structural/oracle tests pass. Its identical-model readout benchmark is slower than dense inference at32 sensors, about1.15x faster at128 and16.00x faster at512 on one local CPU configuration. These are engineering results with no training or predictive advantage established. See `UNIVERSAL_PROPERTY_PROTOTYPE.md`; its direct mixture-factor, Deep Sets and neural-process precedents are explicit. The earlier covariance-transport prototype also remains untrained. Neither is a current-paper efficacy result or a novelty clearance.

## Remaining scientific work

Independent forensic replay could test whether the collected checkpoint bytes reproduce existing scores, but it cannot retrospectively supply pre-scoring attestation. Such a separately labeled analysis must preserve this failed protocol gate. A new confirmation requires a genuinely eligible frozen protocol and cannot relabel these panels as untouched.

For a successor architecture, first establish source-development headroom and freeze three seeds, matched mixture-factor/neural controls, budgets and stopping margins. Publication significance also needs stronger real-domain coverage and a direct novelty audit. No further paid run is launched by this delivery. The completion automation ends with this delivery; further work needs an explicit continuation rather than recurring deadline extensions.
