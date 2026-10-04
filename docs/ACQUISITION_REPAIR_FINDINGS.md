# Successor-repair probe: closed, October 4, 2026

The trained residual repair failed its frozen mechanism gate. This is a synthetic negative result, not real-data utility or evidence of novelty. Protocol/source were committed as `6f6dd00` before fitting; see `ACQUISITION_REPAIR_PROBE.md` and its frozen manifest.

| Method | Mean terminal NLL + acquisition cost |
|---|---:|
| Learned residual repair | 0.681428 |
| Matched direct predictor | 0.693741 |
| Availability-blind predictor | 0.693252 |
| Reuse source successor vectors | 0.680278 |
| Exact availability-aware reference | 0.613253 |
| Exact optimum | 0.613253 |

Repair improves on the direct network by 1.775%, below the prespecified 2%, and loses to source-vector reuse. Exact availability-aware planning improves over source-vector reuse by 9.85%; this is privileged-reference headroom, not a learned advantage. Its equality with the optimum applies only to this enumerated probe.

Nine 1,029-parameter networks (three models, three seeds) each received 600 updates using identical source population teachers, initialization and minibatch schedules. There were 3,456 fitting examples. The 24 evaluated cost/availability combinations per checkpoint share one four-bit population; they are not independent datasets. Exact population conditionals, terminal risks and source-policy vectors are privileged information. Deployment reads observed values, availability, budget and prices; hidden values are revealed only by selected actions. No real dataset or protected confirmation seed was accessed.

`experiments/acquisition_repair_audit.py` reloads all nine checkpoints and independently enumerates 216 population-risk cells. Maximum replay error is 3.33e-16; teacher feasibility, source hashes and gate recomputation pass. Audit uses independent enumeration, not an independent human or agent review. Runtime was 9.969 CPU seconds for the probe; cloud calls and charges were zero. Artifacts, traces, source controls and hashes are in `artifacts/reports/acquisition_repair_probe/`.

Close this probe without retuning these cells. The remaining hypothesis is structural planning with learned components and hard constraints at every future step, detailed in `FEASIBLE_SUCCESSOR_PLAN.md`. It still needs a distinct contribution and real-data evidence. October 7-16 milestones are pending, not completed.
