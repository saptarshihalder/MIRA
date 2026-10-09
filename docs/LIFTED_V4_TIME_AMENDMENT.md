# V4 execution-time amendment — October 9

This changes only the cloud execution limit, before the first scientific v4 fit or compared-model evaluation. Models, training steps, seeds, input files, package pins, margins, checkpoint selection and all endpoints remain fixed. The original source gates remain archived with their original decisions.

Actual A100 source smoke passed in 179.694 seconds; its padded estimate was 45,329.76 seconds. Actual NVIDIA L40S source smoke passed in 178.240 seconds. Timing on 128 repetitions of four source tasks avoids treating interpreter startup as per-task cost; it does not increase independent evidence. Its estimate is 20,520.962 seconds, including the original 25% safety multiplier. Both estimates fail the original 15,900-second execution limit.

The explicit L40S main extension permits 21,000 child seconds, 21,300 function seconds and 120 startup seconds, with zero automatic retries. The original smoke result and its failed original time gate must remain unchanged. A separate amendment identity and launcher are bound before execution; source/panel/weight identities must still match. No sign-wrapper or model change is authorized by this amendment.

Provider summaries after all source calls show September/October gross USD3.03766887; all apps have zero tasks. Absorb completed calls into the retained USD3.50 account provision. Reserve USD13.25 before this one main call and protect USD3 for reproduction: USD19.75 total within the USD20 cap, with USD0.25 unallocated. Worst-case function/startup compute is about USD12.926; the main provision also covers build, storage and egress contingency. Final invoices remain open. See `artifacts/manifests/lifted_v4_extended_budget_reconciliation.json`.

Main runs all three seeds and all frozen comparisons. No partial endpoint decisions or model selection. A timeout is an incomplete experiment, not permission to lower scientific requirements. This amendment supplies no efficacy, novelty or publication-readiness result.
