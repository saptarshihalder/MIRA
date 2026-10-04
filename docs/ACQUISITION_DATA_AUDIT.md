# Acquisition data gate — October 4

Current state, October 4: Wine Quality raw data are now used development, prepared under wine_acquisition_data_audit.json and evaluated in wine_acquisition_v1. The other seven entries remain metadata-only. The original registry below describes its earlier stage. ID447 is hydraulic condition monitoring, not Sensorless Drive; see ACQUISITION_ELIGIBILITY_CHECK.md.


Metadata-only registry saved at artifacts/manifests/acquisition_candidate_panel.json; official UCI responses are hash-pinned. Eight domain candidates were requested. No raw rows, labels or outcomes were downloaded/scored. Prior WDBC, ionosphere, banknote, spambase, sonar and heart panels, plus native diabetes/BRFSS/roads, are already used and cannot supply fresh confirmation. A repository name search in docs/configs/experiments/manifests found no matches for the eight candidate names; this cannot exclude undocumented or pretrained-model access.

Gate UNRESOLVED. Metadata completeness does not establish identity independence, current license, absence of target proxies, feasible measurement costs or a frozen raw-data version. Subject grouping is especially important for human activity/speech; derived features may share one acquisition. Eight domain labels are not proof of eight statistically independent families. Raw split/duplicate/proxy and license checks remain required before training. Metadata reports are not external validation. Novelty currently fails at the broad-claim level, so no labels were consumed for a successor experiment.
