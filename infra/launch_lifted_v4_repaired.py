"""Explicit packaging repair: import the frozen worker by its available package name.

The failed bootstrap remains recorded. This launcher never changes model code,
panels, worker bytes, recipes, margins, or retries. Execute with modal run -m.
"""
import hashlib
import json
from pathlib import Path
import modal
from infra import modal_lifted_v4 as w

app = modal.App('mira-lifted-v4-a100-packaging-repair')
workers = {stage: app.function(image=w.image, gpu='A100-40GB', cpu=(2, 2),
    memory=(16384, 16384), timeout=w.FUNCTION_SECONDS[stage], startup_timeout=120,
    min_containers=0, max_containers=1, scaledown_window=2, retries=0,
    volumes={'/results': w.volume}, serialized=True, name='lifted_v4_' + stage)
    (w.worker_factory(stage)) for stage in w.CHILD_MINUTES}

@app.local_entrypoint()
def launch(stage: str = 'smoke'):
    if stage not in workers:
        raise ValueError('Only fixed smoke/main stages are permitted')
    # Separate authorized engineering attempt; the original submission is closed.
    w.RESERVATIONS = {'smoke': ('lifted_v4_modal_source_smoke_packaging_repair', 1.),
                      'main': ('lifted_v4_modal_main', 11.5)}
    repo = Path(__file__).resolve().parents[1]
    ledger, attempt, run_id = w.authorize_submission(repo, stage, w.expected)
    repair = dict(worker_module='infra.modal_lifted_v4',
        original_failure='modal_lifted_v4 not on remote Python import path; no worker body executed',
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_code_changed=False, automatic_retries=0)
    try:
        call = workers[stage].spawn(stage, w.expected)
        w.record_submission(ledger, attempt, run_id,
                            dict(status='running', function_call_id=call.object_id, packaging_repair=repair))
    except Exception as error:
        w.record_submission(ledger, attempt, run_id, dict(status='failed_submission', error=str(error), packaging_repair=repair))
        raise
    print(json.dumps(dict(stage=stage, function_call_id=call.object_id,
                          root=w.RESULTS.as_posix(), packaging_repair=repair)))
