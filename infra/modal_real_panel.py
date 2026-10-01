"""One bounded frozen panel dataset per Modal call; no retries or dataset selection."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT / "src"))
from infra.modal_mechanism import app, dispatch
from mira.artifacts import file_sha256, source_hashes


@app.local_entrypoint()
def panel(dataset: str, run_id: str):
    path=ROOT / "configs/real_panel_v1.json"
    plan=json.loads(path.read_text())
    digest=file_sha256(path)
    if not plan["frozen"] or digest != path.with_suffix(".sha256").read_text().strip() or dataset not in plan["datasets"]:
        raise ValueError("Frozen listed dataset required")
    if source_hashes() != plan["runner_source_sha256"]:
        raise ValueError("Frozen runner source changed")
    for filename, expected in plan["data_sha256"][dataset].items():
        if file_sha256(ROOT / "artifacts/data/real_panel" / dataset / filename) != expected:
            raise ValueError("Frozen input changed")
    dispatch(run_id,["--dataset",dataset],"panel",{"phase":"real_covariate_panel",
        "dataset":dataset,"model":" ".join(plan["models"]),"frozen_protocol_sha256":digest})
