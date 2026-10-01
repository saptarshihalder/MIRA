"""Execute a committed, hash-frozen confirmation plan through the bounded wrapper."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from infra.modal_mechanism import app, dispatch
from mira.runner import parser, _configuration


def load_plan(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise ValueError("Frozen confirmation plan changed")
    plan = json.loads(path.read_text())
    argv = plan["argv"]
    if not plan.get("frozen") or any(v.split("=", 1)[0] in
        {"--out", "--device", "--checkpoint-dirs", "--help", "-h"} for v in argv):
        raise ValueError("Frozen plan required; wrapper owns output/device/cache")
    command = parser()
    command.allow_abbrev = False
    config = _configuration(command.parse_args(argv))
    if config["protocol"] != "confirmation" or config["seeds"] != list(range(60000, 60020)):
        raise ValueError("Require all twenty reserved confirmation tasks")
    return plan, config, digest


@app.local_entrypoint()
def confirm(run_id: str, protocol_file: str = "configs/confirmation_pairwise_v1.json"):
    plan, config, digest = load_plan(protocol_file)
    dispatch(run_id, plan["argv"], "core", {"phase": "confirmation",
             "frozen_protocol_sha256": digest, "protocol_file": protocol_file,
             "model": " ".join(config["models"])})
