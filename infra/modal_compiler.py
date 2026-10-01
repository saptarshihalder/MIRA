"""Two hash-frozen compiler calls maximum; no retry, background service, or core edits."""
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/mask_compiler"))
from infra.modal_mechanism import cache, image, ledger_lock, atomic_ledger_write, finish_call, validate_run_id
from run_backbone import load_protocol, natural_episodes, MODELS

app = modal.App("mira-mask-compiler")
compiler_image = image.add_local_dir(ROOT / "experiments/mask_compiler", "/opt/mira/experiments/mask_compiler")
compiler_image = compiler_image.add_local_dir(ROOT / "infra", "/opt/mira/infra")
compiler_image = compiler_image.add_local_dir(ROOT / "configs", "/opt/mira/configs")
for directory in ("artifacts/reports/trained_compiler_v1", "artifacts/reports/matched_compiler_v1", "artifacts/data/natural_compiler"):
    if (ROOT / directory).exists():
        compiler_image = compiler_image.add_local_dir(ROOT / directory, "/opt/mira/" + directory)


@app.function(image=compiler_image, gpu="T4", cpu=(2, 2), memory=(8192, 8192), timeout=900,
              max_containers=1, scaledown_window=2, retries=0, volumes={"/cache": cache})
def run_compiler(protocol_relative: str, model: str):
    out = Path(tempfile.mkdtemp(prefix="mira_compiler_"))
    started = time.monotonic()
    command = [sys.executable, "/opt/mira/experiments/mask_compiler/run_backbone.py", "--protocol",
               "/opt/mira/"+protocol_relative, "--model", model, "--cache-root", "/cache", "--out", str(out)]
    try:
        result = subprocess.run(command, cwd="/opt/mira", timeout=780)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
    except Exception as error:
        code = 1
        (out/"wrapper_error.json").write_text(json.dumps({"error": repr(error)}))
    seconds = time.monotonic()-started
    try:
        cache.commit()
    except Exception as error:
        code = code or 1
        (out/"cache_commit_error.json").write_text(json.dumps({"error": repr(error)}))
    (out/"modal_runtime.json").write_text(json.dumps({"seconds": seconds, "exit_code": code, "command": command,
        "gpu": "T4", "max_seconds": 900, "estimated_active_compute_usd": seconds*(.000164+2*.0000131+8*.00000222),
        "invoice_status": "estimate; excludes startup/build/provider rounding"}, indent=2))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    return {"archive": buffer.getvalue(), "seconds": seconds, "exit_code": code}


def reserve_compiler(ledger, budget, run_id, model, digest, version):
    validate_run_id(run_id)
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        old = [entry for entry in entries if entry.get("phase") == "mask_compiler_backbone"]
        if len(old) >= 2 or sum(float(e["reserved_usd"]) for e in old)+.5 > 1+1e-9:
            raise ValueError("Compiler budget exhausted: two calls/$1 including crashes and failures")
        if any(e["run_id"] == run_id for e in entries) or any(e["model"] == model for e in old):
            raise ValueError("Run ID/backbone already reserved; no automatic retry")
        if any(e["protocol_sha256"] != digest for e in old):
            raise ValueError("Both backbones must use the same frozen protocol")
        values = [float(e["reserved_usd"]) for e in entries]
        cap = float(budget["total_cap_usd"])
        if not math.isfinite(cap) or any(not math.isfinite(v) or v < 0 for v in values) or sum(values)+.5 > cap-3+1e-9:
            raise ValueError("Global compute cap reached or invalid; preserve $3 reserve")
        entries.append({"run_id": run_id, "phase": "mask_compiler_backbone", "model": model,
                        "reserved_usd": .5, "status": "reserved", "max_seconds": 900,
                        "protocol_sha256": digest, "protocol_version": version, "date": datetime.now(timezone.utc).isoformat()})
        atomic_ledger_write(ledger, entries)


@app.local_entrypoint()
def main(run_id: str, model: str, protocol_file: str = "configs/compiler_backbone_v1.json"):
    validate_run_id(run_id)
    if model not in MODELS:
        raise ValueError("Use the explicit TabPFN v2 or TabICL v2 backbone")
    protocol = Path(protocol_file).resolve()
    if not protocol.is_relative_to(ROOT.resolve()):
        raise ValueError("Protocol must be inside the uploaded repository")
    plan, digest = load_protocol(protocol, ROOT)
    list(natural_episodes(plan, ROOT))  # Validate sizes/groups/ranking before reserving paid compute.
    target = ROOT/"artifacts/runs"/run_id
    if target.exists():
        raise ValueError("Preserve existing results; choose a new run ID")
    ledger = ROOT/"artifacts/manifests/compute_ledger.json"
    budget = json.loads((ROOT/"configs/budget.json").read_text())
    reserve_compiler(ledger, budget, run_id, model, digest, plan["protocol_version"])
    updates = {"status": "failed"}
    try:
        payload = run_compiler.remote(str(protocol.relative_to(ROOT)).replace("\\", "/"), model)
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload["archive"])) as archive:
            if any(not (target/member.filename).resolve().is_relative_to(target.resolve()) for member in archive.infolist()):
                raise ValueError("Unsafe returned archive path")
            archive.extractall(target)
        updates.update(status="completed" if payload["exit_code"] == 0 else "failed",
                       active_seconds=payload["seconds"], exit_code=payload["exit_code"], results=str(target))
        print(json.dumps({"run_id": run_id, "model": model, **updates}))
        if payload["exit_code"]:
            raise SystemExit(payload["exit_code"])
    except BaseException as error:
        updates["error"] = repr(error)
        raise
    finally:
        finish_call(ledger, run_id, updates)
