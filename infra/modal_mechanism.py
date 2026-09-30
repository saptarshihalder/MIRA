"""Bounded ephemeral Modal run; no scheduled service or idle GPU."""
from pathlib import Path
import json
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("mira-mechanism")
cache = modal.Volume.from_name("mira-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.3.5", "scipy==1.17.0", "scikit-learn==1.8.0",
                 "tabpfn==9.0.0", "tabicl==2.2.0", "xgboost==3.4.1")
    .env({"HF_HOME": "/cache/huggingface", "TABPFN_MODEL_CACHE_DIR": "/cache/tabpfn",
          "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
          "TABPFN_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "infra", "/opt/pilot")
    .add_local_dir(ROOT / "src", "/opt/mira/src")
    .add_local_dir(ROOT / "scripts", "/opt/mira/scripts")
    .add_local_file(ROOT / "pyproject.toml", "/opt/mira/pyproject.toml")
)

@app.function(image=image, gpu="T4", cpu=(2, 2), memory=(8192, 8192), timeout=900,
              max_containers=1, scaledown_window=2, retries=0, volumes={"/cache": cache})
def run_pilot(argv: list[str], engine: str = "pilot") -> dict:
    import io
    import os
    import subprocess
    import sys
    import zipfile
    out = Path("/tmp/mira_result")
    out.mkdir(exist_ok=True)
    start = time.monotonic()
    os.environ["PYTHONPATH"] = "/opt/mira/src"
    script = "/opt/pilot/tfm_mechanism.py" if engine == "pilot" else "/opt/mira/scripts/run_mechanism.py"
    command = [sys.executable, script, *argv,
               "--device", "cuda", "--checkpoint-dirs", "/cache", "--out", str(out)]
    try:
        result = subprocess.run(command, cwd="/opt/pilot", timeout=780)
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    seconds = time.monotonic() - start
    (out / "modal_runtime.json").write_text(json.dumps({
        "seconds": seconds, "exit_code": exit_code, "command": command,
        "gpu": "T4", "cpu_physical_cores": 2, "memory_mib": 8192,
        "estimated_active_compute_usd": seconds * (0.000164 + 2*0.0000131 + 8*0.00000222),
        "invoice_status": "estimate; excludes startup, image build and provider rounding"
    }, indent=2))
    cache.commit()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    return {"archive": payload.getvalue(), "seconds": seconds, "exit_code": exit_code}

@app.local_entrypoint()
def main(run_id: str = "smoke_tabpfn_v2", model: str = "tabpfn:v2", phase: str = "smoke"):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Use a simple unique run ID")
    if model not in {"tabpfn:v2", "tabpfn:v3.5", "tabicl:v2", "xgboost"}:
        raise ValueError("Model must be explicitly named")
    if phase not in {"smoke", "development", "collision"}:
        raise ValueError("Use smoke, development or collision; confirmation requires frozen design")
    target = ROOT / "artifacts" / "runs" / run_id
    if target.exists():
        raise ValueError("Preserve previous results; choose a new run ID")
    budget = json.loads((ROOT / "configs" / "budget.json").read_text())
    ledger = ROOT / "artifacts" / "manifests" / "compute_ledger.json"
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    reservation = budget["initial_smoke_reservation_usd"] if phase == "smoke" else 0.5
    if sum(x["reserved_usd"] for x in entries) + reservation > budget["total_cap_usd"] - 3:
        raise ValueError("Compute cap reached, preserving $3 reproduction reserve")
    from datetime import datetime, timezone
    entry = {"run_id": run_id, "model": model if phase == "smoke" else "tabpfn:v2 tabicl:v2 xgboost",
             "phase": phase, "reserved_usd": reservation,
             "status": "reserved", "date": datetime.now(timezone.utc).isoformat()}
    entries.append(entry)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(entries, indent=2))
    argv = ["--models", model, "--modes", "native", "native_indicators", "native_shuffled",
            "--gammas", "0", "0.8", "--seeds", "1", "--seed-start", "40000",
            "--context", "256", "--queries", "1024", "--ensembles", "4"]
    if phase != "smoke":
        argv = ["--models", "tabpfn:v2", "tabicl:v2", "xgboost",
                "--modes", "native", "native_indicators", "native_shuffled",
                "--families", "label_only", "--gammas", "0", ".25", ".5", ".75", ".9",
                "--seeds", "3", "--seed-start", "40000", "--protocol", "development",
                "--context", "256", "--queries", "1024", "--ensembles", "4"]
        if phase == "collision":
            argv += ["--value-distribution", "zero_collision"]
    try:
        payload = run_pilot.remote(argv, "pilot" if phase == "smoke" else "core")
        import io
        import zipfile
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload["archive"])) as archive:
            for member in archive.infolist():
                resolved = (target / member.filename).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(target)
        entry.update(status="completed" if payload["exit_code"] == 0 else "failed",
                     active_seconds=payload["seconds"], exit_code=payload["exit_code"],
                     results=str(target))
        print(json.dumps({k:v for k,v in entry.items() if k != "results"}))
        if payload["exit_code"]:
            raise SystemExit(payload["exit_code"])
    except Exception as exc:
        entry.update(status="failed", error=repr(exc))
        raise
    finally:
        ledger.write_text(json.dumps(entries, indent=2))
