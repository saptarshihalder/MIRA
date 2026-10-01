"""Bounded ephemeral Modal run; no scheduled service or idle GPU."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
import sys
import tempfile
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
CALL_RESERVATION_USD = 0.5
REPRODUCTION_RESERVE_USD = 3.0
MODELS = {"tabpfn:v2", "tabpfn:v3.5", "tabicl:v2", "xgboost"}
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
    if engine not in {"pilot", "core"}:
        raise ValueError("Unknown experiment engine")
    # A warm Modal container may execute multiple calls; never reuse previous data.
    out = Path(tempfile.mkdtemp(prefix="mira_result_"))
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
    try:
        cache.commit()
    except Exception as error:
        # Keep partial artifacts even when committing the checkpoint cache fails.
        (out / "cache_commit_error.json").write_text(json.dumps({"error": repr(error)}))
        exit_code = exit_code or 1
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    return {"archive": payload.getvalue(), "seconds": seconds, "exit_code": exit_code}

def validate_run_id(run_id: str) -> None:
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Use a simple unique run ID")


def validate_core_argv(argv: list[str]) -> dict:
    """Validate through the current core CLI while preserving the exact argument list."""
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv):
        raise ValueError("argv must be a nonempty JSON list of strings")
    for token in argv:
        if token.split("=", 1)[0] in {"--out", "--device", "--checkpoint-dirs", "--help", "-h"}:
            raise ValueError("Wrapper owns output, CUDA device and exact checkpoint-cache paths")
    sys.path.insert(0, str(ROOT / "src"))
    from mira.runner import parser, _configuration
    command = parser()
    command.allow_abbrev = False
    try:
        args = command.parse_args(argv)
    except SystemExit as error:
        raise ValueError("Invalid core CLI arguments") from error
    configuration = _configuration(args)
    if configuration["protocol"] != "development":
        raise ValueError("Confirmation is unavailable through this wrapper")
    seeds = configuration["seeds"]
    if not seeds or any(not 40000 <= seed < 50000 for seed in seeds):
        raise ValueError("Only development seeds 40000–49999 are permitted")
    if any(model not in MODELS for model in configuration["models"]):
        raise ValueError("Model must be explicitly supported and versioned")
    return configuration


def load_matrix(path: Path, selected: str = "") -> list[dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Matrix requires a nonempty entries list")
    names = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every matrix entry must be an object")
        validate_run_id(entry.get("name", ""))
        names.append(entry["name"])
    if len(set(names)) != len(names):
        raise ValueError("Matrix entry names must be unique")
    if selected and selected not in names:
        raise ValueError("Requested matrix entry does not exist")
    chosen = [entry for entry in entries if not selected or entry["name"] == selected]
    for entry in chosen:
        validate_core_argv(entry.get("argv"))
    return chosen


@contextmanager
def ledger_lock(ledger: Path, timeout: float = 10):
    """OS releases this lock on a process crash; persisted reservations remain charged."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Compute ledger is locked by another process")
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_ledger_write(ledger: Path, entries: list[dict]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="compute_ledger_", suffix=".tmp", dir=ledger.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ledger)
    finally:
        Path(temporary).unlink(missing_ok=True)


def reserve_call(ledger: Path, budget: dict, run_id: str, metadata: dict) -> dict:
    validate_run_id(run_id)
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
        if any(entry["run_id"] == run_id for entry in entries):
            raise ValueError("Run ID already reserved; preserve crash/failed calls and use a new ID")
        reservations = [float(entry["reserved_usd"]) for entry in entries]
        cap = float(budget["total_cap_usd"])
        if not math.isfinite(cap) or any(not math.isfinite(value) or value < 0 for value in reservations):
            raise ValueError("Invalid compute budget or ledger")
        if sum(reservations) + CALL_RESERVATION_USD > cap - REPRODUCTION_RESERVE_USD + 1e-9:
            raise ValueError("Compute cap reached, preserving $3 reproduction reserve")
        entry = {**metadata, "run_id": run_id, "reserved_usd": CALL_RESERVATION_USD,
                 "status": "reserved", "date": datetime.now(timezone.utc).isoformat()}
        entries.append(entry)
        atomic_ledger_write(ledger, entries)
        return entry


def finish_call(ledger: Path, run_id: str, updates: dict) -> dict:
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        entry = next(entry for entry in entries if entry["run_id"] == run_id)
        entry.update(updates)
        atomic_ledger_write(ledger, entries)
        return entry


def dispatch(run_id: str, argv: list[str], engine: str, metadata: dict) -> None:
    validate_run_id(run_id)
    target = ROOT / "artifacts" / "runs" / run_id
    if target.exists():
        raise ValueError("Preserve previous results; choose a new run ID")
    budget = json.loads((ROOT / "configs" / "budget.json").read_text())
    ledger = ROOT / "artifacts" / "manifests" / "compute_ledger.json"
    reserve_call(ledger, budget, run_id, {**metadata, "argv": argv, "engine": engine, "max_seconds": 900})
    updates = {"status": "failed"}
    try:
        payload = run_pilot.remote(argv, engine)
        import io
        import zipfile
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload["archive"])) as archive:
            for member in archive.infolist():
                resolved = (target / member.filename).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(target)
        updates.update(status="completed" if payload["exit_code"] == 0 else "failed",
                       active_seconds=payload["seconds"], exit_code=payload["exit_code"], results=str(target))
        print(json.dumps({"run_id": run_id, **{k: v for k, v in updates.items() if k != "results"}}))
        if payload["exit_code"]:
            raise SystemExit(payload["exit_code"])
    except BaseException as error:
        updates["error"] = repr(error)
        raise
    finally:
        # Reload under the lock: another process may have reserved a call meanwhile.
        finish_call(ledger, run_id, updates)


@app.local_entrypoint()
def main(run_id: str = "smoke_tabpfn_v2", model: str = "tabpfn:v2", phase: str = "",
         matrix: str = "", entry: str = ""):
    validate_run_id(run_id)
    if matrix:
        if phase:
            raise ValueError("--matrix and --phase are mutually exclusive")
        for selected in load_matrix(Path(matrix), entry):
            configuration = validate_core_argv(selected["argv"])
            dispatch(f"{run_id}_{selected['name']}", selected["argv"], "core",
                     {"phase": "matrix_development", "matrix": str(Path(matrix).resolve()),
                      "matrix_entry": selected["name"], "model": " ".join(configuration["models"])})
        return
    if entry:
        raise ValueError("--entry requires --matrix")
    phase = phase or "smoke"
    if model not in MODELS:
        raise ValueError("Model must be explicitly named")
    if phase not in {"smoke", "development", "collision"}:
        raise ValueError("Use smoke, development or collision; confirmation requires frozen design")
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
    if phase != "smoke":
        validate_core_argv(argv)
    dispatch(run_id, argv, "pilot" if phase == "smoke" else "core",
             {"phase": phase, "model": model if phase == "smoke" else "tabpfn:v2 tabicl:v2 xgboost"})
