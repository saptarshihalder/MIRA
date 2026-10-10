"""Explicit user-authorized continuation after packaging failure; frozen science unchanged.

    modal run --detach -m infra.launch_lifted_v4_l40s_resume --spawn

Original L40S wrapper and failed 265-minute gate are preserved byte-for-byte.
This separate function runs the unchanged main/confirm/pack stages once, with a
240-minute child cap. The pre-existing source estimate already includes 25% safety.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

ORIGINAL_WRAPPER_SHA256 = "f485d9bc45266492d4399c8c7f6650a2fab0c4fe646844f783cf18645177dc0c"
ORIGINAL_GATE_SHA256 = "ee9b0c6d074a0ba0f9433c936533b9cf8fce35e53559cd7e4c1dceeec94ae1e3"
ORIGINAL_RUNTIME_SHA256 = "4c627918fdfa3da632bf3ab2ef40aac8ec0eb9c1ed56a66695512b6f7071d268"
PRETRAINED_WEIGHT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
RUN_ID = "lifted_v4_modal_main_l40s_resume"
CHILD_SECONDS = 14400
FUNCTION_SECONDS = 14700
PROVISION = 9.5


def validate_original(gate, runtime, expected):
    if (gate.get("passed") is not True or gate["identity"]["inputs"] != expected
            or runtime.get("status") != "completed" or runtime.get("exit_code") != 0):
        raise ValueError("Require the unchanged successful L40S source smoke")
    estimate = gate["execution_estimate"]
    if (estimate.get("fits_cap") is not False or estimate.get("child_cap_seconds") != 15900
            or estimate.get("safety_multiplier") != 1.25
            or estimate.get("estimated_seconds") != 20520.96200275374
            or estimate["estimated_seconds"] * 2 / 3 > CHILD_SECONDS):
        raise ValueError("Original failed gate/25%-padded source estimate changed")
    files = gate["pretrained_weights"]["files"]
    if files.get("tabpfn/tabpfn-v2-regressor.ckpt", {}).get("sha256") != PRETRAINED_WEIGHT_SHA256:
        raise ValueError("Original pretrained TabPFN identity changed")
    return estimate


def worker(expected, amendment):
    # Import the qualified helper module on the Linux worker. Caps are explicit
    # here; no locally monkeypatched globals are assumed to propagate remotely.
    from infra import modal_lifted_v4_l40s as original
    import importlib.metadata
    import platform
    import shutil
    import threading
    import torch

    source, panels, data = Path("/opt/mira"), Path("/opt/mira_panels"), Path("/opt/mira_data")
    root = Path("/results/lifted_v4_l40s_20261009")
    original_file = source / "infra/modal_lifted_v4_l40s.py"
    extension_file = source / "infra/launch_lifted_v4_l40s_resume.py"
    started = time.monotonic()
    volume = modal.Volume.from_name("mira-lifted-v4-l40s")
    volume.reload()
    if (original.digest(original_file) != ORIGINAL_WRAPPER_SHA256
            or original.digest(extension_file) != amendment["extension_sha256"]):
        raise ValueError("Original wrapper or explicit extension changed")
    prior_runtime = root / "modal_main_extended_runtime.json"
    prior_log = root / "logs/modal_main_extended.log"
    prior = json.loads(prior_runtime.read_text())
    if (prior.get("status") != "failed" or prior.get("exit_code") != 1
            or prior.get("timed_out", False) or prior.get("final_commit_error")
            or original.digest(prior_runtime) != amendment["previous_runtime_sha256"]
            or original.digest(prior_log) != amendment["previous_log_sha256"]
            or "FileNotFoundError: [Errno 2] No such file or directory: 'git'" not in prior_log.read_text()):
        raise ValueError("Only the recorded missing-git failure permits this manual continuation")
    protected = amendment["preserved_seed1_files"]
    if sum(Path(x).name == "model.pt" for x in protected) != 8 or sum(x.endswith(".npz") for x in protected) != 15:
        raise ValueError("Require all eight seed1 learned checkpoints and15 score archives")
    if any(original.digest(root / name) != sha for name, sha in protected.items()):
        raise ValueError("Preserved seed1 files differ before continuation")
    git_version = subprocess.run(["git", "--version"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    actual = original.frozen_inputs(source, panels, data, original_file)
    if actual != expected or amendment["original_inputs"] != expected:
        raise ValueError("Frozen scientific inputs/packages/profiler changed")
    if (amendment["child_seconds"] != 14400 or amendment["function_seconds"] != 14700
            or amendment["startup_seconds"] != 120 or amendment["run_id"] != RUN_ID
            or amendment.get("scientific_changes") is not False):
        raise ValueError("Only the explicit 240-minute engineering amendment is accepted")
    gate_path, runtime_path = root / "modal_smoke_gate.json", root / "modal_smoke_runtime.json"
    if (original.digest(gate_path) != ORIGINAL_GATE_SHA256
            or original.digest(runtime_path) != ORIGINAL_RUNTIME_SHA256
            or amendment["original_gate_sha256"] != ORIGINAL_GATE_SHA256
            or amendment["original_runtime_sha256"] != ORIGINAL_RUNTIME_SHA256):
        raise ValueError("Saved original smoke gate/runtime changed")
    gate, smoke_runtime = json.loads(gate_path.read_text()), json.loads(runtime_path.read_text())
    estimate = validate_original(gate, smoke_runtime, expected)
    if original.estimate_main(root) != estimate:
        raise ValueError("Source-only profile/estimate changed")
    if original.digest(root / "smoke/report.json") != gate["smoke_report_sha256"]:
        raise ValueError("Original source smoke report changed")
    weights = original.weight_identity(root)
    if weights != gate["pretrained_weights"]:
        raise ValueError("Actual pretrained weight bytes changed")
    packages = {p: importlib.metadata.version(p) for p in original.PACKAGES}
    if packages != original.PACKAGES or not torch.cuda.is_available():
        raise ValueError("Pinned packages and CUDA required")
    device = torch.cuda.get_device_name(0)
    if "L40S" not in device or not 44 <= torch.cuda.get_device_properties(0).total_memory / (1 << 30) <= 50:
        raise ValueError("Require the same L40S hardware class")
    bound_identity = dict(schema=1, source=actual["source"], panels=actual["panels"],
                          panel_manifest=actual["panel_manifest"],
                          packages={p: packages[p] for p in ("torch", "numpy", "scipy", "tabpfn")},
                          device=device, cuda=torch.version.cuda)
    identity = dict(inputs=actual, bounded_identity=bound_identity)
    if (identity != gate["identity"] or json.loads((root / "modal_identity.json").read_text()) != identity
            or json.loads((root / "run_identity.json").read_text())["identity"] != bound_identity):
        raise ValueError("Original GPU/runtime/scientific identity changed")
    lock = root / "MODAL_RUNNING.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    process, thread = None, None
    stopping, durability_failed = threading.Event(), threading.Event()
    commits = []
    status = dict(stage="main_resume", root=str(root), status="failed", exit_code=1,
                  requested_gpu="L40S", cpu=2, memory_mib=16384, child_minutes=240,
                  function_timeout_seconds=14700, startup_timeout_seconds=120, automatic_retries=0,
                  packages=packages, cuda=torch.version.cuda, device=device, python=platform.python_version(),
                  original_gate_sha256=ORIGINAL_GATE_SHA256, extension=amendment,
                  estimated_seconds=estimate["estimated_seconds"])
    result_path = root / "modal_main_resume_runtime.json"
    attempt = root / "modal_main_resume_attempt.json"
    def periodic_commit():
        while not stopping.wait(60):
            try:
                volume.commit()
                commits.append(dict(seconds=time.monotonic()-started, ok=True))
            except Exception as error:
                commits.append(dict(seconds=time.monotonic()-started, ok=False, error=str(error)))
                durability_failed.set(); return
    try:
        if attempt.exists() or (root / "modal_main_attempt.json").exists() or (root / "RUNNING.lock").exists():
            raise ValueError("Main was already attempted or a process remains; no automatic retries")
        original.write_json(attempt, dict(identity=identity, amendment=amendment, started_unix=time.time()))
        original.write_json(root / "main_resume_execution_amendment.json", amendment)
        # The original gate remains false at its original cap; only this sidecar
        # records acceptance under the new execution-time allowance.
        original.write_json(root / "main_resume_execution_gate.json",
                            dict(original_gate_sha256=ORIGINAL_GATE_SHA256, original_gate=gate,
                                 amendment=amendment, resume_fits_cap=True))
        shutil.copytree(data / "beijing", root / "data/beijing", dirs_exist_ok=True)
        code = source / "experiments/lifted_cavity"
        command = [sys.executable, "-u", str(code / "bounded_v4.py"), "--root", str(root),
                   "--panels", str(panels), "--minutes", "240", "main", "confirm", "pack"]
        env = dict(os.environ, HF_HOME=str(root / "cache/huggingface"),
                   TABPFN_MODEL_CACHE_DIR=str(root / "cache/tabpfn"), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TABPFN_DISABLE_TELEMETRY="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        remaining = 14700 - (time.monotonic()-started) - 45
        if remaining < 14400:
            raise ValueError("Preparation consumed the reserved child time budget")
        # Exercise the unchanged pack function on an empty temporary root before training.
        # Archive source has no .git: its manifest git field may be empty. Frozen
        # run_identity.json/MIRA_SOURCE_COMMIT.txt supplies the separately verified identity.
        preflight = "import tempfile; from pathlib import Path; import colab_v4 as d; t=tempfile.TemporaryDirectory(); d.ROOT=Path(t.name); d.RUNS=d.ROOT/'runs'; d.REAL=d.ROOT/'runs_real'; d.LOGS=d.ROOT/'logs'; d.DEV='cpu'; d.stage_pack(); assert (d.ROOT/'mira_v4_results.zip').is_file()"
        smoke_pack = subprocess.run([sys.executable, "-c", preflight], cwd=code, env=env,
                                   capture_output=True, text=True, timeout=90, check=True)
        status.update(git_version=git_version, packaging_preflight_passed=True,
                      packaging_source_identity="Frozen source archive identity; Git HEAD may be unavailable")
        status["command"] = command
        original.write_json(result_path, dict(status, status="running"))
        volume.commit()
        thread = threading.Thread(target=periodic_commit, daemon=True); thread.start()
        with (root / "logs/modal_main_resume.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, cwd=code, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+remaining
            while process.poll() is None:
                if durability_failed.is_set() or time.monotonic() >= deadline:
                    original.stop_tree(process); status["timed_out"] = not durability_failed.is_set(); break
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        status["exit_code"] = process.returncode
        if process.returncode != 0 or durability_failed.is_set() or status.get("timed_out"):
            raise RuntimeError("Extended bounded main failed, timed out, or lost periodic durability")
        confirmation = root / "runs/confirm_v4.json"
        if not confirmation.is_file():
            raise ValueError("Main did not produce complete three-seed confirmation")
        if original.weight_identity(root) != weights or original.digest(gate_path) != ORIGINAL_GATE_SHA256:
            raise ValueError("Pretrained weights or original source gate changed during main")
        if any(original.digest(root / name) != sha for name, sha in protected.items()):
            raise ValueError("Seed1 files changed during continuation")
        status["preserved_seed1_hashes_unchanged"] = True
        status.update(status="completed", confirmation_sha256=original.digest(confirmation))
    except Exception as error:
        status["error"] = str(error)
    finally:
        if process is not None and process.poll() is None:
            original.stop_tree(process)
        stopping.set()
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                status.update(status="failed", error="Periodic commit did not finish")
        status.update(seconds=time.monotonic()-started, periodic_commits=commits)
        original.write_json(result_path, status); lock.unlink(missing_ok=True)
        try:
            volume.commit()
        except Exception as error:
            status.update(status="failed", final_commit_error=str(error))
    return {k: status[k] for k in ("stage", "root", "status", "exit_code", "seconds", "error",
                                   "estimated_seconds", "final_commit_error") if k in status}


if modal.is_local():
    from infra import modal_lifted_v4_l40s as original
    repo = Path(__file__).resolve().parents[1]
    reports = repo / "artifacts/reports/lifted_v4_launch"
    if original.digest(repo / "infra/modal_lifted_v4_l40s.py") != ORIGINAL_WRAPPER_SHA256:
        raise ValueError("Preserve original L40S wrapper bytes")
    gate_file, runtime_file = reports / "l40s_smoke_gate.json", reports / "l40s_smoke_runtime.json"
    if original.digest(gate_file) != ORIGINAL_GATE_SHA256 or original.digest(runtime_file) != ORIGINAL_RUNTIME_SHA256:
        raise ValueError("Original downloaded gate/runtime changed")
    gate = json.loads(gate_file.read_text()); runtime = json.loads(runtime_file.read_text())
    expected = original.expected
    estimate = validate_original(gate, runtime, expected)
    recovery = json.loads((repo / "artifacts/manifests/lifted_v4_failed_seed1_recovery.json").read_text())
    preserved = {name: row["sha256"] for name, row in recovery["files"].items()
                 if name.startswith(("runs/", "runs_real/")) or name in ("run_identity.json", "cache/tabpfn/tabpfn-v2-regressor.ckpt")}
    if sum(Path(x).name == "model.pt" for x in preserved) != 8 or sum(x.endswith(".npz") for x in preserved) != 15:
        raise ValueError("Incomplete local seed1 recovery inventory")
    amendment = dict(schema=1, run_id=RUN_ID, amendment="Manual continuation after missing-git packaging failure; no scientific changes; retain seed1 exactly",
                     scientific_changes=False, extension_sha256=original.digest(Path(__file__)),
                     original_wrapper_sha256=ORIGINAL_WRAPPER_SHA256, original_gate_sha256=ORIGINAL_GATE_SHA256,
                     original_runtime_sha256=ORIGINAL_RUNTIME_SHA256, original_inputs=expected,
                     original_estimated_seconds=estimate["estimated_seconds"], original_fits_cap=False,
                     original_child_seconds=15900, child_seconds=14400, function_seconds=14700,
                     startup_seconds=120, provision_usd=9.5, automatic_retries=0)
    amendment.update(previous_runtime_sha256=original.digest(reports / "l40s_main_extended_final_runtime.json"),
                     previous_log_sha256=original.digest(reports / "l40s_main_extended_failure.log"),
                     preserved_seed1_files=preserved, manual_continuation=True,
                     authorization="User: we are one day late, we need to complete it, check (2026-10-11 IST)",
                     remaining_estimate_seconds=estimate["estimated_seconds"] * 2 / 3)
    manifest = repo / "artifacts/manifests/lifted_v4_l40s_resume_amendment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != amendment:
        ledger = json.loads((repo / "artifacts/manifests/compute_ledger.json").read_text())
        if any(r.get("run_id") == RUN_ID and r.get("status") != "reserved_not_launched" for r in ledger):
            raise ValueError("Preserve amendment after first submission")
        original.write_json(manifest, amendment)
    elif not manifest.exists():
        original.write_json(manifest, amendment)
    app = modal.App("mira-lifted-v4-l40s-resume")
    volume = modal.Volume.from_name("mira-lifted-v4-l40s")
    # Build Git into the cached dependency image before attaching local mounts.
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install(*(f"{package}=={version}" for package, version in original.PACKAGES.items()))
             .apt_install("git")
             .env({"PYTHONPATH": "/opt/mira", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "PYTHONUNBUFFERED": "1", "TABPFN_DISABLE_TELEMETRY": "1"})
             .add_local_dir(original.source, "/opt/mira")
             .add_local_file(repo / "infra/modal_lifted_v4_l40s.py", "/opt/mira/infra/modal_lifted_v4_l40s.py")
             .add_local_file(repo / "infra/profile_lifted_v4_source.py", "/opt/mira/infra/profile_lifted_v4_source.py")
             .add_local_dir(original.panels, "/opt/mira_panels")
             .add_local_dir(original.data, "/opt/mira_data")
             .add_local_file(Path(__file__), "/opt/mira/infra/launch_lifted_v4_l40s_resume.py"))
    run_main = app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(16384, 16384),
                            timeout=14700, startup_timeout=120, min_containers=0, max_containers=1,
                            scaledown_window=2, retries=0, serialized=True, volumes={"/results": volume})(worker)

    @app.local_entrypoint()
    def launch(spawn: bool = True):
        ledger_path = repo / "artifacts/manifests/compute_ledger.json"
        attempt = repo / "artifacts/manifests" / (RUN_ID + "_submission.json")
        with original.ledger_lock(ledger_path):
            entries = json.loads(ledger_path.read_text())
            matches = [r for r in entries if r.get("run_id") == RUN_ID]
            if (len(matches) != 1 or matches[0].get("status") != "reserved_not_launched"
                    or matches[0].get("function_call_id") or attempt.exists()
                    or float(matches[0]["reserved_usd"]) != 9.5):
                raise ValueError("Require one unused $9.5 resume-main reservation")
            cap = float(json.loads((repo / "configs/budget.json").read_text())["total_cap_usd"])
            reservations = [float(r.get("reserved_usd", 0)) for r in entries]
            if (not math.isfinite(cap) or not 0 < cap <= 20
                    or any(not math.isfinite(x) or x < 0 for x in reservations)
                    or sum(reservations)+3 > cap):
                raise ValueError("Budget cap/reproduction reserve would be exceeded")
            rates = json.loads((repo / "artifacts/manifests/modal_rates_20261011.json").read_text())
            hourly = float(rates["gpu_hour_cost_l40s"])+2*float(rates["cpu_hour_cost"])+16*float(rates["mem_gib_hour_cost"])
            cost = (14700+120)*hourly/3600
            if not math.isfinite(cost) or not 0 < cost < 9.5:
                raise ValueError("Current bounded compute/startup cost exceeds provision")
            entry = matches[0]
            entry.update(status="launch_requested", launch_requested_unix=time.time(),
                         amendment_sha256=original.digest(manifest), bounded_compute_estimate_usd=cost,
                         remaining_build_storage_egress_provision_usd=9.5-cost,
                         automatic_retries=0, function_timeout_seconds=14700, startup_timeout_seconds=120)
            original.write_json(ledger_path, entries); original.write_json(attempt, entry)
        try:
            call = run_main.spawn(expected, amendment)
            original.record_submission(ledger_path, attempt, RUN_ID, dict(status="running", function_call_id=call.object_id))
        except Exception as error:
            original.record_submission(ledger_path, attempt, RUN_ID, dict(status="failed_submission", error=str(error))); raise
        print(json.dumps(dict(run_id=RUN_ID, function_call_id=call.object_id,
                              amendment_sha256=original.digest(manifest), root=original.RESULTS.as_posix())))
        if not spawn:
            result = call.get()
            original.record_submission(ledger_path, attempt, RUN_ID, dict(status=result["status"], result=result))
            print(json.dumps(result))
