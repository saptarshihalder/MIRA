"""Prepare/launch the frozen v4 stages; importing this file never launches a job.

After local inputs and the compute ledger are prepared:
    modal run --detach -m infra.modal_lifted_v4_l40s --stage smoke --spawn
    modal run --detach -m infra.modal_lifted_v4_l40s --stage main --spawn

--detach keeps a spawned invocation alive after the launcher exits. There are no
automatic retries. Results stay on mira-lifted-v4-l40s; return values contain no
archive or checkpoint bytes. Local input paths may be set through MIRA_V4_SOURCE_DIR,
MIRA_V4_PANELS_DIR, and MIRA_V4_DATA_DIR. The remote result root is fixed.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

import modal

SOURCE_COMMIT = "e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9"
PANEL_COMMIT = "799f473459c99e8decbb39ff9924c86a90cc8dc7"
REMOTE_SOURCE = PurePosixPath("/opt/mira")
REMOTE_PANELS = PurePosixPath("/opt/mira_panels")
REMOTE_DATA = PurePosixPath("/opt/mira_data")
RESULTS = PurePosixPath("/results/lifted_v4_l40s_20261009")
PACKAGES = {"torch": "2.8.0", "numpy": "2.3.5", "scipy": "1.17.0",
            "pandas": "2.3.3", "tabpfn": "9.1.0", "scikit-learn": "1.8.0",
            "psutil": "7.0.0"}
CHILD_MINUTES = {"smoke": 18, "main": 265}
FUNCTION_SECONDS = {"smoke": 1200, "main": 16200}
RESERVATIONS = {"smoke": ("lifted_v4_modal_source_smoke_l40s", 1.),
                "main": ("lifted_v4_modal_main_l40s", 11.5)}


@contextmanager
def ledger_lock(path):
    with path.with_suffix(".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        deadline = time.monotonic() + 10
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Compute ledger is locked")
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def authorize_submission(repo, stage, expected):
    """Atomically consume an existing provision before exactly one spawn."""
    run_id, provision = RESERVATIONS[stage]
    ledger = repo / "artifacts/manifests/compute_ledger.json"
    attempt = repo / "artifacts/manifests" / (run_id + "_submission.json")
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text())
        matching = [entry for entry in entries if entry.get("run_id") == run_id]
        if len(matching) != 1 or matching[0].get("status") != "reserved_not_launched" or attempt.exists():
            raise ValueError("Stage needs a unique unused reserved_not_launched provision")
        entry = matching[0]
        if float(entry["reserved_usd"]) != provision or entry.get("function_call_id"):
            raise ValueError("Stage reservation changed or was already submitted")
        reservations = [float(row.get("reserved_usd", 0)) for row in entries]
        cap = float(json.loads((repo / "configs/budget.json").read_text())["total_cap_usd"])
        if (not math.isfinite(cap) or cap > 20 or cap <= 0
                or any(not math.isfinite(x) or x < 0 for x in reservations)
                or sum(reservations) + 3 > cap):
            raise ValueError("Reservations exceed the $20 cap or consume the $3 reproduction reserve")
        rates = json.loads((repo / "artifacts/manifests/modal_rates_20261009.json").read_text())
        hourly = (float(rates["gpu_hour_cost_l40s"]) + 2 * float(rates["cpu_hour_cost"])
                  + 16 * float(rates["mem_gib_hour_cost"]))
        bounded_compute = (FUNCTION_SECONDS[stage] + 120) * hourly / 3600
        if not math.isfinite(hourly) or hourly <= 0 or bounded_compute >= provision:
            raise ValueError("Current rates exceed the stage provision, including bounded startup")
        entry.update(status="launch_requested", launch_requested_unix=time.time(),
                     input_identity_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                     bounded_compute_estimate_usd=bounded_compute,
                     remaining_build_storage_egress_provision_usd=provision - bounded_compute,
                     function_timeout_seconds=FUNCTION_SECONDS[stage], startup_timeout_seconds=120,
                     automatic_retries=0, root=RESULTS.as_posix())
        write_json(ledger, entries)
        write_json(attempt, entry)
    return ledger, attempt, run_id


def record_submission(ledger, attempt, run_id, updates):
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text())
        entry = next(row for row in entries if row.get("run_id") == run_id)
        entry.update(updates)
        write_json(ledger, entries)
        write_json(attempt, entry)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def frozen_inputs(source, panels, data, wrapper):
    """Read bytes/hashes only: do not deserialize evaluation panels."""
    if (source / "MIRA_SOURCE_COMMIT.txt").read_text().strip() != SOURCE_COMMIT:
        raise ValueError("Require the frozen e1b1d9d source archive")
    if (panels / "MIRA_PANEL_COMMIT.txt").read_text().strip() != PANEL_COMMIT:
        raise ValueError("Require the frozen 799f473 panel archive")
    code = source / "experiments/lifted_cavity"
    source_files = sorted(code.glob("*.py")) + [source / "docs/LIFTED_CAVITY_PROTOCOL_V4.md"]
    for name in ("bounded_v4.py", "colab_v4.py", "tabpfn_colab.py", "score_validation.py"):
        if not (code / name).is_file():
            raise ValueError("Missing frozen source: " + name)
    sums = {}
    for line in (panels / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            checksum, name = line.split()
            name = name.lstrip("*")
            if Path(name).name != name or name in sums or not name.endswith(".pt"):
                raise ValueError("Invalid panel manifest entry")
            sums[name] = checksum
    if len(sums) != 19 or set(sums) != {p.name for p in panels.glob("*.pt")}:
        raise ValueError("Require exactly the nineteen frozen panels")
    for name, checksum in sums.items():
        if digest(panels / name) != checksum:
            raise ValueError("Frozen panel changed: " + name)
    tree = ast.parse((code / "colab_v4.py").read_text(encoding="utf-8"))
    station_hashes = next(ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "BEIJING" for t in node.targets))
    raw = {}
    for station, checksum in station_hashes.items():
        relative = f"beijing/PRSA_Data_{station}_20130301-20170228.csv"
        if digest(data / relative) != checksum:
            raise ValueError("Raw Beijing input changed: " + relative)
        raw[relative] = checksum
    return dict(schema=1, source_commit=SOURCE_COMMIT, panel_commit=PANEL_COMMIT,
                source={str(p.relative_to(source)).replace("\\", "/"): digest(p) for p in source_files},
                panels=dict(sorted(sums.items())), panel_manifest=digest(panels / "SHA256SUMS"),
                raw_data=raw, packages=PACKAGES, wrapper_sha256=digest(wrapper),
                source_profiler_sha256=digest(wrapper.parent / "profile_lifted_v4_source.py"))


def weight_identity(root):
    """Hash the actual downloaded v2 regression checkpoint, not its version name."""
    cache = root / "cache"
    weights = {}
    for folder in (cache / "tabpfn", cache / "huggingface"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}:
                weights[str(path.relative_to(cache))] = dict(sha256=digest(path), bytes=path.stat().st_size)
    actual = [p for p in weights if Path(p).name.lower().startswith("tabpfn-v2-regressor")]
    if not actual or any(weights[p]["bytes"] < 1_000_000 for p in actual):
        raise ValueError("Actual TabPFN v2 regression weights were not located in the durable cache")
    return dict(package="9.1.0", version="v2", n_estimators=8, random_state=0, files=weights)


def estimate_main(root):
    """Freeze a source-only estimate for main+confirm+pack, without historical panels."""
    report = json.loads((root / "smoke/report.json").read_text())
    if report.get("source_only") is not True:
        raise ValueError("Smoke report must be source-only")
    profile = json.loads((root / "source_profile.json").read_text())
    if profile.get("source_only") is not True or profile.get("tasks") != 128:
        raise ValueError("Require the independent 128-task source-only scoring profile")
    for model in ("pfn_L", "lct_L"):
        for kind in ("synthetic_seconds", "real_seconds"):
            seconds = float(profile["models"][model][kind])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Invalid source-only scoring profile walltime")
    rates, setup = {}, []
    for model in ("pfn_L", "lct_L"):
        ft = json.loads((root / "smoke" / (model + "_ft") / "train.json").read_text())["finetune"]
        if ft["steps"] != 20:
            raise ValueError("Require the unchanged twenty-step fine-tuning smoke")
        rates[model] = float(ft["seconds"]) / 20
        setup.append(max(0., float(report[model + "_ft_pool_and_20_steps_sec"]) - float(ft["seconds"])))
    timing = json.loads((root / "smoke/tabpfn/train.json").read_text())["timing"]
    syn = timing["smoke_source_synthetic"]
    real = timing["smoke_source_beijing"]
    if syn["calls"] != 16 or real["calls"] != 4:
        raise ValueError("Unexpected source TabPFN smoke call counts")
    components = dict(
        training=3 * 40000 * sum(float(report[m + "_sec_per_step"]) for m in rates),
        fine_tuning=3 * 3 * 2000 * sum(rates.values()),
        source_pool_and_startup=3 * max(setup),
        learned_synthetic_scoring=3 * (256 + 128) / 128 * sum(float(profile["models"][m]["synthetic_seconds"]) for m in rates),
        learned_real_scoring=3 * (719 + 719 + 713) / 128 * sum(float(profile["models"][m]["real_seconds"]) for m in rates),
        tabpfn_synthetic=1536 * float(syn["seconds"]) / syn["calls"],
        tabpfn_real=2151 * float(real["seconds"]) / real["calls"],
        fixed_confirmation_pack_and_baseline_overhead=300.)
    if any(not math.isfinite(x) or x < 0 for x in components.values()):
        raise ValueError("Non-finite or negative throughput estimate")
    if any(not math.isfinite(x) or x <= 0 for x in rates.values()):
        raise ValueError("Invalid fine-tuning throughput")
    seconds = 1.25 * sum(components.values())
    return dict(components_seconds=components, fine_tuning_seconds_per_step=rates,
                safety_multiplier=1.25, estimated_seconds=seconds, child_cap_seconds=15900,
                fits_cap=seconds <= 15900, synthetic_tabpfn_calls=1536, real_tabpfn_calls=2151,
                source_profile_sha256=digest(root / "source_profile.json"), scoring_profile_tasks=128,
                note="Source-only repeated-fixture throughput, not efficacy; setup once per target; no historical panels")


def stop_tree(process):
    import psutil
    import signal
    if process.poll() is not None:
        return
    descendants = psutil.Process(process.pid).children(recursive=True)
    groups = set()
    for pid in [process.pid] + [p.pid for p in descendants]:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    groups.discard(os.getpgrp())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def worker_factory(fixed_stage):
    def worker(stage, expected):
        import importlib.metadata
        import platform
        import shutil
        import threading
        import torch

        # Construct concrete paths on the Linux worker, never pickle WindowsPath.
        REMOTE_SOURCE = Path("/opt/mira")
        REMOTE_PANELS = Path("/opt/mira_panels")
        REMOTE_DATA = Path("/opt/mira_data")
        RESULTS = Path("/results/lifted_v4_l40s_20261009")

        if stage != fixed_stage or stage not in CHILD_MINUTES:
            raise ValueError("Worker accepts its fixed smoke/main stage only")
        started = time.monotonic()
        volume = modal.Volume.from_name("mira-lifted-v4-l40s")
        volume.reload()
        actual = frozen_inputs(REMOTE_SOURCE, REMOTE_PANELS, REMOTE_DATA,
                               REMOTE_SOURCE / "infra/modal_lifted_v4_l40s.py")
        if actual != expected:
            raise ValueError("Mounted inputs differ from the immutable local launch identity")
        packages = {p: importlib.metadata.version(p) for p in PACKAGES}
        if packages != PACKAGES or not torch.cuda.is_available():
            raise ValueError("Pinned packages and CUDA are required; no CPU fallback")
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        if "L40S" not in device or not 44 <= memory <= 50:
            raise ValueError("Require an L40S with forty-eight GB of GPU memory")
        code = REMOTE_SOURCE / "experiments/lifted_cavity"
        sys.path.insert(0, str(code))
        import bounded_v4
        bound_identity = dict(schema=1, source=actual["source"], panels=actual["panels"],
                              panel_manifest=actual["panel_manifest"],
                              packages={p: packages[p] for p in ("torch", "numpy", "scipy", "tabpfn")},
                              device=device, cuda=torch.version.cuda)
        bounded_v4.bind_manifest(RESULTS, bound_identity,
                                dict(commit=SOURCE_COMMIT, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        identity = dict(inputs=actual, bounded_identity=bound_identity)
        identity_path = RESULTS / "modal_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Fixed results root is already bound to a different identity")
        if not identity_path.exists():
            write_json(identity_path, identity)
        lock = RESULTS / "MODAL_RUNNING.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
        process, thread = None, None
        stopping, durability_failed = threading.Event(), threading.Event()
        commits = []
        status = dict(stage=stage, root=str(RESULTS), status="failed", exit_code=1,
                      requested_gpu="L40S", cpu=2, memory_mib=16384,
                      function_timeout_seconds=FUNCTION_SECONDS[stage], child_minutes=CHILD_MINUTES[stage],
                      automatic_retries=0, device=device, cuda=torch.version.cuda, packages=packages,
                      python=platform.python_version())
        logs = RESULTS / "logs"; logs.mkdir(exist_ok=True)
        runtime_path = RESULTS / ("modal_" + stage + "_runtime.json")
        attempt_path = RESULTS / ("modal_" + stage + "_attempt.json")
        def commit_periodically():
            while not stopping.wait(60):
                try:
                    volume.commit()
                    commits.append(dict(seconds=time.monotonic() - started, ok=True))
                except Exception as error:
                    commits.append(dict(seconds=time.monotonic() - started, ok=False, error=str(error)))
                    durability_failed.set()
                    return
        try:
            if attempt_path.exists():
                raise ValueError("This fixed stage was already attempted; no automatic retries")
            if (RESULTS / "RUNNING.lock").exists():
                raise ValueError("An earlier bounded process has not been cleared")
            write_json(attempt_path, dict(identity=identity, stage=stage, started_unix=time.time()))
            shutil.copytree(REMOTE_DATA / "beijing", RESULTS / "data/beijing", dirs_exist_ok=True)
            if stage == "main":
                gate = json.loads((RESULTS / "modal_smoke_gate.json").read_text())
                smoke_status = json.loads((RESULTS / "modal_smoke_runtime.json").read_text())
                if smoke_status.get("status") != "completed" or smoke_status.get("exit_code") != 0:
                    raise ValueError("Source smoke runtime did not complete successfully")
                if gate.get("passed") is not True or gate["identity"] != identity:
                    raise ValueError("Main requires passing smoke with the identical immutable identity")
                if digest(RESULTS / "smoke/report.json") != gate["smoke_report_sha256"]:
                    raise ValueError("Passing source smoke report changed")
                weights = weight_identity(RESULTS)
                if weights != gate["pretrained_weights"]:
                    raise ValueError("Actual pretrained TabPFN weights changed after source smoke")
                estimate = gate["execution_estimate"]
                if estimate != estimate_main(RESULTS) or not estimate["fits_cap"]:
                    raise ValueError("Frozen source throughput does not fit the main hard cap")
                write_json(RESULTS / "main_execution_gate.json", gate)
                status["estimated_seconds"] = estimate["estimated_seconds"]
            env = dict(os.environ, HF_HOME=str(RESULTS / "cache/huggingface"),
                       TABPFN_MODEL_CACHE_DIR=str(RESULTS / "cache/tabpfn"),
                       TABPFN_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONUNBUFFERED="1")
            if stage == "main":
                env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            remaining = FUNCTION_SECONDS[stage] - (time.monotonic() - started) - 45
            if remaining < CHILD_MINUTES[stage] * 60:
                raise ValueError("Preparation consumed the reserved child wall-time budget")
            stages = ["smoke"] if stage == "smoke" else ["main", "confirm", "pack"]
            command = [sys.executable, "-u", str(code / "bounded_v4.py"), "--root", str(RESULTS),
                       "--panels", str(REMOTE_PANELS), "--minutes", str(CHILD_MINUTES[stage]), *stages]
            status["command"] = command
            write_json(runtime_path, dict(status, status="running"))
            volume.commit()
            thread = threading.Thread(target=commit_periodically, daemon=True)
            thread.start()
            with (logs / ("modal_" + stage + ".log")).open("w", buffering=1) as log:
                process = subprocess.Popen(command, cwd=code, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + remaining
                while process.poll() is None:
                    if durability_failed.is_set() or time.monotonic() >= deadline:
                        stop_tree(process)
                        status["timed_out"] = not durability_failed.is_set()
                        break
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            status["exit_code"] = process.returncode
            if process.returncode != 0 or durability_failed.is_set() or status.get("timed_out"):
                raise RuntimeError("Bounded stage failed, timed out, or lost periodic durability")
            if stage == "smoke":
                profile_script = REMOTE_SOURCE / "infra/profile_lifted_v4_source.py"
                if digest(profile_script) != actual["source_profiler_sha256"]:
                    raise ValueError("Source-only profiler changed before execution")
                profile_remaining = FUNCTION_SECONDS[stage] - (time.monotonic() - started) - 45
                if profile_remaining < 180:
                    raise ValueError("Insufficient remaining smoke budget for the bounded source profile")
                profile_command = [sys.executable, "-u", str(profile_script), "--root", str(RESULTS),
                                   "--out", str(RESULTS / "source_profile.json")]
                status["source_profile_command"] = profile_command
                status["source_profile_timeout_seconds"] = 180
                with (logs / "source_profile.log").open("w", buffering=1) as log:
                    process = subprocess.Popen(profile_command, cwd=code, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = time.monotonic() + 180
                    while process.poll() is None:
                        if durability_failed.is_set() or time.monotonic() >= deadline:
                            stop_tree(process)
                            status["timed_out"] = not durability_failed.is_set()
                            break
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                if process.returncode != 0 or durability_failed.is_set() or status.get("timed_out"):
                    raise RuntimeError("Bounded source-only scoring profile failed or timed out")
                estimate = estimate_main(RESULTS)
                weights = weight_identity(RESULTS)
                gate = dict(passed=True, identity=identity, smoke_report_sha256=digest(RESULTS / "smoke/report.json"),
                            pretrained_weights=weights, execution_estimate=estimate)
                write_json(RESULTS / "pretrained_weights.json", weights)
                write_json(RESULTS / "modal_smoke_gate.json", gate)
                status.update(estimated_seconds=estimate["estimated_seconds"], main_fits_cap=estimate["fits_cap"])
            else:
                confirmation = RESULTS / "runs/confirm_v4.json"
                if not confirmation.is_file():
                    raise ValueError("Main exited without complete three-seed confirmation")
                if weight_identity(RESULTS) != weights:
                    raise ValueError("Pretrained weight files changed during main")
                status["confirmation_sha256"] = digest(confirmation)
            status["status"] = "completed"
        except Exception as error:
            status["error"] = str(error)
        finally:
            if process is not None and process.poll() is None:
                stop_tree(process)
            stopping.set()
            if thread is not None:
                thread.join(timeout=10)
                if thread.is_alive():
                    status.update(status="failed", error="Periodic volume commit did not finish")
            status.update(seconds=time.monotonic() - started, periodic_commits=commits)
            write_json(runtime_path, status)
            lock.unlink(missing_ok=True)
            try:
                volume.commit()
            except Exception as error:
                status.update(status="failed", final_commit_error=str(error))
        return {key: status[key] for key in ("stage", "root", "status", "exit_code", "seconds", "error",
                                            "estimated_seconds", "main_fits_cap", "final_commit_error") if key in status}
    return worker


if modal.is_local():
    repo = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("MIRA_V4_SOURCE_DIR", repo / "artifacts/inputs/lifted_v4_source")).resolve()
    panels = Path(os.environ.get("MIRA_V4_PANELS_DIR", repo / "artifacts/inputs/lifted_v4_panels")).resolve()
    data = Path(os.environ.get("MIRA_V4_DATA_DIR", repo / "artifacts/inputs/lifted_v4_data")).resolve()
    expected = frozen_inputs(source, panels, data, Path(__file__))
    manifest = repo / "artifacts/manifests/bound_source_identity_l40s.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        ledger = repo / "artifacts/manifests/compute_ledger.json"
        previous = json.loads(ledger.read_text()) if ledger.exists() else []
        if any(row.get("run_id") in {run_id for run_id, _ in RESERVATIONS.values()}
               and row.get("status") != "reserved_not_launched" for row in previous):
            raise ValueError("Preserve launch manifest after the first actual submission")
        write_json(manifest, expected)
    elif not manifest.exists():
        write_json(manifest, expected)
    app = modal.App("mira-lifted-v4-l40s")
    volume = modal.Volume.from_name("mira-lifted-v4-l40s", create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install(*(f"{package}=={version}" for package, version in PACKAGES.items()))
             .env({"PYTHONPATH": "/opt/mira", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "PYTHONUNBUFFERED": "1", "TABPFN_DISABLE_TELEMETRY": "1"})
             .add_local_dir(source, REMOTE_SOURCE.as_posix())
             .add_local_file(Path(__file__), (REMOTE_SOURCE / "infra/modal_lifted_v4_l40s.py").as_posix())
             .add_local_file(Path(__file__).with_name("profile_lifted_v4_source.py"),
                             (REMOTE_SOURCE / "infra/profile_lifted_v4_source.py").as_posix())
             .add_local_dir(panels, REMOTE_PANELS.as_posix())
             .add_local_dir(data, REMOTE_DATA.as_posix()))
    workers = {stage: app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(16384, 16384),
                                  timeout=FUNCTION_SECONDS[stage], startup_timeout=120,
                                  min_containers=0, max_containers=1, scaledown_window=2, retries=0,
                                  volumes={"/results": volume}, serialized=True, name="lifted_v4_l40s_" + stage)
               (worker_factory(stage)) for stage in CHILD_MINUTES}

    @app.local_entrypoint()
    def launch(stage: str = "smoke", spawn: bool = True):
        if stage not in workers:
            raise ValueError("Only smoke or main is accepted")
        ledger, attempt, run_id = authorize_submission(repo, stage, expected)
        try:
            call = workers[stage].spawn(stage, expected)
            record_submission(ledger, attempt, run_id, dict(status="running", function_call_id=call.object_id))
        except Exception as error:
            record_submission(ledger, attempt, run_id, dict(status="failed_submission", error=str(error)))
            raise
        print(json.dumps(dict(stage=stage, function_call_id=call.object_id, root=RESULTS.as_posix(),
                              manifest=str(manifest), note="Use modal run --detach to survive launcher exit")))
        if not spawn:
            try:
                result = call.get()
                record_submission(ledger, attempt, run_id, dict(status=result["status"], result=result))
                print(json.dumps(result))
            except Exception as error:
                record_submission(ledger, attempt, run_id, dict(status="failed_wait", error=str(error)))
                raise
