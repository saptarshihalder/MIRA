"""Hydration-safe successor: preserve the failed reservation and frozen v1 plan."""
from pathlib import Path
import modal

IS_LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[1] if IS_LOCAL else Path("/opt/mira")


def make_worker():
    # A nested function is pickled by value, with no infrastructure/core globals.
    def worker(protocol_relative: str, model: str, cache_volume):
        import io
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time
        import zipfile

        root = Path("/opt/mira")
        protocol = (root/protocol_relative).resolve()
        if not protocol.is_relative_to(root.resolve()) or model not in ("tabpfn:v2", "tabicl:v2"):
            raise ValueError("Invalid frozen remote protocol/backbone")
        out = Path(tempfile.mkdtemp(prefix="mira_compiler_v2_"))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(str(root/p) for p in ("src", ".", "experiments/mask_compiler"))
        command = [sys.executable, str(root/"experiments/mask_compiler/run_backbone.py"), "--protocol",
                   str(protocol), "--model", model, "--cache-root", "/cache", "--out", str(out)]
        started = time.monotonic()
        try:
            with (out/"runner.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=str(root), env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=780)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        except Exception as error:
            code = 1
            (out/"wrapper_error.json").write_text(json.dumps({"error":repr(error)}), encoding="utf-8")
        seconds = time.monotonic()-started
        try:
            cache_volume.commit()
        except Exception as error:
            code = code or 1
            (out/"cache_commit_error.json").write_text(json.dumps({"error":repr(error)}), encoding="utf-8")
        (out/"modal_runtime.json").write_text(json.dumps({"seconds":seconds,"exit_code":code,"command":command,
            "gpu":"T4","max_seconds":900,"wrapper":"modal_compiler_v2","remote_root":str(root),
            "estimated_active_compute_usd":seconds*(.000164+2*.0000131+8*.00000222),
            "invoice_status":"estimate; excludes startup/build/provider rounding"}, indent=2), encoding="utf-8")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob("*"):
                if path.is_file():
                    archive.write(path,path.relative_to(out))
        return {"archive":buffer.getvalue(),"seconds":seconds,"exit_code":code}
    return worker


cloud_worker = make_worker()


def load_successor(protocol, root):
    """Verify deployment-only amendment; the experimental plan remains byte-pinned."""
    import json
    import sys
    sys.path.insert(0,str(root/"src"))
    sys.path.insert(0,str(root/"experiments/mask_compiler"))
    from run_backbone import load_protocol, contained, sha
    plan,digest = load_protocol(protocol,root)
    files = plan["file_sha256"]
    predecessor = plan.get("predecessor",{})
    required = ("run_id","protocol_file","protocol_sha256","evidence_file","evidence_sha256")
    if any(not isinstance(predecessor.get(key),str) or not predecessor[key] for key in required):
        raise ValueError("Explicit aborted-before-inference predecessor evidence required")
    if "infra/modal_compiler_v2.py" not in files:
        raise ValueError("Successor omits the new wrapper source hash")
    uploaded = ("src/","scripts/","tests/","experiments/mask_compiler/","infra/","configs/","artifacts/manifests/",
                "artifacts/reports/trained_compiler_v1/","artifacts/reports/matched_compiler_v1/","artifacts/data/natural_compiler/")
    if any(path != "pyproject.toml" and not path.startswith(uploaded) for path in files):
        raise ValueError("Frozen hash map contains files outside the uploaded remote layout")
    previous_file = contained(root,predecessor["protocol_file"])
    if (sha(previous_file) != predecessor["protocol_sha256"] or
            previous_file.with_suffix(".sha256").read_text().strip() != predecessor["protocol_sha256"] or
            files.get(predecessor["protocol_file"]) != predecessor["protocol_sha256"]):
        raise ValueError("Historical frozen protocol proof differs")
    previous = json.loads(previous_file.read_text())
    if not previous.get("frozen") or plan["protocol_version"] == previous.get("protocol_version"):
        raise ValueError("Use a new protocol version while preserving the historical freeze")
    experiment_fields = ("models","ensembles","synthetic","real","views","compiler_checkpoint",
                         "linear_checkpoint","deepsets_checkpoint","checkpoint_sha256","decision_gate","budgets")
    if any(plan.get(key) != previous.get(key) for key in experiment_fields):
        raise ValueError("Hydration repair cannot change experiments, weights, checkpoints or budgets")
    if any(files.get(path) != checksum for path,checksum in previous["file_sha256"].items()):
        raise ValueError("Historical source/data hashes must remain unchanged")
    evidence_file = contained(root,predecessor["evidence_file"])
    if sha(evidence_file) != predecessor["evidence_sha256"] or files.get(predecessor["evidence_file"]) != predecessor["evidence_sha256"]:
        raise ValueError("Hydration/cost evidence SHA differs")
    evidence = json.loads(evidence_file.read_text())
    if (evidence.get("run_id") != predecessor["run_id"] or evidence.get("protocol_sha256") != predecessor["protocol_sha256"] or
            evidence.get("stage") != "container_hydration" or not str(evidence.get("app_id","")).startswith("ap-") or
            evidence.get("app_stopped") is not True or evidence.get("reservation_consumed") is not True or
            evidence.get("reserved_usd") != .5 or type(evidence.get("model_invocations")) is not int or evidence["model_invocations"] != 0 or
            type(evidence.get("prediction_files")) is not int or evidence["prediction_files"] != 0):
        raise ValueError("Evidence must establish stopped hydration failure, zero inference and consumed $.50 reservation")
    log_file = contained(root,evidence.get("log_file",""))
    if sha(log_file) != evidence.get("log_sha256") or files.get(evidence["log_file"]) != evidence["log_sha256"]:
        raise ValueError("Hydration log proof SHA differs")
    log = log_file.read_text(encoding="utf-8",errors="replace")
    if "ModuleNotFoundError" not in log or "infra" not in log:
        raise ValueError("Hydration log does not corroborate the reported import failure")
    result = root/"artifacts/runs"/predecessor["run_id"]
    if result.exists() and (any((result/"predictions").rglob("*.npz")) or
                            (result/"manifest.json").exists() and json.loads((result/"manifest.json").read_text()).get("completed_cells",0)):
        raise ValueError("Predecessor has inference artifacts; deployment-only amendment rejected")
    return plan,digest,evidence


def reserve_successor(ledger,budget,run_id,model,plan,digest,evidence):
    import json
    import math
    from datetime import datetime,timezone
    from infra.modal_mechanism import ledger_lock,atomic_ledger_write,validate_run_id
    validate_run_id(run_id)
    with ledger_lock(ledger):
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        old = [e for e in entries if e.get("phase") == "mask_compiler_backbone"]
        if len(old) != 1 or sum(float(e["reserved_usd"]) for e in old)+.5 > 1+1e-9:
            raise ValueError("Compiler budget permits only two calls/$1 including the hydration failure")
        first = old[0]
        if (first.get("run_id") != evidence["run_id"] or first.get("protocol_sha256") != evidence["protocol_sha256"] or
                first.get("status") != "failed" or first.get("reserved_usd") != .5 or
                first.get("failure_stage") != "container_hydration" or first.get("app_id") != evidence["app_id"] or
                first.get("app_stopped") is not True or first.get("model_invocations") != 0):
            raise ValueError("Evidence does not match the failed, charged predecessor ledger entry")
        if model not in plan["models"] or first.get("model") == model or any(e.get("run_id") == run_id for e in entries):
            raise ValueError("Only the previously unreserved backbone may run; no retry")
        values = [float(e["reserved_usd"]) for e in entries]
        cap = float(budget["total_cap_usd"])
        if not math.isfinite(cap) or any(not math.isfinite(v) or v < 0 for v in values) or sum(values)+.5 > cap-3+1e-9:
            raise ValueError("Global compute cap reached or invalid; preserve $3 reserve")
        entries.append({"run_id":run_id,"phase":"mask_compiler_backbone","model":model,"reserved_usd":.5,
                        "status":"reserved","max_seconds":900,"protocol_sha256":digest,"protocol_version":plan["protocol_version"],
                        "predecessor_run_id":first["run_id"],"predecessor_protocol_sha256":first["protocol_sha256"],
                        "failure_evidence_sha256":plan["predecessor"]["evidence_sha256"],"date":datetime.now(timezone.utc).isoformat()})
        atomic_ledger_write(ledger,entries)


def main(run_id:str,model:str,protocol_file:str="configs/trained_compiler_validation_v2.json"):
    if not IS_LOCAL:
        raise RuntimeError("Local entrypoint cannot execute in a cloud container")
    import io
    import json
    import zipfile
    from infra.modal_mechanism import finish_call,validate_run_id
    from run_backbone import natural_episodes,MODELS
    validate_run_id(run_id)
    if model not in MODELS:
        raise ValueError("Explicit v2 backbone required")
    protocol = Path(protocol_file).resolve()
    if not protocol.is_relative_to(ROOT.resolve()):
        raise ValueError("Protocol must remain inside the uploaded repository")
    plan,digest,evidence = load_successor(protocol,ROOT)
    list(natural_episodes(plan,ROOT))
    target = ROOT/"artifacts/runs"/run_id
    if target.exists():
        raise ValueError("Preserve existing results; choose a fresh run ID")
    ledger = ROOT/"artifacts/manifests/compute_ledger.json"
    budget = json.loads((ROOT/"configs/budget.json").read_text())
    reserve_successor(ledger,budget,run_id,model,plan,digest,evidence)
    updates = {"status":"failed"}
    try:
        payload = run_compiler.remote(str(protocol.relative_to(ROOT)).replace("\\","/"),model,cache)
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload["archive"])) as archive:
            if any(not (target/member.filename).resolve().is_relative_to(target.resolve()) for member in archive.infolist()):
                raise ValueError("Unsafe returned archive path")
            archive.extractall(target)
        updates.update(status="completed" if payload["exit_code"] == 0 else "failed",active_seconds=payload["seconds"],
                       exit_code=payload["exit_code"],results=str(target))
        print(json.dumps({"run_id":run_id,"model":model,**updates}))
        if payload["exit_code"]:
            raise SystemExit(payload["exit_code"])
    except BaseException as error:
        updates["error"] = repr(error)
        raise
    finally:
        finish_call(ledger,run_id,updates)


if IS_LOCAL:
    import sys
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/"experiments/mask_compiler"))
    from infra.modal_mechanism import cache,image
    app = modal.App("mira-mask-compiler-v2")
    compiler_image = image
    for directory in ("experiments/mask_compiler","infra","configs","tests","artifacts/manifests",
                      "artifacts/reports/trained_compiler_v1","artifacts/reports/matched_compiler_v1","artifacts/data/natural_compiler"):
        compiler_image = compiler_image.add_local_dir(ROOT/directory,"/opt/mira/"+directory)
    run_compiler = app.function(image=compiler_image,gpu="T4",cpu=(2,2),memory=(8192,8192),timeout=900,
        max_containers=1,scaledown_window=2,retries=0,serialized=True,name="run_compiler",volumes={"/cache":cache})(cloud_worker)
    main = app.local_entrypoint()(main)
