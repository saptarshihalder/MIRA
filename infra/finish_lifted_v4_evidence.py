"""One-shot local evidence completion after the existing GPU observer finishes.

No GPU launch, retries, recipe changes, manuscript claims or Git publication.
Fail closed on every stage. A later review interprets results and edits the paper.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
MODAL = REPO.parent / ".venv-modal/Scripts/python.exe"
CPU = REPO.parent / ".venv-compiler-cpu/Scripts/python.exe"
DELIVERY = REPO / "artifacts/runs/lifted_v4_delivery"
RESULT = REPO / "artifacts/reports/lifted_v4_launch/resume_function_result_20261011.json"
ROOT = "artifacts/runs/lifted_v4_final"
PANELS = "artifacts/inputs/lifted_v4_panels"
INPUTS = ["infra/collect_lifted_v4_results.py", "infra/verify_lifted_v4_results.py",
          "infra/replay_lifted_v4_predictions.py", "infra/integrate_lifted_v4_verified.py",
          "infra/summarize_lifted_v4_seed_sensitivity.py", "infra/build_lifted_review_package.py",
          "infra/reproduce_lifted_paper_exports.py", "infra/build_lifted_supplement.py",
          "experiments/lifted_cavity/make_paper.py", "infra/finish_lifted_v4_evidence.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = DELIVERY / "completion_plan.json"
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if set(plan["tool_hashes"]) != set(INPUTS):
        raise ValueError("Completion plan must bind every completion tool")
    with (DELIVERY / "worker_started.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), plan_sha256=digest(manifest)), stream)
    stage, report = "wait", dict(status="waiting", automatic_retries=0, new_gpu_calls=0)
    try:
        for _ in range(240):
            if RESULT.exists():
                break
            time.sleep(60)
        else:
            raise TimeoutError("One-shot local wait expired; no GPU call was restarted")
        upstream = json.loads(RESULT.read_text(encoding="utf-8"))
        if upstream.get("status") != "completed" or upstream.get("exit_code") != 0:
            raise ValueError("Upstream GPU call failed/incomplete; preserve all partial evidence")
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                           PATH=str(CPU.parent) + os.pathsep + os.environ.get("PATH", ""))

        def run(name, command, timeout):
            nonlocal stage
            stage = name
            for file, sha in plan["tool_hashes"].items():
                if digest(REPO / file) != sha:
                    raise ValueError(f"Completion tool changed after launch: {file}")
            if (DELIVERY / f"{name}.stdout.txt").exists() or (DELIVERY / f"{name}.stderr.txt").exists():
                raise ValueError(f"Stage already attempted: {name}")
            print(f"START {name}", flush=True)
            value = subprocess.run([str(x) for x in command], cwd=REPO, env=environment,
                                   capture_output=True, text=True, timeout=timeout,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            (DELIVERY / f"{name}.stdout.txt").write_text(value.stdout, encoding="utf-8")
            (DELIVERY / f"{name}.stderr.txt").write_text(value.stderr, encoding="utf-8")
            if value.returncode:
                raise RuntimeError(f"{name} exit {value.returncode}; inspect preserved logs")
            print(f"PASS {name}", flush=True)
            return value

        run("collect", [MODAL, "infra/collect_lifted_v4_results.py", "--collect"], 900)
        verified = run("verify", [CPU, "infra/verify_lifted_v4_results.py", "--root", ROOT, "--panels", PANELS], 600)
        verification = json.loads(verified.stdout)
        if verification.get("verified") is not True:
            raise ValueError("Independent verification did not pass")
        run("replay", [CPU, "infra/replay_lifted_v4_predictions.py", "--root", ROOT, "--panels", PANELS], 960)
        run("stage", [CPU, "infra/integrate_lifted_v4_verified.py", "--root", ROOT, "--panels", PANELS], 600)
        run("merge", [CPU, "infra/integrate_lifted_v4_verified.py", "--root", ROOT, "--panels", PANELS, "--merge"], 600)
        run("seed_audit", [CPU, "infra/summarize_lifted_v4_seed_sensitivity.py", "--root", ROOT, "--panels", PANELS], 600)
        archive = DELIVERY / "lifted_review_package.zip"
        run("package", [CPU, "infra/build_lifted_review_package.py", "--root", ROOT, "--panels", PANELS, "--out", archive], 600)
        extract = DELIVERY / "review_extract"
        extract.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                target = extract / info.filename
                if (not target.resolve().is_relative_to(extract.resolve()) or "\\" in info.filename
                        or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError("Unsafe local review archive path")
            package.extractall(extract)
        run("regenerate", [CPU, extract / "reproduce_paper.py"], 360)
        run("extracted_model_tests", [CPU, "-m", "pytest", extract / "tests/test_lifted_cavity.py",
                                      extract / "tests/test_lifted_gauge.py", "-q"], 300)
        parse = lambda file: dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", Path(file).read_text(encoding="utf-8")))
        before = parse(REPO / "paper/lifted_cavity/generated/numbers.tex")
        after = parse(extract / "regenerated/generated/numbers.tex")
        changed = [key for key, value in before.items() if after.get(key) != value]
        if changed:
            raise ValueError("Historical numeric macros changed: " + ", ".join(changed))
        report.update(status="evidence-ready-manuscript-pending", checkpoints=verification["checkpoints"],
                      score_files=verification["score_files"], endpoints=verification["endpoints"],
                      historical_macros_unchanged=len(before), generated_macros=len(after),
                      review_archive_sha256=digest(archive),
                      remaining="Interpret every result, edit and compile the same manuscript, review layout/anonymity, reconcile charges, push evidence; no readiness verdict yet")
    except Exception as error:
        report.update(status="failed", stage=stage, reason=f"{type(error).__name__}: {error}")
    with (DELIVERY / "completion_result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "evidence-ready-manuscript-pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
