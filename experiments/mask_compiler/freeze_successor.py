"""Freeze a hydration repair without changing the scored experiment design."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/"configs/trained_compiler_validation_v1.json"
    target=ROOT/"configs/trained_compiler_validation_v2.json"
    if target.exists():raise ValueError("Preserve existing successor freeze")
    evidence=ROOT/"artifacts/manifests/compiler_validation_v1_startup_failure.json"
    failure=json.loads(evidence.read_text())
    plan=json.loads(old.read_text())
    plan["protocol_version"]="trained_compiler_validation_v2"
    plan["predecessor"]={"run_id":failure["run_id"],"protocol_file":old.relative_to(ROOT).as_posix(),"protocol_sha256":sha(old),
                         "evidence_file":evidence.relative_to(ROOT).as_posix(),"evidence_sha256":sha(evidence),"app_id":failure["app_id"]}
    plan["device_plan"]={"tabpfn:v2":"local CPU, torch2.8 CPU wheel; same exact cached checkpoint/four ensembles",
                         "tabicl:v2":"Modal T4 CUDA, pinned image/four ensembles"}
    plan["cpu_execution"]={"model":"tabpfn:v2","device":"cpu","threads":2,"max_seconds":2400}
    plan["engineering_benchmark_before_successor_freeze"]="One TabPFN native order2/gamma0.9/seed74000 episode benchmarked on CPU; 2.312 seconds, exact checkpoint hashes, no warnings. Excluded as an engineering timing result; all pilot seeds are development. Device/backend numerical equivalence is not established."
    plan["repair_scope"]="Container hydration only. First paid TabPFN reservation remains consumed with zero predictions. Second paid call is the other backbone, never a retry. Local CPU TabPFN completion uses the unchanged episode/selection design and discloses device differences."
    files=[old,old.with_suffix('.sha256'),evidence,ROOT/failure["log_file"],Path(__file__),
           ROOT/"infra/modal_compiler_v2.py",ROOT/"tests/test_modal_compiler_v2.py",
           ROOT/"experiments/mask_compiler/run_cpu_backbone.py",ROOT/"experiments/mask_compiler/real_baselines_v2.py"]
    for p in files:plan["file_sha256"][p.relative_to(ROOT).as_posix()]=sha(p)
    target.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    target.with_suffix('.sha256').write_text(sha(target)+"\n",encoding="utf-8")
    print(json.dumps({"protocol_sha256":sha(target),"frozen_files":len(plan["file_sha256"]),"cells":324,"repair_only":True}))
if __name__=="__main__":main()
