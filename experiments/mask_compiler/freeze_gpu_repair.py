"""One new GPU repair authorized after the original pilot was exhausted."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=ROOT/"configs/trained_compiler_validation_v2.json"
    target=ROOT/"configs/compiler_gpu_repair_v1.json"
    if target.exists():raise ValueError("Preserve frozen GPU repair")
    plan=json.loads(base.read_text())
    entries=json.loads((ROOT/"artifacts/manifests/compute_ledger.json").read_text())
    failures=[e for e in entries if e.get("phase")=="mask_compiler_backbone"]
    if len(failures)!=2 or any(e["status"]!="failed" for e in failures):raise ValueError("Unexpected pilot ledger")
    ledger_hash=hashlib.sha256(json.dumps(failures,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    plan["protocol_version"]="compiler_gpu_repair_v1"
    plan["gpu_repair"]={"authorization":{"origin":"human_user","date":"2026-10-02","request":"use gpu man, also minimize the token usage lol, use modal and colab pro"},
        "base_protocol_file":base.relative_to(ROOT).as_posix(),"base_protocol_sha256":sha(base),"phase":"mask_compiler_gpu_repair",
        "model":"tabicl:v2","max_calls":1,"reservation_usd":.5,"failed_pilot_ledger_sha256":ledger_hash}
    files=[base,base.with_suffix('.sha256'),Path(__file__),ROOT/"infra/modal_compiler_v3.py",ROOT/"tests/test_modal_compiler_v3.py",
           ROOT/"experiments/mask_compiler/analyze_validation_v2.py",ROOT/"experiments/mask_compiler/build_lineage_analyzer.py",
           ROOT/"artifacts/manifests/compiler_validation_v2_prefit_failure.json"]
    for p in files:plan["file_sha256"][p.relative_to(ROOT).as_posix()]=sha(p)
    target.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    target.with_suffix('.sha256').write_text(sha(target)+"\n",encoding="utf-8")
    print(json.dumps({"protocol_sha256":sha(target),"failed_pilot_ledger_sha256":ledger_hash,"additional_reservation_usd":.5}))
if __name__=="__main__":main()
