"""Create a saved analyzer accepting a hash-verified repair lineage."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/"experiments/mask_compiler/analyze_validation.py").read_text()
old='        if manifest["protocol_sha256"]!=digest: raise ValueError("Protocol mismatch")'
new='''        run_digest=manifest["protocol_sha256"]
        if run_digest!=digest:
            candidates=[p for p in (ROOT/"configs").glob("*.json") if sha(p)==run_digest]
            if len(candidates)!=1: raise ValueError("Unknown run protocol")
            child=json.loads(candidates[0].read_text())
            repair=child.get("gpu_repair",{})
            if repair.get("base_protocol_sha256")!=digest: raise ValueError("Unverified repair lineage")
            for field in ("models","ensembles","synthetic","real","views","compiler_checkpoint","linear_checkpoint","deepsets_checkpoint","checkpoint_sha256","decision_gate","budgets"):
                if child[field]!=plan[field]: raise ValueError("Changed experimental design")
            if any(child["file_sha256"].get(p)!=h for p,h in plan["file_sha256"].items()): raise ValueError("Historical source changed")
            if manifest["source_sha256"]!=child["file_sha256"]: raise ValueError("Run source mismatch")
        elif manifest["source_sha256"]!=plan["file_sha256"]: raise ValueError("Run source mismatch")'''
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('pairs.setdefault(r["episode_id"],set()).add(r["data_sha256"])','pairs.setdefault(r["episode_id"],{})[r["data_sha256"]]=data')
source=source.replace('"manifest_sha256":sha(directory/"manifest.json")','"manifest_sha256":sha(directory/"manifest.json"),"actual_protocol_sha256":manifest["protocol_sha256"]')
old='    if {a["model"] for a in audits}!=set(plan["models"]) or any(len(v)!=1 for v in pairs.values()): raise ValueError("Backbone pairing mismatch")'
new='''    if {a["model"] for a in audits}!=set(plan["models"]): raise ValueError("Backbone pairing mismatch")
    maximum_input_drift=0.
    for files in pairs.values():
        paths=list(files.values())
        for path in paths[1:]:
            with np.load(paths[0],allow_pickle=False) as a,np.load(path,allow_pickle=False) as b:
                if set(a.files)!=set(b.files): raise ValueError("Episode schema mismatch")
                for name in a.files:
                    if a[name].dtype.kind=="f":
                        if not np.allclose(a[name],b[name],atol=1e-12,rtol=0,equal_nan=True): raise ValueError("Episode floating inputs differ")
                        finite=np.isfinite(a[name])&np.isfinite(b[name])
                        if finite.any():maximum_input_drift=max(maximum_input_drift,float(np.max(np.abs(a[name][finite]-b[name][finite]))))
                    elif not np.array_equal(a[name],b[name]): raise ValueError("Episode labels/masks/IDs differ")'''
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('"cells":len(rows),"runs":audits','"cells":len(rows),"maximum_cross_platform_input_drift":maximum_input_drift,"runs":audits')
source=source.replace('trained_compiler_validation_v1.json','trained_compiler_validation_v2.json')
source=source.replace('"""Audit saved predictions before aggregating the frozen development screen."""','"""Audit originals and verified repair lineage; no artifact metadata normalization."""')
target=ROOT/"experiments/mask_compiler/analyze_validation_v2.py"
if target.exists():raise ValueError("Preserve existing analyzer")
target.write_text(source,encoding="utf-8")
print(target.name)
