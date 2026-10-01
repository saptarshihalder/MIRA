"""Audit originals and verified repair lineage; no artifact metadata normalization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def nll(p,y):
    p=np.clip(p,1e-6,1-1e-6)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
def write(p,value): p.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")

def analyze(protocol, runs, out):
    plan=json.loads(protocol.read_text()); digest=sha(protocol)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; audits=[]; pairs={}; discrepancies=[]
    for directory in runs:
        manifest=json.loads((directory/"manifest.json").read_text())
        records=json.loads((directory/"results.json").read_text())
        errors=json.loads((directory/"errors.json").read_text())
        run_digest=manifest["protocol_sha256"]
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
        elif manifest["source_sha256"]!=plan["file_sha256"]: raise ValueError("Run source mismatch")
        if manifest["status"]!="complete" or errors: raise ValueError("Incomplete/failing run")
        if len(records)!=manifest["expected_cells"] or len(records)!=162: raise ValueError("Missing cells")
        seen=set(); episodes=set()
        for r in records:
            key=(r["episode_id"],r["mode"])
            if key in seen: raise ValueError("Duplicate cell")
            seen.add(key); episodes.add(r["episode_id"])
            prediction=directory/r["prediction_file"]; data=directory/r["data_file"]
            if sha(prediction)!=r["prediction_sha256"] or sha(data)!=r["data_sha256"]: raise ValueError("Array hash mismatch")
            with np.load(prediction,allow_pickle=False) as z: p=z["probability"].astype(float)
            with np.load(data,allow_pickle=False) as z:
                y=z["y_query_evaluator_only"]
                if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid probability")
                values={"empirical_nll":nll(p,y),"empirical_brier":float(np.mean((p-y)**2))}
                if r["kind"]=="synthetic": values["expected_nll"]=nll(p,z["oracle"])
                else:
                    if set(z["support_ids"])&set(z["query_ids"]): raise ValueError("Row overlap")
                r["auroc"]=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
            for name,value in values.items():
                delta=abs(value-r[name]); discrepancies.append(delta)
                if delta>1e-10: raise ValueError("Score reconstruction mismatch")
            pairs.setdefault(r["episode_id"],{})[r["data_sha256"]]=data
            rows.append(r)
        for episode in episodes:
            if {v for e,v in seen if e==episode}!=set(plan["views"]): raise ValueError("View matrix incomplete")
        audits.append({"model":manifest["model"],"cells":len(records),"episodes":len(episodes),"seconds":manifest["seconds"],"manifest_sha256":sha(directory/"manifest.json"),"actual_protocol_sha256":manifest["protocol_sha256"]})
    if {a["model"] for a in audits}!=set(plan["models"]): raise ValueError("Backbone pairing mismatch")
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
                    elif not np.array_equal(a[name],b[name]): raise ValueError("Episode labels/masks/IDs differ")
    summaries=[]; contrasts=[]
    for model in plan["models"]:
        for kind in ("synthetic","natural"):
            conditions=sorted({(r["order"],r["gamma"]) if kind=="synthetic" else (r["dataset"],None) for r in rows if r["model"]==model and r["kind"]==kind})
            metric="expected_nll" if kind=="synthetic" else "empirical_nll"
            for first,second in conditions:
                selected=[r for r in rows if r["model"]==model and r["kind"]==kind and ((r["order"],r["gamma"]) if kind=="synthetic" else (r["dataset"],None))==(first,second)]
                table={(r["episode_id"],r["mode"]):r for r in selected}
                ids=sorted({r["episode_id"] for r in selected})
                for mode in plan["views"]:
                    values=[table[e,mode][metric] for e in ids]; weights=[table[e,mode]["queries"] for e in ids]
                    summaries.append({"model":model,"kind":kind,"order":first if kind=="synthetic" else None,"gamma":second,"dataset":first if kind=="natural" else None,"mode":mode,"nll":float(np.average(values,weights=weights)),"episodes":len(ids)})
                for reference in [v for v in plan["views"] if v!="trained"]:
                    gain=[table[e,reference][metric]-table[e,"trained"][metric] for e in ids]
                    weights=[table[e,"trained"]["queries"] for e in ids]
                    contrasts.append({"model":model,"kind":kind,"condition":[first,second],"reference":reference,"gain_over_reference":float(np.average(gain,weights=weights)),"paired_episode_gains":gain})
    gates=[]
    for model in plan["models"]:
        by_gamma={}
        for gamma in (0.,.9):
            values=[]
            for seed in plan["synthetic"]["seeds"]:
                s=[r for r in rows if r["model"]==model and r["kind"]=="synthetic" and r["gamma"]==gamma and r["seed"]==seed]
                d={(r["order"],r["mode"]):r["expected_nll"] for r in s}
                values.append(float(np.mean([d[o,"identity"]-d[o,"trained"] for o in plan["synthetic"]["orders"]])))
            by_gamma[str(gamma)]=values
        gain=float(np.mean(by_gamma["0.9"])); harm=-float(np.mean(by_gamma["0.0"]))
        gates.append({"model":model,"high_signal_gain":gain,"null_signal_harm":harm,"passes_numeric_development_screen":gain>=.01 and harm<=.01,"seed_units":by_gamma,"limits":"Three seeds with orders averaged within seed; descriptive development only. Passing also requires evidence beyond strong equally informed baselines and a separate fresh confirmation."})
    report={"status":"audited","protocol_sha256":digest,"source_sha256":sha(Path(__file__)),"cells":len(rows),"maximum_cross_platform_input_drift":maximum_input_drift,"runs":audits,"maximum_score_discrepancy":max(discrepancies),"gates":gates,"uncertainty":"No confirmatory intervals. Natural results aggregate overlapping grouped folds within each of two fixed datasets; not six independent replications.","summary":summaries,"contrasts":contrasts}
    write(out/"report.json",report)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    lines=["# Frozen trained-compiler development validation","","All saved probability/episode hashes and losses reconstructed. Positive gain means trained loss is lower; no confirmation claim.","","| Backbone | Kind | Condition | Native | Identity | Trained | Heuristic | Linear | DeepSets | CV |","|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in plan["models"]:
        conditions=[]
        for r in summaries:
            condition=(r["kind"],r["order"],r["gamma"],r["dataset"])
            if r["model"]==model and condition not in conditions:conditions.append(condition)
        for condition in conditions:
            d={r["mode"]:r["nll"] for r in summaries if r["model"]==model and (r["kind"],r["order"],r["gamma"],r["dataset"])==condition}
            lines.append(f"| {model} | {condition[0]} | {condition[3] or ('k'+str(condition[1])+', gamma='+str(condition[2]))} | "+" | ".join(f"{d[v]:.5f}" for v in ("native","identity","trained","heuristic","matched_linear","deepsets","cv_basis"))+" |")
    for g in gates:lines.append(f"\n{g['model']}: high-signal gain {g['high_signal_gain']:.6f}; null harm {g['null_signal_harm']:.6f}; numeric screen pass={g['passes_numeric_development_screen']}.")
    lines+= ["",report["uncertainty"],"", "Confidence fallback is not a no-harm guarantee. Natural support encoder uses zero U proxy and numeric category codes; clinical and general missingness benefits are not established."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","cells","maximum_score_discrepancy","gates")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,default=ROOT/"configs/trained_compiler_validation_v2.json");p.add_argument("--runs",type=Path,nargs="+",required=True);p.add_argument("--out",type=Path,default=ROOT/"artifacts/reports/trained_compiler_backbone_v1");a=p.parse_args();analyze(a.protocol,a.runs,a.out)
