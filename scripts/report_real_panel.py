"""Recompute real-panel scores from saved predictions; datasets are aggregation units."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from mira.artifacts import atomic_json, file_sha256, load_rows
from mira.metrics import expected_nll
from mira.reporting import mean_interval


def verify(run):
    manifest=json.loads((run / "manifest.json").read_text())
    rows=load_rows(run)
    if manifest["status"]!="complete" or len(rows)!=manifest["expected_cells"] or manifest["errors"]:
        raise ValueError("Incomplete panel; no silent denominator change")
    truth={}; predictions={}
    for row in rows:
        for field in ("data", "prediction"):
            if file_sha256(run / row[field+"_file"])!=row[field+"_sha256"]:
                raise ValueError("Saved panel hash mismatch")
        filename=row["data_file"]
        if filename not in truth:
            with np.load(run / filename,allow_pickle=False) as data:
                truth[filename]={k:data[k].copy() for k in data.files}
        data=truth[filename]
        if set(data["support_row_ids"]) & set(data["query_row_ids"]) or set(data["support_group_ids"]) & set(data["query_group_ids"]):
            raise ValueError("Panel row/group leakage")
        for suffix in ("c","q"):
            if not np.array_equal(data["m"+suffix], data["native_m"+suffix] | data["imposed_m"+suffix]) or not np.array_equal(np.isnan(data["x"+suffix]),data["m"+suffix]):
                raise ValueError("Native/imposed mask union mismatch")
        with np.load(run / row["prediction_file"],allow_pickle=False) as saved:
            p=saved["probability"]
            if not np.array_equal(saved["query_row_ids"],data["query_row_ids"]):
                raise ValueError("Prediction IDs mismatch")
            if p.shape!=data["yq"].shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError("Invalid probabilities")
            scores={"empirical_nll":expected_nll(p,data["yq"]),"brier":float(np.mean((p-data["yq"])**2)),"auroc":float(roc_auc_score(data["yq"],p))}
        if any(not np.isclose(row[k],v,rtol=0,atol=1e-12) for k,v in scores.items()):
            raise ValueError("Stored score does not reproduce")
        predictions[(row["fold"],row["rate"],row["gamma"],row["model"],row["mode"])]=row
    effects=[]
    for rate in manifest["protocol"]["rates"]:
        for gamma in manifest["protocol"]["gammas"]:
            for model in manifest["protocol"]["models"]:
                for baseline, comparison, name in (("native","native_indicators","native_to_indicators"),
                    ("native_shuffled","native_indicators","shuffled_to_indicators"),("native","native_shuffled","native_to_shuffled")):
                    pairs=[(predictions[(fold,rate,gamma,model,baseline)],predictions[(fold,rate,gamma,model,comparison)]) for fold in range(5)]
                    if any(a["data_sha256"]!=b["data_sha256"] for a,b in pairs):
                        raise ValueError("Unpaired panel episode")
                    weights=np.array([a["queries"] for a,b in pairs],float)
                    effects.append({"dataset":manifest["dataset"],"rate":rate,"gamma":gamma,"model":model,"comparison":name,
                        "gain":float(np.average([a["empirical_nll"]-b["empirical_nll"] for a,b in pairs],weights=weights)),
                        "baseline_nll":float(np.average([a["empirical_nll"] for a,b in pairs],weights=weights)),
                        "comparison_nll":float(np.average([b["empirical_nll"] for a,b in pairs],weights=weights)),
                        "queries":int(weights.sum()),"folds":5})
    return manifest, effects


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-root",type=Path,default=ROOT / "artifacts/runs")
    parser.add_argument("--prefix",default="real_panel_v1_")
    parser.add_argument("--out",type=Path,default=ROOT / "artifacts/reports/real_panel_v1")
    parser.add_argument("--dataset")
    args=parser.parse_args()
    plan=json.loads((ROOT / "configs/real_panel_v1.json").read_text())
    datasets=[args.dataset] if args.dataset else plan["datasets"]
    effects=[]; cells=0; provenance={}
    for dataset in datasets:
        run=args.run_root / (args.prefix+dataset)
        manifest, entries=verify(run)
        if manifest["frozen_protocol_sha256"]!=file_sha256(ROOT / "configs/real_panel_v1.json"):
            raise ValueError("Frozen panel identity mismatch")
        effects.extend(entries); cells+=manifest["completed_cells"]
        provenance[dataset]={"manifest_sha256":file_sha256(run / "manifest.json"),"checkpoint_sha256":manifest["checkpoint_sha256"],"dataset_sha256":manifest["dataset_sha256"],"split_sha256":manifest["split_sha256"]}
    args.out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for row in effects: grouped[(row["rate"],row["gamma"],row["model"],row["comparison"])].append(row)
    aggregate=[]
    for (rate,gamma,model,comparison), rows in sorted(grouped.items()):
        aggregate.append({"rate":rate,"gamma":gamma,"model":model,"comparison":comparison,
            **mean_interval([r["gain"] for r in rows]),"baseline_nll":float(np.mean([r["baseline_nll"] for r in rows])),
            "comparison_nll":float(np.mean([r["comparison_nll"] for r in rows]))})
    summary={"completed_cells":cells,"datasets":datasets,"dataset_effects":effects,"aggregate":aggregate,"provenance":provenance,
        "interpretation":"Descriptive fixed-panel empirical losses. Folds averaged with query weights within each dataset; dataset-balanced aggregation. Six datasets, not30 folds or query rows, are uncertainty units. Unadjusted nominal95% t intervals; no real-data oracle; retrospective imposed label association, not natural-missingness or clinical benefit."}
    stem=args.dataset or "summary"
    atomic_json(args.out / (stem+".json"),summary)
    if not args.dataset:
        lines=["# Frozen six-dataset real-covariate panel","",f"Verified {cells} cells across six datasets and five frozen folds each.","",summary["interpretation"],"", "| Rate | Gamma | Model | Native minus indicators | Interval |", "|---|---|---|---:|---|"]
        for row in aggregate:
            if row["comparison"]=="native_to_indicators":
                lines.append(f"| {row['rate']} | {row['gamma']} | {row['model']} | {row['mean']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |")
        (args.out / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"datasets":len(datasets),"verified_cells":cells}))

if __name__=="__main__": main()
