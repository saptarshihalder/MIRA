"""Full-feature CPU controls on immutable naturally missing grouped episodes."""
import argparse
import json
import hashlib
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
from mira.panel_data import load_panel, validate_splits
from mira.panel_runner import logistic


def run(protocol, out):
    plan=json.loads(protocol.read_text())
    if out.exists(): raise ValueError("Preserve prior outputs")
    out.mkdir(parents=True)
    rows=[]
    with threadpool_limits(limits=1):
        for spec in plan["real"]:
            directory=ROOT / spec["directory"]
            panel=load_panel(directory)
            splits=json.loads((directory / "splits.json").read_text())
            validate_splits(panel,splits)
            for split in splits["folds"]:
                if split["fold"] not in spec["folds"]: continue
                ci,qi=np.array(split["support_indices"]),np.array(split["query_indices"])
                episode=f"{directory.name}_fold{split['fold']}"
                assert not set(panel.group_ids[ci]) & set(panel.group_ids[qi])
                xc,xq=panel.X[ci],panel.X[qi]
                yc=panel.y[ci]
                for mode in ("native","identity"):
                    c=xc if mode=="native" else np.column_stack([xc,panel.native_mask[ci]])
                    q=xq if mode=="native" else np.column_stack([xq,panel.native_mask[qi]])
                    for kind in ("logistic","histgb"):
                        model=logistic(92000+split["fold"],yc,panel.group_ids[ci]) if kind=="logistic" else HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1.,early_stopping=False,random_state=92000+split["fold"])
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(c,yc)
                            p=model.predict_proba(q)[:,int(np.where(model.classes_==1)[0][0])]
                        path=out / f"{episode}_{kind}_{mode}.npz"
                        np.savez_compressed(path,p=p,query_row_ids=panel.row_ids[qi])
                        yq=panel.y[qi]
                        pc=np.clip(p,1e-6,1-1e-6)
                        rows.append({"episode_id":episode,"dataset":directory.name,"fold":split["fold"],"kind":"natural",
                            "model":kind,"mode":mode,"queries":len(qi),"empirical_nll":float(np.mean(-yq*np.log(pc)-(1-yq)*np.log1p(-pc))),
                            "brier":float(np.mean((p-yq)**2)),"auroc":float(roc_auc_score(yq,p)),
                            "prediction_file":path.name,"prediction_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                            "warnings":[str(w.message) for w in caught],"chosen_C":model.best_params_["clf__C"] if kind=="logistic" else None})
    (out / "results.json").write_text(json.dumps(rows,indent=2)+"\n")
    manifest={"status":"complete","predictions":len(rows),"paid_usd":0,"protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"scope":"Full numeric/native-mask inputs; original categorical codes treated numerically. Group-isolated support CV for logistic; fixed HistGB. Empirical naturallymissing-data scores, no oracle or clinicalclaim."}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=ROOT / "configs/trained_compiler_validation_v1.json")
    parser.add_argument("--out",type=Path,default=ROOT / "artifacts/reports/natural_compiler_baselines_v1")
    a=parser.parse_args(); run(a.protocol,a.out)
