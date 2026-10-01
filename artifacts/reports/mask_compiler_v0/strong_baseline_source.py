"""Additional sparse baseline; reuse saved episodes, never rerun prior predictions."""
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from feasibility import ROOT, CS, features
from mira.data import Observations
from mira.metrics import expected_nll

out = ROOT / "artifacts/reports/mask_compiler_v0"
target = out / "strong_baseline.json"
if target.exists():
    raise ValueError("Preserve existing baseline outputs")
old = json.loads((out / "results.json").read_text())
rows = []
for file in sorted(out.glob("*_episode.npz")):
    name = file.stem.removesuffix("_episode")
    saved = np.load(file, allow_pickle=False)
    obs = Observations(saved["uc"], saved["xc"], saved["mc"])
    query = Observations(saved["uq"], saved["xq"], saved["mq"])
    xc, xq = features(obs, all_parities=True), features(query, all_parities=True)
    cv = StratifiedKFold(3, shuffle=True, random_state=91)
    fit = GridSearchCV(LogisticRegression(solver="liblinear", l1_ratio=1., max_iter=1000),
                       {"C":list(CS)}, cv=cv, scoring="neg_log_loss", n_jobs=1, error_score="raise")
    fit.fit(xc, saved["yc"])
    p = fit.predict_proba(xq)[:, 1]
    prediction = out / (name+"_full_parity_l1.npz")
    np.savez_compressed(prediction, p=p)
    row = next(r for r in old if r["task"]==name)
    rows.append({"task":name,"order":row["order"],"context":row["context"],"gamma":row["gamma"],
                 "seed":row["seed"],"method":"full_parity_l1","expected_nll":expected_nll(p,saved["oracle"]),
                 "selection":{"C":fit.best_params_["C"],"cv_nll":-fit.best_score_,"C_count":len(CS)},
                 "prediction_file":prediction.name,"prediction_sha256":hashlib.sha256(prediction.read_bytes()).hexdigest()})
assert len(rows)==80
target.write_text(json.dumps({"rows":rows,"paid_usd":0,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "scope":"Same support labels,3fold seed91/C grid. L1 full15-parity basis versus12 candidate4-bit bases; candidate/model complexity differs and is disclosed."},indent=2)+"\n")
lines=["# Sparse full-parity control", "", "80 additional predictions, reusing saved development episodes. No paid call or TFM inference.", "", "| Order | Labels | Gamma | Selected compiler | Full-parity L1 | L1 minus compiler |", "|---|---|---|---:|---:|---:|"]
for order in range(1,5):
    for context in (32,128):
        for gamma in (0.,.9):
            compiler=np.mean([r["expected_nll"] for r in old if (r["order"],r["context"],r["gamma"],r["method"])==(order,context,gamma,"support_selected")])
            l1=np.mean([r["expected_nll"] for r in rows if (r["order"],r["context"],r["gamma"])==(order,context,gamma)])
            lines.append(f"| {order} | {context} | {gamma} | {compiler:.5f} | {l1:.5f} | {l1-compiler:.5f} |")
(out / "strong_baseline.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"additional_predictions":80,"paid_usd":0}))
