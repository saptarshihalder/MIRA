"""Plot descriptive effects from the verified six-dataset panel summary."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
summary=json.loads((root / "artifacts/reports/real_panel_v1/summary.json").read_text())
models=["tabpfn:v2","tabicl:v2","xgboost","logistic"]
fig, axes=plt.subplots(2,2,figsize=(8.2,6),sharex=True,layout="constrained")
for ax,(rate,gamma) in zip(axes.flat,[(.1,0),(.1,.8),(.5,0),(.5,.8)]):
    for x,model in enumerate(models):
        values=[r["gain"] for r in summary["dataset_effects"] if r["rate"]==rate and r["gamma"]==gamma and r["model"]==model and r["comparison"]=="native_to_indicators"]
        assert len(values)==6
        row=next(r for r in summary["aggregate"] if r["rate"]==rate and r["gamma"]==gamma and r["model"]==model and r["comparison"]=="native_to_indicators")
        ax.scatter(x+np.linspace(-.13,.13,6),values,s=20,color="#999999",alpha=.8)
        ax.errorbar(x,row["mean"],yerr=[[row["mean"]-row["ci_low"]],[row["ci_high"]-row["mean"]]],fmt="o",capsize=4,color="#1c6391")
    ax.axhline(0,color="black",linewidth=.7)
    ax.set_title(f"Baseline rate {rate}; association γ={gamma}",fontsize=10)
    ax.set_xticks(range(4),["TabPFN\nv2","TabICL\nv2","XGBoost","Logistic"],fontsize=9)
    ax.set_ylabel("Native − indicator log-loss (nats)",fontsize=9)
fig.suptitle("Fixed real-covariate panel under imposed missingness",fontsize=13)
fig.supxlabel("Gray: individual datasets; blue: dataset-balanced mean and nominal 95% interval\nSix dataset units; folds overlap; retrospective label association",fontsize=9)
fig.savefig(root / "artifacts/figures/real_panel_v1.png",dpi=180)
fig.savefig(root / "artifacts/figures/real_panel_v1.svg")
