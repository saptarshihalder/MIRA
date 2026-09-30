"""Plot development effects from verified saved-prediction summaries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
colors = {"tabpfn:v2": "#2563a6", "tabicl:v2": "#d36b27", "xgboost": "#31845b"}
for ax, run, title in zip(axes, ["development_gaussian", "development_collision"],
                          ["Gaussian observed values", "Observed values equal imputation constant"]):
    summary = json.loads((root / "artifacts/reports" / run / "summary.json").read_text())
    for model, color in colors.items():
        rows = sorted((e for e in summary["effects"] if e["model"] == model
                       and e["comparison"] == "native_to_indicators"), key=lambda e:e["gamma"])
        x, y = [e["gamma"] for e in rows], [e["mean"] for e in rows]
        errors = [[e["mean"]-e["ci_low"] for e in rows], [e["ci_high"]-e["mean"] for e in rows]]
        ax.errorbar(x, y, yerr=errors, fmt="o-", label=model, color=color,
                    linewidth=1.5, markersize=4, capsize=3)
    ax.axhline(0, color="#555555", linewidth=.8, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Mask signal strength (gamma)")
    ax.set_xticks([0,.25,.5,.75,.9])
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Native minus indicator log-loss (nats)")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("MIRA development: representation effects depend on observed values", fontsize=12)
fig.text(.5,.012,"3 task draws per condition; 95% paired t intervals. Development only; confirmation pending.",
         ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95))
out = root / "artifacts/figures"
out.mkdir(parents=True,exist_ok=True)
for extension in ["png", "svg"]:
    fig.savefig(out / f"development_representation.{extension}", dpi=200)
print(out / "development_representation.png")
