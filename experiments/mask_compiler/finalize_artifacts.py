"""Save provenance and a development figure from audited results."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/"artifacts/reports/trained_compiler_backbone_v1"
report=json.loads((out/"report.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
for i,g in enumerate(report["gates"]):
    values=g["seed_units"]["0.9"]
    axes[0].bar(i,np.mean(values),color=["#356b94","#98703e"][i],width=.55,alpha=.7)
    axes[0].scatter(np.array([-.1,0,.1])+i,values,color="black",s=18,zorder=3)
axes[0].set(xticks=[0,1],xticklabels=["TabPFN v2 (CPU)","TabICL v2 (T4)"],ylabel="Identity NLL minus compiler NLL",title="Development: three seeds, orders averaged")
axes[0].axhline(0,color="gray",linewidth=.8)
for name,color in [("tabpfn_cpu","#356b94"),("tabicl_gpu","#98703e")]:
    rows=json.loads((out/name/"results.json").read_text())
    table={(r["episode_id"],r["mode"]):r for r in rows if r["kind"]=="synthetic"}
    ids=sorted({e for e,m in table})
    x=[table[e,"heuristic"]["expected_nll"] for e in ids];y=[table[e,"trained"]["expected_nll"] for e in ids]
    axes[1].scatter(x,y,color=color,s=24,alpha=.75,label=name.split("_")[0])
axes[1].plot([.18,.68],[.18,.68],color="gray",linewidth=.8)
axes[1].set(xlabel="Fixed heuristic expected NLL",ylabel="Trained compiler expected NLL",title="All synthetic predictions match the heuristic")
axes[1].legend(frameon=False,fontsize=8)
for suffix in ("png","svg"):fig.savefig(out/("compiler_development."+suffix),dpi=180)
plt.close(fig)
paper=ROOT/"paper/main.tex";source=paper.read_text()
begins=re.findall(r"\\begin\{([^}]+)\}",source);ends=re.findall(r"\\end\{([^}]+)\}",source)
stack=[]
for match in re.finditer(r"\\(begin|end)\{([^}]+)\}",source):
    kind,name=match.groups()
    if kind=="begin":stack.append(name)
    elif not stack or stack.pop()!=name:raise ValueError("TeX environment imbalance")
if stack:raise ValueError("Unclosed TeX environment")
keys=set(re.findall(r"\\bibitem\{([^}]+)\}",source))
citations={k for block in re.findall(r"\\cite\{([^}]+)\}",source) for k in block.split(",")}
labels=set(re.findall(r"\\label\{([^}]+)\}",source));refs=set(re.findall(r"\\ref\{([^}]+)\}",source))
if citations-keys or refs-labels:raise ValueError("Unresolved source citation/reference")
for name in ("trained_compiler_validation_v1","trained_compiler_validation_v2","compiler_gpu_repair_v1"):
    p=ROOT/"configs"/(name+".json");plan=json.loads(p.read_text())
    if p.with_suffix('.sha256').read_text().strip()!=hashlib.sha256(p.read_bytes()).hexdigest():raise ValueError("Protocol digest changed")
    for rel,h in plan["file_sha256"].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError("Frozen file changed: "+rel)
record={"path":str(paper),"source_sha256":hashlib.sha256(paper.read_bytes()).hexdigest(),"native_compiler_status":"compile-failed",
        "diagnostic":"Unable to find standard directories for platform","pdf_verified":False,"source_checks":{"environments_balanced":True,"citations_resolved":True,"references_resolved":True},
        "frozen_protocols_rechecked":3,"limitation":"Source checks do not establish successful compilation or PDF layout. Same native document remains open."}
(ROOT/"artifacts/manifests/paper_compile.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({"figure":"compiler_development.png","frozen_protocols":3,"source_checks":"pass","pdf_verified":False}))
