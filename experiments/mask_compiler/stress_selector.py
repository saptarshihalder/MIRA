"""Predeclared CPU null-signal/OOD diagnostic; not a downstream efficacy test."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from train_selector import ROOT, Compiler, encode
from mira.data import Support, Observations

out=ROOT / "artifacts/reports/compiler_null_stress_v1"
if out.exists(): raise ValueError("Preserve previous diagnostic")
out.mkdir(parents=True)
torch.set_num_threads(1)
model=Compiler()
path=ROOT / "artifacts/reports/trained_compiler_v1/trained_parameters.npz"
with np.load(path,allow_pickle=False) as data:
    model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
model.eval()
rows=[]
for prevalence in (.2,.5,.8):
    for rate in (.1,.5,.9):
        for policy in ("independent","shared_block"):
            xs=[]
            for seed in range(96000,96050):
                rng=np.random.default_rng([seed, int(prevalence*10),int(rate*10),int(policy=="shared_block"), 778])
                y=(rng.random(128)<prevalence).astype(int)
                mask=(rng.random((128,4) if policy=="independent" else (128,1))<rate).astype(np.uint8)
                if policy=="shared_block": mask=np.repeat(mask,4,axis=1)
                x=rng.normal(size=mask.shape); x[mask.astype(bool)]=np.nan
                xs.append(encode(Support(Observations(np.zeros(128),x,mask),y)))
            with torch.no_grad(): p=model(torch.from_numpy(np.array(xs))).softmax(-1).numpy()
            raw=p.argmax(1); selected=np.where(p.max(1)>=.8,raw,0)
            rows.append({"prevalence":prevalence,"missing_rate":rate,"policy":policy,"tasks":50,
                "raw_nonidentity_fraction":float(np.mean(raw!=0)),"guarded_nonidentity_fraction":float(np.mean(selected!=0)),
                "mean_max_probability":float(p.max(1).mean())})
(out / "report.json").write_text(json.dumps({"status":"complete","paid_usd":0,"tasks":900,"used_development_seeds":[96000,96049],
    "rng_namespace":778,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    "conditions":rows,"interpretation":"Masks generated independently of labels, including correlated acquisition blocks. Nonidentity selection is not a downstream harm estimate; confidence is not a safety guarantee. U proxyzero is explicitlyOOD."},indent=2)+"\n")
lines=["# Null-signal label-prevalence and mask-policy stress", "", "900 CPU-only exploratory tasks; query predictive harm not measured.","", "| Prevalence | Rate | Policy | Guarded nonidentity fraction |", "|---|---|---|---:|"]
for r in rows: lines.append(f"| {r['prevalence']} | {r['missing_rate']} | {r['policy']} | {r['guarded_nonidentity_fraction']:.2f} |")
(out / "report.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"tasks":900,"maximum_guarded_nonidentity_fraction":max(r['guarded_nonidentity_fraction'] for r in rows),"paid_usd":0}))
