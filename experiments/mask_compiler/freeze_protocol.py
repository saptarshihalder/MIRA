"""Materialize the bounded development protocol before paid inference."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    target=ROOT/"configs/trained_compiler_validation_v1.json"
    if target.exists(): raise ValueError("Preserve the existing frozen protocol")
    plan={"frozen":True,"protocol_version":"trained_compiler_validation_v1","frozen_local_date":"2026-10-02",
        "phase":"development; no confirmation or venue-readiness claim",
        "models":["tabpfn:v2","tabicl:v2"],"ensembles":4,
        "synthetic":{"orders":[2,4],"gammas":[0,.9],"seeds":[74000,74001,74002],"context":128,"queries":256},
        "real":[{"directory":"artifacts/data/natural_compiler/"+d,"folds":[0,1,2]} for d in ("hepatitis","horse_colic")],
        "views":["native","identity","trained","heuristic","matched_linear","cv_basis","random_basis","deepsets","shuffled"],
        "compiler_checkpoint":"artifacts/reports/matched_compiler_v1/prototype_selector.npz",
        "linear_checkpoint":"artifacts/reports/matched_compiler_v1/linear_selector.npz",
        "deepsets_checkpoint":"artifacts/reports/matched_compiler_v1/deepsets_selector.npz",
        "decision_gate":{"primary_comparison":"identity minus trained expected NLL; average orders within each of three seed units",
            "minimum_high_signal_gain":.01,"maximum_null_harm":.01,
            "advance_requires":"Numerical screen plus substantive evidence beyond fixed heuristic, equally informed learned selectors and support-CV controls. No fresh confirmation if simple controls explain the result.",
            "natural_scope":"Two fixed datasets; query-weighted folds within dataset; descriptive only. Native NaNs retained; support-only choice of four mask columns; zero U proxy outside pretraining distribution; categorical codes treated numerically.",
            "known_before_freeze":"512 CPU development tasks already show prototype/heuristic equality on all256 informative tasks. These GPU seeds are development, not untouched confirmation."},
        "budgets":{"paid_call_reservation_usd":.5,"max_calls_including_failures":2,"initial_total_usd":1,"project_cap_usd":26,"reproduction_reserve_usd":3,"automatic_retries":0},
        "checkpoint_sha256":{
            "tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt":"cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49",
            "huggingface/hub/blobs/45/45c13f180e7197fbc11b24a9cf5e3be795902967bcc61e2223dc19245559a934":"bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"}}
    files=[p for p in (ROOT/"src/mira").glob("*.py")]
    files += [ROOT/p for p in ("infra/modal_mechanism.py","infra/modal_compiler.py","pyproject.toml")]
    files += [p for p in (ROOT/"experiments/mask_compiler").glob("*.py")]
    files += [ROOT/plan[k] for k in ("compiler_checkpoint","linear_checkpoint","deepsets_checkpoint")]
    files += [p for r in plan["real"] for p in (ROOT/r["directory"]).rglob("*") if p.is_file()]
    plan["file_sha256"]={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    target.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    target.with_suffix(".sha256").write_text(sha(target)+"\n",encoding="utf-8")
    print(json.dumps({"protocol_sha256":sha(target),"frozen_files":len(files),"expected_cells":324}))
if __name__=="__main__":main()
