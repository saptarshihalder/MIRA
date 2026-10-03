import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    plan=json.loads((ROOT/'configs/spectral_v1.json').read_text());plan.pop('file_sha256',None)
    plan.update(version='crossfit_v1',run_id='crossfit_v1_gpu',train_seeds=list(range(640001,640017)),validation_seeds=[645001,645002],development_seeds=list(range(650001,650021)),training_seed=660001,gate='Both null upper95 harm<=.001; sign-flip gain>.003 with lower95>0 over frozen; neural benefit over guarded logistic/no-query/scalar required separately; no automatic native confirmation',guard_threshold=1.645,soft_guard_scale=.005,loss_weights=dict(soft_guarded=.5,unguarded=.5),scientific_status='fresh adaptive development; no confirmation')
    (ROOT/'configs/crossfit_v1.json').write_text(json.dumps(plan,indent=2)+'\n',newline='\n')
    for old,new in [('experiments/spectral_v1/freeze.py','experiments/crossfit_v1/freeze.py'),('experiments/spectral_v1/launch.py','experiments/crossfit_v1/launch.py'),('infra/modal_spectral_v1.py','infra/modal_crossfit_v1.py')]:
        text=(ROOT/old).read_text().replace('spectral_v1','crossfit_v1').replace('SPECTRAL_PROTOCOL','CROSSFIT_PROTOCOL').replace('mira-spectral-v1','mira-crossfit-v1').replace('train_spectral','train_crossfit').replace('recursive_jepa_engineering','crossfit_development')
        if new.endswith('freeze.py'):text=text.replace("'experiments/bridge_v1/run.py'","'experiments/spectral_v1/model.py', 'experiments/bridge_v1/run.py'")
        if new.startswith('infra/'):
            text=text.replace("    run_training = app.function", "    image = image.add_local_file(ROOT / 'experiments/spectral_v1/model.py', '/opt/mira/experiments/spectral_v1/model.py')\n    run_training = app.function")
        (ROOT/new).write_text(text,newline='\n')
if __name__=='__main__':main()
