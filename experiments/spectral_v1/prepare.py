"""Prepare one successor protocol from already audited bounded infrastructure."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    plan=json.loads((ROOT/'configs/recursive_jepa_v1.json').read_text())
    plan.pop('file_sha256',None)
    for key in ('jepa_weight','variance_weight','covariance_weight','deep_supervision_weight','teacher_decay'):plan.pop(key,None)
    plan.update(version='spectral_v1',run_id='spectral_v1_gpu',scientific_status='bounded fresh development, no confirmation',train_seeds=list(range(540001,540017)),validation_seeds=[545001,545002],development_seeds=list(range(550001,550009)),train_widths=[6,10],development_widths=[6,10],training_seed=560001,updates=500,steps=6,batch_size=8,learning_rate=.003,gate='Require >=.003 NLL improvement over both full/global controls with positive paired seed interval on each shifted regime; each null upper95 harm<=.001; pilot remains unconfirmed')
    (ROOT/'configs/spectral_v1.json').write_text(json.dumps(plan,indent=2)+'\n',newline='\n')
    for old,new in [('experiments/recursive_jepa_v1/freeze.py','experiments/spectral_v1/freeze.py'),('experiments/recursive_jepa_v1/launch.py','experiments/spectral_v1/launch.py'),('infra/modal_recursive_jepa_v1.py','infra/modal_spectral_v1.py')]:
        s=(ROOT/old).read_text().replace('recursive_jepa_v1','spectral_v1').replace('RECURSIVE_JEPA_PROTOCOL','SPECTRAL_PROTOCOL').replace('mira-recursive-jepa-v1','mira-spectral-v1').replace('train_recursive_jepa','train_spectral').replace("plan['updates'] != 64","plan['updates'] != 500")
        (ROOT/new).write_text(s,newline='\n')
if __name__=='__main__':main()
