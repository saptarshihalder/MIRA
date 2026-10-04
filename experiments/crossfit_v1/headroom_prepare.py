"""Prepare audited patient subsets with original source-only categorical encoding."""
import sys,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/policy_adapter_v1'))
import native as n
OUT=ROOT/'artifacts/reports/unscored_headroom_inputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    raw=ROOT/'artifacts/runs/diabetes_native_raw/diabetic_data.csv'
    assert sha(raw)=='0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97'
    old=json.loads((ROOT/'artifacts/reports/policy_native_gpu_v1/report.json').read_text())
    rows,columns=n.load(raw);assert len(rows)==69990
    fit=np.array(old['source_fit_ids']);x,category=n.encode(rows,columns,fit)
    ids=np.array([r['patient_nbr'] for r in rows]);y=np.array([int(r['readmitted']=='<30') for r in rows])
    candidate=np.arange(47114,55992);digest=hashlib.sha256(('\n'.join(ids[candidate])+'\n').encode()).hexdigest()
    assert digest=='ea169ed49d468f09ceb6a89411559657c1812cf6cba1bacf5b838fd866fcdd95'
    used=set(old['source_fit_ids']+old['source_context_ids']+old['native_training_ids']+old['native_validation_ids'])
    for p in (ROOT/'artifacts/reports/policy_native_gpu_v1/predictions').glob('*.npz'):
        with np.load(p) as a:
            for k in ('support_ids','query_ids'):
                if k in a.files:used.update(a[k].tolist())
    assert not set(candidate)&used and len(set(ids[candidate]))==8878
    OUT.mkdir(parents=True,exist_ok=False)
    for episode in range(5):
        index=np.arange(47114+episode*1536,47114+(episode+1)*1536)
        np.savez_compressed(OUT/f'episode{episode}.npz',x=x[index].astype(np.float32),y=y[index],patient_ids=ids[index],representative_indices=index,categorical=np.array(category),source_x=x[fit].astype(np.float32))
    (OUT/'report.json').write_text(json.dumps(dict(patient_count=8878,scored_episode_patients=7680,remaining_unscored=1198,candidate_patient_digest=digest,raw_csv_sha256=sha(raw),columns=columns,source_only_encoding=True,scope='Previously unscored recorded modeling cohort; full-data/label preprocessing previously exposed; not confirmation'),indent=2)+'\n')
    print('Prepared5patient-disjoint episodes; no predictions scored.')

if __name__=='__main__':main()
