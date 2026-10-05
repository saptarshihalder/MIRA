"""Portable saved-model/source-boundary replay; no fits or new scientific cells."""
import csv,io,json,pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import topology_predictor_gate as p

def main():
    torch.set_num_threads(1);threadpool_limits(limits=1)
    for mf,base in ((p.ROOT/'artifacts/manifests/topology_predictor_freeze.json',p.ROOT),(p.OUT/'fitted_before_scoring.json',p.OUT),(p.OUT/'manifest.json',p.OUT)):
        for name,h in json.loads(mf.read_text()).items():assert p.sha(base/name)==h
    report=json.loads((p.OUT/'report.json').read_text());cells=0;error=0.
    for color in ('red','white'):
        rows=list(csv.reader(io.StringIO((p.ROOT/f'data/raw/wine_acquisition/winequality-{color}.csv').read_text()),delimiter=';'))[1:]
        raw=np.array([[float(v) for v in row[:-1]] for row in rows])
        for seed in (816001,816002,816003):
            folder=p.OUT/f'{color}_{seed}';s=np.load(p.SAMPLE/f'{color}_{seed}.npz');z=np.load(folder/'validation_inputs.npz');idx=z['source_lines']-2;pool=s['pool_source_lines']-2
            groups=np.array([p.prep.sha256(p.prep.canonical_features(raw[j])) for j in idx]);assert np.array_equal(groups,z['groups']) and all(p.prep.partition_for(g)==0 for g in groups)
            poolgroups={p.prep.sha256(p.prep.canonical_features(raw[j])) for j in pool};assert not poolgroups&set(groups)
            x=np.clip((raw[idx]-s['median'])/s['scale'],-5,5);assert np.array_equal(x,z['features'])
            choices={'topological':s['topological_indices'],'uniform':s['uniform_indices'],'farthest':p.fps(np.clip((raw[pool]-s['median'])/s['scale'],-5,5),64)}
            y=np.tile(np.array([int(float(rows[j][-1])>=6) for j in idx]),3);inputs=p.encode(np.tile(x,(3,1)),z['masks'].reshape(-1,11))
            for arm in (*p.ARMS,'topological_tree'):
                if arm=='topological_tree':
                    with (folder/'topological_tree.pkl').open('rb') as f:model=pickle.load(f)
                    probability=model.predict_proba(inputs)
                else:
                    selected=np.load(folder/f'{arm}_selection.npz');assert np.array_equal(selected['source_lines'],pool[choices[arm]]+2) and len(set(selected['source_lines']))==64
                    weights=np.load(folder/f'{arm}_model.npz');model=p.Model();model.load_state_dict({k:torch.tensor(weights[k]) for k in weights.files})
                    with torch.no_grad():probability=model(torch.tensor(inputs)).softmax(-1).numpy()
                score=p.loss(y,probability);saved=np.load(folder/f'{arm}_scores.npz');error=max(error,float(abs(saved['probability']-probability).max()),float(abs(saved['score']-score).max()))
                row=next(v for v in report['records'] if v['color']==color and v['seed']==seed and v['arm']==arm);assert abs(score.mean()-row['nll'])<1e-6;cells+=1
    assert cells==24 and error<1e-6
    for color,m in report['means'].items():
        gate=all(m[a]-m['topological']>=.01 for a in ('uniform','farthest')) and all(next(v['nll'] for v in report['records'] if v['color']==color and v['seed']==seed and v['arm']==a)>next(v['nll'] for v in report['records'] if v['color']==color and v['seed']==seed and v['arm']=='topological') for seed in (816001,816002,816003) for a in ('uniform','farthest'));assert gate==report['gates'][color]
    result=dict(passed=True,models=24,score_cells=cells,max_replay_error=error,scientific_gate=report['passed'],auditor_sha256=p.sha(p.ROOT/'experiments/topology_predictor_replay.py'))
    p.save(p.OUT/'audit.json',result);p.save(p.OUT/'manifest.json',{f.relative_to(p.OUT).as_posix():p.sha(f) for f in p.OUT.rglob('*') if f.is_file() and f.name!='manifest.json'});print(result)
if __name__=='__main__':main()
