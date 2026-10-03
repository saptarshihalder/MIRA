"""Independent saved-probability, NumPy checkpoint and partition audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax

ROOT=Path(__file__).resolve().parents[2]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nll(probability,labels):
    probability=np.clip(np.asarray(probability,dtype=float),1e-7,1-1e-7)
    return float(np.mean(-labels*np.log(probability)-(1-labels)*np.log1p(-probability)))


def identities(array):
    return set(map(tuple,np.asarray(array).reshape(-1,2).tolist()))


def model_predict(state,context,experts,linear):
    if linear:
        score=context@state['score.weight'].T+state['score.bias']
    else:
        value=context
        for index in (0,2):
            value=value@state[f'score.{index}.weight'].T+state[f'score.{index}.bias']
            value=value*expit(value)
        score=value@state['score.4.weight'].T+state['score.4.bias']
    weight=softmax(score[:,0]+context[:,0]*np.log(15.))
    return experts@weight


def audit(out,protocol):
    manifest=json.loads((out/'manifest.json').read_text())
    protocol=Path(protocol)
    assert manifest['protocol_sha256']==checksum(protocol)
    plan=json.loads(protocol.read_text())
    report_rows=[]
    probability_arrays=0
    max_error=0.
    max_model_error=0.
    convergence=[]
    source_cache={}
    for item in manifest['reports']:
        spec=next(task for task in plan['tasks'] if task['name']==item['task'])
        if item['task'] not in source_cache:
            source_path=ROOT/'artifacts/runs/large_native_data'/spec['npz']
            assert checksum(source_path)==spec['npz_sha256']
            with np.load(source_path) as source:
                ids=(source['year'].astype(np.int64)<<32)+source['row_id']
                order=np.argsort(ids)
                source_cache[item['task']]=(ids[order],source['groups'][order],source['y'][order])
        source_ids,source_groups,source_labels=source_cache[item['task']]
        def lookup(ids):
            ids=np.asarray(ids).reshape(-1,2)
            keys=(ids[:,0].astype(np.int64)<<32)+ids[:,1]
            positions=np.searchsorted(source_ids,keys)
            assert np.all(positions<len(source_ids)) and np.array_equal(source_ids[positions],keys)
            return positions
        directory=out/f"{item['task']}_seed{item['seed']}"
        report=json.loads((directory/'report.json').read_text())
        assert report['gpu'].startswith('NVIDIA A100') and report['source_backbone_device'].startswith('cuda')
        for filename,key in [('source_backbone.json','source_backbone_sha256'),
                             ('boundaries_and_meta.npz','boundary_sha256'),('calibration_weights.npz','calibration_sha256')]:
            assert checksum(directory/filename)==report[key]
        with np.load(directory/'boundaries_and_meta.npz') as boundary:
            used=set()
            for key in ('source_fit_ids','source_context_ids','meta_support_ids','meta_query_ids'):
                ids=identities(boundary[key])
                assert len(ids)==boundary[key].reshape(-1,2).shape[0]
                assert not used.intersection(ids)
                assert all(year==2024 for year,_ in ids)
                assert np.all(source_groups[lookup(boundary[key])]%5<3)
                used.update(ids)
        states={}
        for name,trace in report['training_models'].items():
            assert trace['parameter_device']=='cuda:0'
            path=directory/(name+'_weights.npz')
            assert checksum(path)==trace['checkpoint_sha256']
            with np.load(path) as state:
                states[name]={key:state[key].copy() for key in state.files}
        conv=report['training_models']['native_linear_converged']['optimization']
        convergence.append(dict(task=item['task'],seed=item['seed'],linear=conv,
                                global_calibration=report['global_calibration']))
        scores={(row['split'],row['group']):row for row in report['results']}
        for path in sorted((directory/'predictions').glob('*.npz')):
            split,group=path.stem.split('_group')
            group=int(group)
            assert group%5==(3 if split=='validation' else 4)
            with np.load(path) as data:
                support,query=identities(data['support_ids']),identities(data['query_ids'])
                assert not support.intersection(query) and not used.intersection(support|query)
                assert all(year==2025 for year,_ in support|query)
                used.update(support|query)
                labels=data['labels'].astype(float)
                assert np.all(source_groups[lookup(data['support_ids'])]==group)
                assert np.all(source_groups[lookup(data['query_ids'])]==group)
                assert np.array_equal(source_labels[lookup(data['query_ids'])],labels)
                assert np.array_equal(source_labels[lookup(data['support_ids'])],data['support_y'])
                for name,expected in scores[(split,group)]['nll'].items():
                    probability=data[name]
                    assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
                    score=nll(probability,labels)
                    error=abs(score-expected)
                    max_error=max(error,max_error)
                    assert error<1e-6
                    probability_arrays+=1
                for name,state in states.items():
                    calculated=model_predict(state,data['context'].astype(float),data['experts'].astype(float),name!='native_scratch')
                    error=float(np.max(np.abs(calculated-data[name])))
                    max_model_error=max(error,max_model_error)
                    assert error<2e-6
                if split=='development':
                    report_rows.append(dict(task=item['task'],seed=item['seed'],group=group,
                                            queries=len(labels),nll={name:nll(data[name],labels) for name in scores[(split,group)]['nll']}))
    summary={}
    for task in sorted({row['task'] for row in report_rows}):
        rows=[row for row in report_rows if row['task']==task]
        summary[task]={name:float(np.mean([row['nll'][name] for row in rows])) for name in rows[0]['nll']}
    result=dict(passed=True,trials=len(manifest['reports']),probability_arrays=probability_arrays,
                max_nll_error=max_error,max_numpy_checkpoint_error=max_model_error,development_group_mean_nll=summary,
                development_rows=report_rows,convergence=convergence,
                scope='Conditional group means; seeds share data. Two domains, correlated health tasks. No confirmation or independent nine-dataset claim.')
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('development_rows','convergence')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/large_native_v2.json'))
    args=parser.parse_args()
    audit(Path(args.out),args.config)
