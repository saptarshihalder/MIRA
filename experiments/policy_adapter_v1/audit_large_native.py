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


def audit(out):
    manifest=json.loads((out/'manifest.json').read_text())
    protocol=ROOT/'configs/large_native_v1.json'
    assert manifest['protocol_sha256']==checksum(protocol)
    plan=json.loads(protocol.read_text())
    report_rows=[]
    probability_arrays=0
    max_error=0.
    max_model_error=0.
    convergence=[]
    for item in manifest['reports']:
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
    audit(Path(parser.parse_args().out))
