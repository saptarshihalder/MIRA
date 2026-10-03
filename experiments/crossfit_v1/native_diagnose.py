"""Post-development guard diagnosis; does not change any learned model or rule."""
import json
import numpy as np
import torch
import native_train as r

def main():
    report=json.loads((r.OUT/'report.json').read_text());rows=[];models={};torch.set_num_threads(2)
    for row in report['rows']:
        with np.load(r.OUT/row['prediction_file']) as a:
            inputs=[torch.tensor(a[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in r.x.run.v1.INPUT_KEYS];values={}
            for name in r.NAMES:
                key=f"{row['source_family']}_{row['initialization']}_{name}"
                if key not in models:
                    m=r.model(name,row['initialization'])
                    with np.load(r.OUT/(key+'_weights.npz')) as w:m.load_state_dict({k:torch.from_numpy(w[k].copy()) for k in w.files})
                    models[key]=m.eval()
                with torch.no_grad():out=models[key].predict(models[key].prepare(inputs))
                values[name]=dict(unguarded_nll=r.x.run.v1.metrics(out['unguarded'][0].numpy(),a['labels'])['nll'],hard_nll=row['metrics'][name]['nll'],accepted_branches=int(out['accepted'].sum()),branches=2)
            rows.append(dict(task=row['task'],initialization=row['initialization'],group=row['group'],frozen_nll=row['metrics']['frozen']['nll'],target_platt_nll=row['metrics']['target_platt']['nll'],models=values))
    summary={task:{name:dict(unguarded_nll=float(np.mean([v['models'][name]['unguarded_nll'] for v in rows if v['task']==task])),hard_nll=float(np.mean([v['models'][name]['hard_nll'] for v in rows if v['task']==task])),accepted_branches=sum(v['models'][name]['accepted_branches'] for v in rows if v['task']==task)) for name in r.NAMES} for task in report['means']}
    (r.OUT/'guard_diagnosis.json').write_text(json.dumps(dict(scope='Post-development diagnostic on used query outcomes; no model/threshold/label-boundary change',rows=rows,summary=summary),indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
