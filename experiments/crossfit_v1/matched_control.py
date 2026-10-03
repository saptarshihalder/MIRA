"""One capacity-matched contextual-scalar control; CPU only, fixed500updates."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import run

ROOT=run.ROOT

def equalize_modes(module,args):
    context=args[0]
    return (context.mean(1,keepdim=True).expand_as(context),)

def main():
    torch.set_num_threads(2);started=time.monotonic()
    original=ROOT/'artifacts/reports/crossfit_v1_gpu';report=json.loads((original/'report.json').read_text());plan=report['plan']
    out=ROOT/'artifacts/reports/crossfit_matched_control';out.mkdir(parents=True,exist_ok=False)
    model=run.make_model('query',plan,'cpu');model.network.register_forward_pre_hook(equalize_modes)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==370
    episodes=[e for seed in plan['train_seeds'] for width in plan['train_widths'] for e in run.v1.world(seed,width,plan,training=True)]
    cache={}
    for width in plan['train_widths']:
        rows=[e for e in episodes if e['width']==width]
        with torch.no_grad():parts=model.prepare(run.v1.tensor_input(rows,'cpu'))
        cache[width]=(parts,torch.tensor(np.stack([e['query_y'] for e in rows]),dtype=torch.float64))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate']);rng=np.random.default_rng(plan['training_seed']);trace=[]
    for update in range(500):
        width=plan['train_widths'][int(rng.integers(len(plan['train_widths'])))];parts,y=cache[width]
        index=torch.tensor(rng.choice(len(y),plan['batch_size'],replace=False))
        result=model.predict([{k:value[index] for k,value in p.items()} for p in parts])
        loss=sum(torch.nn.functional.binary_cross_entropy(result[k].clamp(1e-8,1-1e-8),y[index]) for k in ('soft','unguarded'))/2
        optimizer.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        if update%50==0 or update==499:trace.append(dict(update=update+1,loss=float(loss.detach())))
    checkpoint=out/'weights.npz';np.savez_compressed(checkpoint,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    (out/'training.json').write_text(json.dumps(dict(updates=500,parameters=370,device='cpu',trace=trace,checkpoint_sha256=run.v1.checksum(checkpoint)),indent=2)+'\n')
    rows=[];(out/'predictions').mkdir();model.eval()
    restored=run.make_model('query',plan,'cpu');restored.network.register_forward_pre_hook(equalize_modes)
    with np.load(checkpoint) as state:restored.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
    replay_error=0.
    for row in report['results']:
        seed,width,regime=(row[k] for k in ('seed','width','regime'));file=f'{seed}_{width}_{regime}.npz'
        with np.load(original/'predictions'/file) as saved:
            inputs=[torch.tensor(saved[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in run.v1.INPUT_KEYS]
            with torch.no_grad():result=model.predict(model.prepare(inputs));replay=restored.predict(restored.prepare(inputs))
            prediction=result['hard'][0].numpy();replay_error=max(replay_error,float(np.abs(prediction-replay['hard'][0].numpy()).max()))
            np.savez_compressed(out/'predictions'/file,probability=prediction,labels=saved['labels'])
            metric=run.v1.metrics(prediction,saved['labels'])
            rows.append(dict(seed=seed,width=width,regime=regime,nll=metric['nll'],main_nll=row['metrics']['query']['nll'],frozen_nll=row['metrics']['frozen']['nll'],accepted=result['accepted'][:,0].tolist()))
    contrasts={}
    for regime in run.v1.REGIMES:
        effect=np.array([np.mean([r['nll']-r['main_nll'] for r in rows if r['seed']==seed and r['regime']==regime]) for seed in plan['development_seeds']])
        radius=t.ppf(.975,len(effect)-1)*effect.std(ddof=1)/len(effect)**.5
        contrasts[regime]=dict(main_gain=float(effect.mean()),lower95=float(effect.mean()-radius),upper95=float(effect.mean()+radius),control_nll=float(np.mean([r['nll'] for r in rows if r['regime']==regime])))
    assert replay_error==0
    result=dict(scope='Fixed capacity-matched control added after initial development; same used panel, no new confirmation or selection',parameters=370,updates=500,training_protocol_sha256=report['protocol_sha256'],source_sha256=run.v1.checksum(__file__),checkpoint_sha256=run.v1.checksum(checkpoint),max_reload_error=replay_error,device='cpu',seconds=time.monotonic()-started,rows=rows,contrasts=contrasts)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(contrasts=contrasts,seconds=result['seconds'],max_reload_error=replay_error)))

if __name__=='__main__':main()
