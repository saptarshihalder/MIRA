"""Source-only predictive test of saved H0 sampling; one finite run."""
import csv,io,json,hashlib,pickle,time,threading,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import wine_acquisition_prepare as prep

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/topology_predictor_v1'
SAMPLE=ROOT/'artifacts/reports/topological_support_sampling_v1'
ARMS=('topological','uniform','farthest')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def encode(x,mask):return np.concatenate((np.where(mask,x,0),mask),axis=1).astype(np.float32)
def loss(y,p):return -np.log(np.clip(p[np.arange(len(y)),y],1e-8,1))
def fps(x,k):
    selected=[int(np.argmin(((x-x.mean(0))**2).sum(1)))];nearest=((x-x[selected[0]])**2).sum(1)
    while len(selected)<k:
        nearest[selected]=-1;j=int(np.argmax(nearest));selected.append(j);nearest=np.minimum(nearest,((x-x[j])**2).sum(1))
    return np.array(selected)
class Model(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(22,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):return self.net(x)
def fit(x,y,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);trace=[]
    for i in range(300):
        ix=rng.integers(0,len(y),128);mask=rng.random((128,11))<rng.random((128,1));z=torch.tensor(encode(x[ix],mask));value=nn.functional.cross_entropy(m(z),torch.tensor(y[ix]))
        opt.zero_grad();value.backward();opt.step()
        if (i+1)%100==0:trace.append(float(value.detach()))
    return m,trace
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);lim=threadpool_limits(limits=1)
    frozen=json.loads((ROOT/'artifacts/manifests/topology_predictor_freeze.json').read_text())
    for f,h in frozen.items():assert sha(ROOT/f)==h,f
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();timer=threading.Timer(180,lambda:os._exit(124));timer.daemon=True;timer.start()
    pending=[];training=[]
    for color in ('red','white'):
        raw=ROOT/f'data/raw/wine_acquisition/winequality-{color}.csv';rows=list(csv.reader(io.StringIO(raw.read_text()),delimiter=';'))[1:]
        x=np.array([[float(v) for v in row[:-1]] for row in rows]);groups=np.array([prep.sha256(prep.canonical_features(v)) for v in x]);isfit=np.array([prep.partition_for(g)==0 for g in groups])
        for seed in (816001,816002,816003):
            z=np.load(SAMPLE/f'{color}_{seed}.npz');pool=z['pool_source_lines']-2;scale=z['scale'];median=z['median'];assert isfit[pool].all()
            features=np.clip((x-median)/scale,-5,5);px=features[pool];validation=np.where(isfit & ~np.isin(groups,groups[pool]))[0];assert not set(groups[validation])&set(groups[pool])
            choices={'topological':z['topological_indices'],'uniform':z['uniform_indices'],'farthest':fps(px,64)}
            folder=OUT/f'{color}_{seed}';folder.mkdir();models={}
            for arm in ARMS:
                chosen=pool[choices[arm]];y=np.array([int(float(rows[j][-1])>=6) for j in chosen],dtype=np.int64)
                model,trace=fit(features[chosen],y,817001+seed-816001);models[arm]=model
                np.savez_compressed(folder/f'{arm}_model.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
                np.savez_compressed(folder/f'{arm}_selection.npz',source_lines=chosen+2,groups=groups[chosen])
                training.append(dict(color=color,seed=seed,arm=arm,updates=300,parameters=sum(p.numel() for p in model.parameters()),trace=trace))
            chosen=pool[choices['topological']];y=np.array([int(float(rows[j][-1])>=6) for j in chosen]);rng=np.random.default_rng(817001+seed-816001);xx=np.tile(features[chosen],(5,1));yy=np.tile(y,5);mask=rng.random(xx.shape)<rng.random((len(xx),1));mask[:64]=True
            tree=HistGradientBoostingClassifier(max_iter=100,max_depth=3,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=seed);tree.fit(encode(xx,mask),yy)
            with (folder/'topological_tree.pkl').open('wb') as f:pickle.dump(tree,f)
            rng=np.random.default_rng(818000+seed);masks=np.zeros((3,len(validation),11),dtype=bool)
            for n in range(len(validation)):
                order=rng.permutation(11);masks[1,n,order[:1]]=True;masks[2,n,order[:2]]=True
            np.savez_compressed(folder/'validation_inputs.npz',source_lines=validation+2,groups=groups[validation],features=features[validation],masks=masks)
            pending.append((color,seed,folder,models,tree,validation,features[validation],masks,rows))
    save(OUT/'fitted_before_scoring.json',{p.as_posix().split('topology_predictor_v1/')[1]:sha(p) for p in OUT.rglob('*') if p.is_file()})
    records=[]
    for color,seed,folder,models,tree,ids,x,masks,rows in pending:
        y=np.array([int(float(rows[j][-1])>=6) for j in ids]);yy=np.tile(y,3);inputs=encode(np.tile(x,(3,1)),masks.reshape(-1,11))
        for arm in (*ARMS,'topological_tree'):
            if arm=='topological_tree':probs=tree.predict_proba(inputs)
            else:
                with torch.no_grad():probs=models[arm](torch.tensor(inputs)).softmax(-1).numpy()
            score=loss(yy,probs);np.savez_compressed(folder/f'{arm}_scores.npz',probability=probs,score=score)
            records.append(dict(color=color,seed=seed,arm=arm,rows=len(y),nll=float(score.mean())))
    means={c:{arm:float(np.mean([r['nll'] for r in records if r['color']==c and r['arm']==arm])) for arm in (*ARMS,'topological_tree')} for c in ('red','white')}
    gates={c:all(means[c][arm]-means[c]['topological']>=.01 for arm in ('uniform','farthest')) and all(next(r['nll'] for r in records if r['color']==c and r['seed']==seed and r['arm']==arm)>next(r['nll'] for r in records if r['color']==c and r['seed']==seed and r['arm']=='topological') for seed in (816001,816002,816003) for arm in ('uniform','farthest')) for c in means}
    report=dict(records=records,training=training,means=means,gates=gates,passed=all(gates.values()),seconds=time.monotonic()-start,cloud_calls=0,scope='Source-only Wine validation; selection ablation, no architecture novelty or fresh-confirmation claim')
    save(OUT/'report.json',report);save(OUT/'manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'});timer.cancel();lim.restore_original_limits();print(json.dumps({k:report[k] for k in ('means','gates','passed','seconds')}))
if __name__=='__main__':main()
