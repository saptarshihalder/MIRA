"""Frozen, CPU-only Wine development benchmark; no novelty claim."""
import hashlib, itertools, json, pickle, time, os, threading
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/reports/wine_acquisition_v1'
DATA = ROOT / 'artifacts/local/wine_acquisition'
D, K = 11, 16
SEEDS = (813001, 813002, 813003)
TEMPS = (.5, .75, 1., 1.5, 2.)
METHODS = ('masked_depth2', 'joint_depth2', 'masked_greedy', 'masked_static', 'masked_stop', 'tree_depth2', 'tree_greedy', 'tree_static')
COSTS = (np.full(D, .01), np.full(D, .03), .01 + .01 * (np.arange(D) % 3))
AVAIL = tuple(tuple(j for j in range(D) if j not in absent) for absent in ((), (0, 1), (4, 5), (8, 9)))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
def nll(y, p): return -np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.))
def encode(s): return np.concatenate((np.maximum(s, 0)/2, s >= 0), axis=1)
def temper(p, t): return softmax(np.log(np.clip(p, 1e-8, 1))/t, axis=1)

class Mixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, K)*.1)
        self.cat = nn.Parameter(torch.randn(2, K, D, 3)*.3)

    def log_class(self, s):
        # A masked marginal of one normalized joint distribution.
        logcat = self.cat.log_softmax(-1)
        logw = self.weight.flatten().log_softmax(0).reshape(2, K)
        observed = s >= 0
        safe = s.clamp_min(0).long()
        terms = torch.nn.functional.one_hot(safe, 3)*observed[:, :, None]
        score = torch.einsum('ndv,ckdv->nck', terms.float(), logcat) + logw
        return score.logsumexp(-1)

    def prob(self, s): return self.log_class(s).softmax(-1)

def fit_mix(x, y, mode, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Mixture(); opt = torch.optim.Adam(model.parameters(), lr=.01)
    tx, ty = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    trace = []
    for step in range(500):
        ix = rng.integers(0, len(x), 256)
        mask = rng.random((256, D)) < rng.random((256, 1))
        full = model.log_class(tx[ix])
        loss = -full[torch.arange(256), ty[ix]].mean()/12
        if mode == 'masked':
            s = torch.where(torch.tensor(mask), tx[ix], -1)
            loss = loss + nn.functional.cross_entropy(model.log_class(s), ty[ix])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 100 == 0: trace.append(float(loss.detach()))
    return model, trace

def states():
    result = [(-1,)*D]
    for count in (1, 2):
        for js in itertools.combinations(range(D), count):
            for vals in itertools.product(range(3), repeat=count):
                s = [-1]*D
                for j, v in zip(js, vals): s[j] = v
                result.append(tuple(s))
    return np.array(result, dtype=np.int64)

S = states(); INDEX = {tuple(s): i for i, s in enumerate(S)}
CHILD = {}
for i, s in enumerate(S):
    if (s >= 0).sum() == 2: continue
    for a in np.where(s < 0)[0]:
        ch = []
        for v in range(3):
            t = s.copy(); t[a] = v; ch.append(INDEX[tuple(t)])
        CHILD[i, a] = np.array(ch)

def beliefs(model):
    with torch.no_grad(): logc = model.log_class(torch.tensor(S)).double().numpy()
    p = softmax(logc, axis=1)
    mass = np.logaddexp(logc[:, 0], logc[:, 1])
    branch = {k: softmax(mass[v]) for k, v in CHILD.items()}
    return p, branch

def policy(q, branch, pred, avail, costs, method):
    terminal = -(pred*np.log(np.clip(pred, 1e-8, 1))).sum(1)
    value = terminal.copy(); action = np.full(len(S), -1, dtype=np.int16)
    if method == 'stop': return action, []
    if method == 'static':
        candidates = [(terminal[0], ())]
        for a in avail:
            ch = CHILD[0, a]
            candidates.append((costs[a]+branch[0, a]@terminal[ch], (a,)))
        for a, b in itertools.combinations(avail, 2):
            ch = CHILD[0, a]
            expected = sum(branch[0, a][v]*(branch[int(t), b]@terminal[CHILD[int(t), b]]) for v, t in enumerate(ch))
            candidates.append((costs[a]+costs[b]+expected, (a, b)))
        return action, list(min(candidates)[1])
    for i in range(len(S)-1, -1, -1):
        for a in avail:
            if (i, a) not in CHILD: continue
            ch = CHILD[i, a]
            target = terminal if method == 'greedy' else value
            candidate = costs[a] + branch[i, a]@target[ch]
            if candidate < value[i]: value[i], action[i] = candidate, a
    return action, []

def rollout(x, pred, action, fixed, method, avail, costs):
    # Evaluator owns x; policy receives only a state index and public constraints.
    ids = np.zeros(len(x), dtype=int)
    path = np.full((len(x), 2), -1, dtype=np.int16)
    paid = np.zeros(len(x)); done = np.zeros(len(x), dtype=bool)
    for step in range(2):
        choices = np.full(len(x), fixed[step] if step < len(fixed) else -1) if method == 'static' else action[ids]
        choices[done] = -1
        for n, a in enumerate(choices):
            if a < 0: done[n] = True; continue
            assert a in avail and S[ids[n], a] == -1
            ids[n] = CHILD[int(ids[n]), int(a)][int(x[n, a])]
            paid[n] += costs[a]; path[n, step] = a
    return pred[ids], paid, path

def calibrate(p, y):
    losses = [float(nll(y, temper(p, t)).mean()) for t in TEMPS]
    return TEMPS[int(np.argmin(losses))], losses

def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.monotonic(); OUT.mkdir(parents=True, exist_ok=False)
    thread_limiter = threadpool_limits(limits=1)
    watchdog = threading.Timer(600, lambda: os._exit(124)); watchdog.daemon = True; watchdog.start()
    frozen = json.loads((ROOT/'artifacts/manifests/wine_acquisition_freeze.json').read_text())
    for p, h in frozen.items(): assert sha(ROOT/p) == h, p
    records, training, pending = [], [], []
    for color in ('red', 'white'):
        with np.load(DATA/f'wine_{color}.npz') as z:
            x, y, split = z['X'].astype(np.int64), z['y'].astype(np.int64), z['partition']
            ids, groups = z['row_ids'], z['groups']
        fit, val, dev = split == 0, split == 1, split == 2
        assert set(np.unique(split)) == {0, 1, 2}
        assert not (set(groups[fit]) & set(groups[val]) or set(groups[fit]) & set(groups[dev]) or set(groups[val]) & set(groups[dev]))
        assert x.shape[1] == D and x.min() >= 0 and x.max() <= 2
        for seed in SEEDS:
            assert time.monotonic()-start < 600, 'Finite runtime cap reached'
            folder = OUT/f'{color}_{seed}'; folder.mkdir()
            rng = np.random.default_rng(seed+700)
            vx = np.tile(x[val], (4, 1)); vy = np.tile(y[val], 4)
            vm = rng.random(vx.shape) < rng.random((len(vx), 1)); vs = np.where(vm, vx, -1)
            models, qs, branches, predictions, temps = {}, {}, {}, {}, {}
            for mode in ('joint', 'masked'):
                t0 = time.monotonic(); model, trace = fit_mix(x[fit], y[fit], mode, seed)
                checkpoint = folder/f'{mode}.npz'
                np.savez_compressed(checkpoint, **{k:v.detach().numpy() for k,v in model.state_dict().items()})
                with torch.no_grad(): vp = model.prob(torch.tensor(vs)).numpy()
                temp, losses = calibrate(vp, vy); q, branch = beliefs(model)
                models[mode] = model; qs[mode] = q; branches[mode] = branch
                predictions[mode] = temper(q, temp); temps[mode] = temp
                training.append(dict(color=color,seed=seed,model=mode,seconds=time.monotonic()-t0,parameters=sum(p.numel() for p in model.parameters()),updates=500,trace=trace,temperature=temp,validation_nll=losses,sha256=sha(checkpoint)))
            t0 = time.monotonic(); rng = np.random.default_rng(seed+900)
            xx = np.tile(x[fit], (5, 1)); yy = np.tile(y[fit], 5)
            mm = rng.random(xx.shape) < rng.random((len(xx), 1)); mm[:fit.sum()] = True
            tree = HistGradientBoostingClassifier(max_iter=100,max_depth=3,learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=seed)
            tree.fit(encode(np.where(mm, xx, -1)), yy)
            with (folder/'tree.pkl').open('wb') as f: pickle.dump(tree, f)
            temp, losses = calibrate(tree.predict_proba(encode(vs)), vy)
            temps['tree'] = temp; predictions['tree'] = temper(tree.predict_proba(encode(S)), temp)
            training.append(dict(color=color,seed=seed,model='tree',seconds=time.monotonic()-t0,temperature=temp,validation_nll=losses,sha256=sha(folder/'tree.pkl')))
            save(folder/'temperatures.json', temps)
            np.savez_compressed(folder/'state_predictions.npz',states=S,**predictions)
            pending.append((color,seed,folder,qs,branches,predictions,x[dev],y[dev],ids[dev],groups[dev]))
    save(OUT/'fitted_before_scoring.json', {str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file()})
    for color,seed,folder,qs,branches,predictions,dx,dy,di,dg in pending:
        for ai, avail in enumerate(AVAIL):
            for ci, costs in enumerate(COSTS):
                for method in METHODS:
                    base = 'joint' if method.startswith('joint') else 'masked'
                    pred = predictions['tree' if method.startswith('tree') else base]
                    kind = method.split('_')[1]
                    action, fixed = policy(qs[base],branches[base],pred,avail,costs,kind)
                    probability, paid, path = rollout(dx,pred,action,fixed,kind,avail,costs)
                    score = nll(dy,probability)+paid
                    stem = f'a{ai}_c{ci}_{method}'
                    np.savez_compressed(folder/f'{stem}.npz',row_ids=di,groups=dg,probability=probability,paid=paid,path=path,score=score,action=action,fixed=np.array(fixed,dtype=np.int16))
                    records.append(dict(color=color,seed=seed,availability=ai,cost=ci,method=method,n=len(score),nll=float(nll(dy,probability).mean()),paid=float(paid.mean()),risk=float(score.mean())))
        assert time.monotonic()-start < 600, 'Finite runtime cap reached'
    means = {c:{m:float(np.mean([v['risk'] for v in records if v['color']==c and v['method']==m])) for m in METHODS} for c in ('red','white')}
    gates = {}
    for c in means:
        margin = min(means[c][m]-means[c]['masked_depth2'] for m in METHODS if m!='masked_depth2')
        replicated = all(np.mean([r['risk'] for r in records if r['color']==c and r['seed']==s and r['method']==m]) > np.mean([r['risk'] for r in records if r['color']==c and r['seed']==s and r['method']=='masked_depth2']) for s in SEEDS for m in METHODS if m!='masked_depth2')
        gates[c] = dict(minimum_gain=margin,positive_all_seeds=bool(replicated),passed=bool(margin>=.005 and replicated))
    report = dict(records=records,training=training,means=means,gates=gates,passed=all(v['passed'] for v in gates.values()),seconds=time.monotonic()-start,cloud_calls=0,scope='Within-domain Wine development; simulated measurement costs; established architecture, no novelty claim')
    save(OUT/'report.json',report)
    save(OUT/'manifest.json',{str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    watchdog.cancel()
    print(json.dumps({k:report[k] for k in ('means','gates','passed','seconds')}))

if __name__ == '__main__': main()
