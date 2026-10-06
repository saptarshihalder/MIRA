"""Measure parameters, training time and single-thread inference time per task (480 queries: 10 masks x 48) on F1, k=2."""
import itertools, json, time
from pathlib import Path
import numpy as np
import torch
import data, family as fam, fa as fm, nlfa
from train import build

torch.set_num_threads(1)
panel = torch.load('cache/v2_s20261201_n256_p5_nl0.4_sr48.pt', weights_only=False)
pool, bank = panel['pool'], panel['banks'][2]
T = 32
out = {}
for name in ('lift1_s1', 'pfn_s1', 'anchor_mlp_s1'):
    meta = json.loads(Path(f'runs/{name}/train.json').read_text())
    m = build(meta['model'], meta.get('repo'), meta); m.load_state_dict(torch.load(f'runs/{name}/model.pt', weights_only=True)); m.eval()
    batches = [b for _, b in itertools.islice(data.eval_batches(pool, bank, tasks_per_batch=1), T)]
    with torch.no_grad():
        m(batches[0])
        t0 = time.perf_counter()
        for b in batches:
            m(b)
        dt = (time.perf_counter() - t0) / T
    out[name] = dict(parameters=sum(p.numel() for p in m.parameters()), train_seconds=meta.get('seconds'),
                     train_steps=meta.get('steps_done', meta.get('steps')), ms_per_task=1000 * dt)
t0 = time.perf_counter()
for t in range(T):
    sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm')); qx = pool['qx'][t].double().numpy()
    mu, sig = fam.em_gaussian(sx, sy, sm, 40); f1 = fm.fit_fa(mu, sig, 1)
    for msk in bank:
        fm.lifted_poe(f1, qx, np.flatnonzero(msk), 48)
out['fa1'] = dict(ms_per_task=1000 * (time.perf_counter() - t0) / T)
t0 = time.perf_counter()
for t in range(T):
    sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm')); qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
    m = nlfa.fit(sx, sy, sm)
    for msk in bank:
        nlfa.predict_nll(m, qx, qy, np.flatnonzero(msk))
out['nlfa'] = dict(ms_per_task=1000 * (time.perf_counter() - t0) / T)
Path('runs/compute_cost.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
