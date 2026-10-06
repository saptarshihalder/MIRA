"""Fine-tune a model on SOURCE-period episodes of a real dataset (identical recipe for every model and dataset).

python finetune_real.py --dataset beijing --init runs/lift1_s1 --out runs_real/beijing/lift1_s1_ft
Recipe (fixed before any test score): 4,000 resampled source episodes, 2,000 AdamW steps, lr 3e-4 with cosine decay,
64 episodes x 8 queries per step; query masks = natural mask x at most one extra dropped sensor; seed 11.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import models, realdata
from train import build

RAW = {'airq_co': ('airq', 'CO(GT)'), 'airq_no2': ('airq', 'NO2(GT)'), 'beijing': ('beijing', 'PM2.5'), 'beijing_no2': ('beijing', 'NO2'),
       'beijing_co': ('beijing', 'CO'), 'beijing_pm10': ('beijing', 'PM10'), 'gas': ('gas', None)}


def load_raw(name, paths):
    kind, target = RAW[name]
    if kind == 'airq':
        return realdata.load_airq(paths['airq'], target)
    return realdata.load_beijing(paths['beijing'], target) if kind == 'beijing' else realdata.load_gas(paths['gas'])


def batch_from(pool, rng, B=64, Q=8):
    t = torch.as_tensor(rng.choice(pool['n'], B, replace=False))
    batch = {k: pool[k][t] for k in ('c', 'sx', 'sy', 'sm')}
    batch['anchors'] = {K: {k: v[t] for k, v in pool['anchors'][K].items()} for K in (0, 1, 2)}
    q = torch.as_tensor(rng.integers(0, pool['qx'].shape[1], (B, Q))); ar = torch.arange(B)[:, None]
    P = pool['qx'].shape[-1]
    nat = pool['qnat'][t][ar, q]
    drop = torch.as_tensor(rng.integers(0, P + 1, (B, Q)))
    extra = torch.ones(B, Q, P)
    ii, jj = torch.nonzero(drop < P, as_tuple=True)
    extra[ii, jj, drop[ii, jj]] = 0
    qm = nat * extra
    qm = torch.where(qm.sum(-1, keepdim=True) > 0, qm, nat)          # never drop the last observed sensor
    batch['qx'] = pool['qx'][t][ar, q].reshape(B * Q, P); batch['qy'] = pool['qy'][t][ar, q].reshape(-1)
    batch['qm'] = qm.reshape(B * Q, P); batch['tid'] = torch.arange(B).repeat_interleave(Q)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(RAW)); ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--episodes', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--period', nargs=2, default=None, help='development only: override the source period (beijing)')
    ap.add_argument('--airq', default=realdata.DEFAULT['airq'])
    ap.add_argument('--beijing', default=realdata.DEFAULT['beijing'])
    ap.add_argument('--gas', default=realdata.DEFAULT['gas'])
    a = ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((Path(a.init) / 'train.json').read_text())
    model = build(meta['model'], meta.get('repo'), meta)
    if (Path(a.init) / 'model.pt').exists():
        model.load_state_dict(torch.load(Path(a.init) / 'model.pt', weights_only=True))
    raw = load_raw(a.dataset, dict(airq=a.airq, beijing=a.beijing, gas=a.gas))
    pool = realdata.to_pool(realdata.episodes(a.dataset, raw, 'source', a.seed, n_source=a.episodes, period=a.period))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    start = time.time(); run = None
    model.train()
    for step in range(a.steps):
        batch = batch_from(pool, rng)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(meta, finetune=dict(vars(a), dataset=a.dataset, seconds=time.time() - start,
                                                                         final_ema=run, files=raw['files'])), indent=1))
    print(out.name, a.dataset, f'ema {run:.4f} {time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
