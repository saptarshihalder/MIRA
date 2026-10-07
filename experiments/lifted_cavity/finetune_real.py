"""Fine-tune a model on SOURCE-period episodes of a real dataset (identical recipe for every model and dataset).

python finetune_real.py --dataset beijing --init runs/lift1_s1 --out runs_real/beijing/lift1_s1_ft
Recipe (fixed before any test score): 4,000 resampled source episodes, 2,000 AdamW steps, lr 3e-4 with cosine decay,
64 episodes x 8 queries per step; query masks = natural mask x at most one extra dropped sensor; seed 11.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import devutil, models, realdata, resume
from train import build

RAW = {'airq_co': ('airq', 'CO(GT)'), 'airq_no2': ('airq', 'NO2(GT)'), 'beijing': ('beijing', 'PM2.5'), 'beijing_no2': ('beijing', 'NO2'),
       'beijing_co': ('beijing', 'CO'), 'beijing_pm10': ('beijing', 'PM10'), 'beijing_so2': ('beijing', 'SO2'),
       'beijing_o3': ('beijing', 'O3'), 'gas': ('gas', None)}
DEQUANT = {'beijing_pm10', 'beijing_so2', 'beijing_o3'}     # protocol v4, amendment 1: integer readings dequantized


def load_raw(name, paths):
    kind, target = RAW[name]
    if kind == 'airq':
        return realdata.load_airq(paths['airq'], target)
    if kind == 'beijing':
        return realdata.load_beijing(paths['beijing'], target, dequant=name in DEQUANT)
    return realdata.load_gas(paths['gas'])


def source_pool(a):
    """Source-period episodes and their anchors; optionally cached, since every model fine-tunes on the same pool."""
    key = f'src_{a.dataset}_s{a.seed}_n{a.episodes}' + (f'_p{a.period[0]}_{a.period[1]}' if a.period else '')
    path = Path(a.pool_cache) / f'{key}.pt' if a.pool_cache else None
    if path is not None and path.exists():
        d = torch.load(path, weights_only=False)
        return d['pool'], d['files'], d['dequant_seed']
    raw = load_raw(a.dataset, dict(airq=a.airq, beijing=a.beijing, gas=a.gas))
    pool = realdata.to_pool(realdata.episodes(a.dataset, raw, 'source', a.seed, n_source=a.episodes, period=a.period))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
        torch.save(dict(pool=pool, files=raw['files'], dequant_seed=raw.get('dequant_seed')), tmp); os.replace(tmp, path)
    return pool, raw['files'], raw.get('dequant_seed')


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
    ap.add_argument('--seed', type=int, default=11); ap.add_argument('--ckpt-every', type=int, default=50)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--pool-cache', default=None, help='directory in which to reuse the source pool across models '
                    '(the pool depends only on dataset, seed, episodes and period, so reuse changes nothing)')
    ap.add_argument('--period', nargs=2, default=None, help='development only: override the source period (beijing)')
    ap.add_argument('--airq', default=realdata.DEFAULT['airq'])
    ap.add_argument('--beijing', default=realdata.DEFAULT['beijing'])
    ap.add_argument('--gas', default=realdata.DEFAULT['gas'])
    a = ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ck = out / 'ckpt.pt'
    if (out / 'model.pt').exists() and not ck.exists():     # with a checkpoint left, the run was cut off: finish it
        raise SystemExit(f'{out} already holds a fine-tuned model')
    meta = json.loads((Path(a.init) / 'train.json').read_text())
    model = build(meta['model'], meta.get('repo'), meta)
    if (Path(a.init) / 'model.pt').exists():
        model.load_state_dict(torch.load(Path(a.init) / 'model.pt', weights_only=True, map_location='cpu'))
    dev = devutil.pick(a.device); model.to(dev)
    pool, files, dq = source_pool(a)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    run, first, elapsed = None, 0, 0.
    if ck.exists():
        first, extra = resume.load(ck, model, opt, sched, rng); run, elapsed = extra['run'], extra['seconds']
    start = time.time() - elapsed
    model.train()
    for step in range(first, a.steps):
        batch = devutil.to_dev(batch_from(pool, rng), dev)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            resume.save(ck, step + 1, model, opt, sched, rng, dict(run=run, seconds=time.time() - start))
        if os.environ.get('MIRA_STOP_AT') == str(step + 1):        # test hook: simulate an interruption
            raise SystemExit('stopped for resume test')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(meta, finetune=dict(vars(a), dataset=a.dataset, seconds=time.time() - start,
                                                                         final_ema=run, files=files, dequant_seed=dq)), indent=1))
    print(out.name, a.dataset, f'ema {run:.4f} {time.time() - start:.0f}s')
    ck.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
