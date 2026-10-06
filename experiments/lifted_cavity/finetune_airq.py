"""Fine-tune a synthetic-prior model on Air Quality SOURCE weeks only (before 2004-10-01); score on later test weeks.

python finetune_airq.py --csv AirQualityUCI.csv --init runs/lift1_s1 --panel cache/airq_s7.pt --out runs_airq/lift1_s1_ft
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import airq, data, models
from evaluate import score
from train import build


def source_pool(csv, n_episodes, seed, split='2004-10-01', ns=48, nq=48, support_missing=.2, target='CO(GT)'):
    df = airq.load(csv, target)
    df = df[df['ts'] < pd.Timestamp(split)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    X = df[airq.SENSORS].to_numpy(float); Y = np.log(df[target].to_numpy(float))
    ts = df['ts'].to_numpy().astype('datetime64[s]').astype(np.int64)
    week = 7 * 24 * 3600
    lo, hi = ts.min(), ts.max() - week
    xs, ys, sms = [], [], []
    while len(xs) < n_episodes:
        start = lo + int((hi - lo) * rng.random())
        idx = np.flatnonzero((ts >= start) & (ts < start + week))
        if len(idx) < ns + nq:
            continue
        pick = rng.choice(idx, ns + nq, replace=False)
        xs.append(X[pick]); ys.append(Y[pick]); sms.append((rng.random((ns, 5)) > support_missing).astype(float))
    return airq.to_pool(dict(x=np.array(xs), y=np.array(ys), sm=np.array(sms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--init', required=True)
    ap.add_argument('--panel', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=4000); ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--target', default='CO(GT)')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((Path(a.init) / 'train.json').read_text())
    model = build(meta['model'], meta['repo'])
    model.load_state_dict(torch.load(Path(a.init) / 'model.pt', weights_only=True))
    panel_target = torch.load(a.panel, weights_only=False)['meta'].get('target', 'CO(GT)')
    assert panel_target == a.target, (panel_target, a.target)
    pool = source_pool(a.csv, a.episodes, a.seed, target=a.target)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    start = time.time(); run = None
    for step in range(a.steps):
        batch = data.train_batch(pool, rng, 64, 8)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'train.json').write_text(json.dumps(dict(meta, finetune=vars(a), finetune_seconds=time.time() - start, final_ema=run), indent=1))
    panel = torch.load(a.panel, weights_only=False)
    res = {}
    for split in ('source', 'test'):
        cells = score(model, panel['split'][split]['pool'])
        np.savez_compressed(out / f'cells_airq_{split}.npz', **{f'k{k}': v for k, v in cells.items()})
        res[split] = {f'k{k}': float(v.mean()) for k, v in cells.items()}
    (out / 'score_airq.json').write_text(json.dumps(res, indent=1))
    print(out.name, json.dumps(res))


if __name__ == '__main__':
    main()
