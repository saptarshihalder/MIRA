"""UCI Air Quality (De Vito et al. 2008) as episodic in-context sensor-fusion tasks.

Five co-located metal-oxide sensors PT08.S1..S5 (hourly) and a reference analyser (CO(GT)).  Each episode is a
7-day window: 48 labeled support hours (simulated co-location calibration) and 48 query hours, all from rows
where the five sensors and the reference are recorded.  Support sensors are independently deleted with prob 20%
(matching the synthetic source regime); query sensors are deleted at evaluation by enumerating all k-missing masks.
The dataset's own sensor missingness is device-level (all five together), so per-sensor dropout is simulated.
Target: log CO(GT) by default (--target selects another reference column, e.g. NO2(GT)).  Temporal split: windows starting before 2004-10-01 are 'source', later windows 'test'.

python airq.py build --csv AirQualityUCI.csv --seed 7 --out cache/airq.pt
python airq.py refs  --panel cache/airq.pt
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import family as fam, fa as fm, anchors as an

SENSORS = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']


def load(csv, target='CO(GT)'):
    df = pd.read_csv(csv, sep=';', decimal=',').dropna(how='all', axis=1).dropna(how='all', axis=0)
    df = df.replace(-200, np.nan)
    df['ts'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df = df.dropna(subset=SENSORS + [target]).sort_values('ts').reset_index(drop=True)
    df = df[df[target] > 0].reset_index(drop=True)
    df.attrs['target'] = target
    return df


def episodes(df, seed, ns=48, nq=48, window_days=7, stride_days=7, support_missing=.2, split='2004-10-01'):
    rng = np.random.default_rng(seed)
    t0, t1 = df['ts'].min(), df['ts'].max()
    starts = pd.date_range(t0, t1 - pd.Timedelta(days=window_days), freq=pd.Timedelta(days=stride_days))
    X = df[SENSORS].to_numpy(float); Y = np.log(df[df.attrs['target']].to_numpy(float))
    out = {k: [] for k in ('x', 'y', 'sm', 'start', 'source')}
    for s in starts:
        idx = np.flatnonzero((df['ts'] >= s) & (df['ts'] < s + pd.Timedelta(days=window_days)))
        if len(idx) < ns + nq:
            continue
        pick = rng.choice(idx, ns + nq, replace=False)
        out['x'].append(X[pick]); out['y'].append(Y[pick])
        out['sm'].append((rng.random((ns, 5)) > support_missing).astype(float))
        out['start'].append(str(s)); out['source'].append(bool(s < pd.Timestamp(split)))
    return {k: (np.array(v) if k != 'start' else v) for k, v in out.items()}


def to_pool(ep, ns=48):
    x, y, sm = ep['x'], ep['y'], ep['sm']
    c, mean, scale = fam.context(x[:, :ns], y[:, :ns], sm)
    sx = (x[:, :ns] - mean[:, None]) / scale[:, None] * sm
    qx = (x[:, ns:] - mean[:, None]) / scale[:, None]
    pool = dict(c=torch.as_tensor(c), sx=torch.as_tensor(sx, dtype=torch.float32), sy=torch.as_tensor(y[:, :ns], dtype=torch.float32),
                sm=torch.as_tensor(sm, dtype=torch.float32), qx=torch.as_tensor(qx, dtype=torch.float32),
                qy=torch.as_tensor(y[:, ns:], dtype=torch.float32))
    mu, sig = an.em_batched(sx, y[:, :ns], sm)
    pool['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in (0, 1, 2)}
    pool['em_mu'], pool['em_sig'] = mu.float(), sig.float()
    pool['n'] = len(x)
    return pool


def refs(pool):
    out = {}
    for k in (0, 1, 2, 3):
        bank = fam.mask_bank(k)
        cells = {n: np.zeros((pool['n'], len(bank))) for n in ('ridge_repo', 'ridge_cc', 'em_gauss', 'fa1')}
        for t in range(pool['n']):
            sx, sy, sm = (pool[k_][t].double().numpy() for k_ in ('sx', 'sy', 'sm'))
            qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
            mu, sig = fam.em_gaussian(sx, sy, sm, 40); f1 = fm.fit_fa(mu, sig, 1)
            for i, m in enumerate(bank):
                S = np.flatnonzero(m)
                cells['ridge_repo'][t, i] = fam.gnll(*fam.ridge_repo(sx, sy, qx, S), qy).mean()
                cells['ridge_cc'][t, i] = fam.gnll(*fam.ridge_cc(sx, sy, sm.astype(bool), qx, S), qy).mean()
                cells['em_gauss'][t, i] = fam.gnll(*fam.em_conditional(mu, sig, qx, S, len(sy)), qy).mean()
                cells['fa1'][t, i] = fam.gnll(*fm.lifted_poe(f1, qx, S, len(sy)), qy).mean()
        out[k] = cells
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['build'])
    ap.add_argument('--csv', required=True); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--target', default='CO(GT)'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = load(a.csv, a.target)
    ep = episodes(df, a.seed)
    meta = dict(csv_sha256=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(), rows=len(df), episodes=len(ep['y']),
                source=int(np.sum(ep['source'])), test=int(np.sum(~np.array(ep['source']))), seed=a.seed, target=a.target)
    split = {}
    for name, sel in (('source', np.array(ep['source'])), ('test', ~np.array(ep['source']))):
        sub = {k: (v[sel] if k != 'start' else [s for s, f in zip(v, sel) if f]) for k, v in ep.items()}
        pool = to_pool(sub)
        split[name] = dict(pool=pool, refs=refs(pool), starts=sub['start'])
    torch.save(dict(split=split, meta=meta), a.out)
    print(json.dumps(meta))
    for name in split:
        print(name, json.dumps({k: {n: round(float(v.mean()), 4) for n, v in c.items()} for k, c in split[name]['refs'].items()}))
