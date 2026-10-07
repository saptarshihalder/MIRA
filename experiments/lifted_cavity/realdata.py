"""Real multi-sensor benchmarks as in-context episodes: 48 labeled support rows + 48 query rows per episode.

Datasets (raw files are not committed; see README for sources and hashes):
  airq_co / airq_no2  UCI Air Quality (De Vito et al. 2008): 5 metal-oxide sensors -> log CO(GT) / log NO2(GT).
                      Weekly episodes; source = windows starting before 2004-10-01, test = later windows.
                      Sensor dropout is simulated (the device's own outages remove all five sensors at once).
  beijing             Beijing multi-site air quality (Zhang et al. 2017): log PM2.5 at a target station from the
                      other 11 stations, with their NATURAL missingness. Weekly episodes per target station;
                      source = 2013-03..2015-12, test = 2016-01..2017-02.
  gas                 Gas sensor array drift (Vergara et al. 2012): log concentration from 16 metal-oxide sensors
                      (log1p of the steady-state resistance change). Episodes within (batch, gas);
                      source = batches 1-6, test = batches 7-10 (months 21-36, after substantial drift).
Support rows of airq/gas get 20% simulated per-sensor dropout (as in the synthetic source regime); Beijing support
rows keep only their natural missingness. Evaluation conditions add k extra dropped sensors per query (seeded).
"""
import glob, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import family as fam
import fa as fa_mod
import nlfa
import anchors as an

NS = NQ = 48
_RAW = Path(__file__).resolve().parents[2] / 'data' / 'raw'
DEFAULT = dict(airq=os.environ.get('MIRA_AIRQ_CSV', str(_RAW / 'air_quality' / 'AirQualityUCI.csv')),
               beijing=os.environ.get('MIRA_BEIJING_DIR', str(_RAW / 'beijing_multisite' / 'PRSA_Data_20130301-20170228')),
               gas=os.environ.get('MIRA_GAS_DIR', str(_RAW / 'gas_sensor_drift')))
AQ_SENSORS = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ raw loaders: X (rows, P) with NaN, y (rows,), keys
def load_airq(csv, target):
    df = pd.read_csv(csv, sep=';', decimal=',').dropna(how='all', axis=1).dropna(how='all', axis=0).replace(-200, np.nan)
    df['ts'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df = df.dropna(subset=AQ_SENSORS + [target]).sort_values('ts').reset_index(drop=True)
    df = df[df[target] > 0].reset_index(drop=True)
    return dict(X=df[AQ_SENSORS].to_numpy(float), y=np.log(df[target].to_numpy(float)), ts=df['ts'].to_numpy(),
                files={Path(csv).name: sha(csv)})


def load_beijing(folder, pollutant='PM2.5'):
    fs = sorted(glob.glob(str(Path(folder) / 'PRSA_Data_*_20130301-20170228.csv')))
    assert len(fs) == 12, fs
    dfs = [pd.read_csv(f) for f in fs]
    ts = pd.to_datetime(dfs[0][['year', 'month', 'day', 'hour']]).to_numpy()
    for d in dfs:
        assert (pd.to_datetime(d[['year', 'month', 'day', 'hour']]).to_numpy() == ts).all()
    V = np.column_stack([d[pollutant].to_numpy(float) for d in dfs])
    V[V <= 0] = np.nan
    L = np.log(V)
    return dict(L=L, ts=ts, stations=[d['station'].iloc[0] for d in dfs], files={Path(f).name: sha(f) for f in fs})


def load_gas(folder):
    rows = []
    for b in range(1, 11):
        for line in open(Path(folder) / f'batch{b}.dat'):
            parts = line.split()
            g, c = parts[0].split(';')
            v = np.zeros(128)
            for p in parts[1:]:
                i, x = p.split(':'); v[int(i) - 1] = float(x)
            rows.append((b, int(g), float(c), v[0::8]))
    B = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows])
    y = np.log(np.array([r[2] for r in rows])); X = np.log1p(np.clip(np.stack([r[3] for r in rows]), 0, None))
    return dict(X=X, y=y, batch=B, gas=G, files={f'batch{b}.dat': sha(Path(folder) / f'batch{b}.dat') for b in range(1, 11)})


# ------------------------------------------------------------------ episodes
def _episode(rng, X, y, idx, support_missing):
    pick = rng.choice(idx, NS + NQ, replace=False)
    x, yy = X[pick], y[pick]
    nat = ~np.isnan(x)
    sm = nat[:NS] & (rng.random((NS, x.shape[1])) > support_missing)
    return x, yy, sm.astype(float), nat[NS:].astype(float)


def episodes(name, raw, split, seed, n_source=None, period=None):
    """Return list of (x raw with NaN, y, support mask, natural query mask, key) for one split.

    split='test' gives the fixed evaluation episodes; split='source' with n_source draws random training episodes."""
    rng = np.random.default_rng(seed)
    out = []
    if name.startswith('airq'):
        X, y, ts = raw['X'], raw['y'], raw['ts']
        cut = np.datetime64('2004-10-01')
        if split == 'test':
            starts = pd.date_range(pd.Timestamp(ts.min()), pd.Timestamp(ts.max()) - pd.Timedelta(days=7), freq='7D')
            for s in starts:
                if np.datetime64(s) < cut:
                    continue
                idx = np.flatnonzero((ts >= np.datetime64(s)) & (ts < np.datetime64(s + pd.Timedelta(days=7))))
                if len(idx) >= NS + NQ:
                    out.append(_episode(rng, X, y, idx, .2) + (str(s.date()),))
        else:
            src = np.flatnonzero(ts < cut); t = ts[src].astype('datetime64[s]').astype(np.int64)
            week = 7 * 24 * 3600
            while len(out) < n_source:
                s0 = t.min() + int((t.max() - week - t.min()) * rng.random())
                idx = src[(t >= s0) & (t < s0 + week)]
                if len(idx) >= NS + NQ:
                    out.append(_episode(rng, X, y, idx, .2) + ('src',))
    elif name.startswith('beijing'):
        L, ts = raw['L'], raw['ts']
        lo, hi = (np.datetime64('2013-03-01'), np.datetime64('2016-01-01')) if split == 'source' else (np.datetime64('2016-01-01'), np.datetime64('2017-03-01'))
        if period is not None:                                        # development checks only
            lo, hi = np.datetime64(period[0]), np.datetime64(period[1])
        P = L.shape[1]
        if split == 'test':
            starts = pd.date_range(pd.Timestamp(lo), pd.Timestamp(hi) - pd.Timedelta(days=7), freq='7D')
            for s in starts:
                win = np.flatnonzero((ts >= np.datetime64(s)) & (ts < np.datetime64(s + pd.Timedelta(days=7))))
                for tgt in range(P):
                    idx = win[~np.isnan(L[win, tgt])]
                    if len(idx) >= NS + NQ:
                        others = [j for j in range(P) if j != tgt]
                        out.append(_episode(rng, L[:, others], L[:, tgt], idx, 0.) + (f'{s.date()}|{raw["stations"][tgt]}',))
        else:
            sel = np.flatnonzero((ts >= lo) & (ts < hi)); t = ts[sel].astype('datetime64[s]').astype(np.int64)
            week = 7 * 24 * 3600
            while len(out) < n_source:
                tgt = int(rng.integers(P))
                s0 = t.min() + int((t.max() - week - t.min()) * rng.random())
                win = sel[(t >= s0) & (t < s0 + week)]
                idx = win[~np.isnan(L[win, tgt])]
                if len(idx) >= NS + NQ:
                    others = [j for j in range(P) if j != tgt]
                    out.append(_episode(rng, L[:, others], L[:, tgt], idx, 0.) + ('src',))
    elif name == 'gas':
        X, y, B, G = raw['X'], raw['y'], raw['batch'], raw['gas']
        if split == 'test':
            for b in range(7, 11):
                for g in range(1, 7):
                    idx = rng.permutation(np.flatnonzero((B == b) & (G == g)))
                    for e in range(len(idx) // (NS + NQ)):
                        chunk = idx[e * (NS + NQ):(e + 1) * (NS + NQ)]
                        out.append(_episode(rng, X, y, chunk, .2) + (f'b{b}g{g}e{e}',))
        else:
            groups = [np.flatnonzero((B == b) & (G == g)) for b in range(1, 7) for g in range(1, 7)]
            groups = [g for g in groups if len(g) >= NS + NQ]
            while len(out) < n_source:
                out.append(_episode(rng, X, y, groups[rng.integers(len(groups))], .2) + ('src',))
    return out


def to_pool(eps):
    x = np.stack([e[0] for e in eps]); y = np.stack([e[1] for e in eps]); sm = np.stack([e[2] for e in eps])
    qnat = np.stack([e[3] for e in eps])
    x0 = np.nan_to_num(x)
    c, mean, scale = fam.context(x0[:, :NS], y[:, :NS], sm)
    sx = (x0[:, :NS] - mean[:, None]) / scale[:, None] * sm
    qx = np.nan_to_num((x[:, NS:] - mean[:, None]) / scale[:, None]) * qnat
    pool = dict(c=torch.as_tensor(c), sx=torch.as_tensor(sx, dtype=torch.float32), sy=torch.as_tensor(y[:, :NS], dtype=torch.float32),
                sm=torch.as_tensor(sm, dtype=torch.float32), qx=torch.as_tensor(qx, dtype=torch.float32),
                qy=torch.as_tensor(y[:, NS:], dtype=torch.float32), qnat=torch.as_tensor(qnat, dtype=torch.float32))
    mu, sig = an.em_batched(sx, y[:, :NS], sm)
    pool['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in (0, 1, 2)}
    pool['n'] = len(x)
    pool['keys'] = [e[4] for e in eps]
    return pool


def condition_masks(pool, extra, seed):
    """Per-query masks: natural mask with `extra` additional observed sensors dropped (never all)."""
    rng = np.random.default_rng(seed)
    qm = pool['qnat'].numpy().copy()
    if extra:
        for t in range(qm.shape[0]):
            for q in range(qm.shape[1]):
                obs = np.flatnonzero(qm[t, q])
                k = min(extra, len(obs) - 1)
                if k > 0:
                    qm[t, q, rng.choice(obs, k, replace=False)] = 0
    return qm


def closed_forms(pool, qm):
    """Per-episode metrics of the support-only closed forms under per-query masks qm (n, Q, P).
    Returns {name: {metric: (n,)}}; NL-FA has NLL only (its predictive is not Gaussian)."""
    from evaluate2 import gauss_metrics
    gauss = ('em_gauss', 'fa1', 'ridge_cc', 'blr_cc')
    out = {k: {m: np.full(pool['n'], np.nan) for m in ('nll', 'se', 'cov', 'crps')} for k in gauss + ('nlfa',)}
    for t in range(pool['n']):
        sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm'))
        qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
        mu, sig = fam.em_gaussian(sx, sy, sm, 40); f1 = fa_mod.fit_fa(mu, sig, 1); nm = nlfa.fit(sx, sy, sm)
        acc = {k: {m: np.zeros(len(qy)) for m in ('nll', 'se', 'cov', 'crps')} for k in gauss}
        acc_nl = np.zeros(len(qy))
        pats = {}
        for q in range(len(qy)):
            pats.setdefault(qm[t, q].tobytes(), []).append(q)
        for key, idx in pats.items():
            S = np.flatnonzero(qm[t, idx[0]]); idx = np.array(idx)
            preds = dict(em_gauss=fam.em_conditional(mu, sig, qx[idx], S, NS), fa1=fa_mod.lifted_poe(f1, qx[idx], S, NS),
                         ridge_cc=fam.ridge_cc(sx, sy, sm.astype(bool), qx[idx], S), blr_cc=fam.blr_cc(sx, sy, sm.astype(bool), qx[idx], S))
            for name, (pmu, pvar) in preds.items():
                for m, v in gauss_metrics(pmu, pvar, qy[idx]).items():
                    acc[name][m][idx] = v
            acc_nl[idx] = nlfa.predict_nll(nm, qx[idx], qy[idx], S)
        for k in gauss:
            for m in ('nll', 'se', 'cov', 'crps'):
                out[k][m][t] = acc[k][m].mean()
        out['nlfa']['nll'][t] = acc_nl.mean()
    return out


@torch.no_grad()
def score_model(model, pool, qm, tasks_per_batch=8):
    """Per-episode metrics for a learned model under per-query masks qm."""
    from evaluate2 import gauss_metrics
    model.eval()
    res = []
    qmt = torch.as_tensor(qm, dtype=torch.float32)
    for s0 in range(0, pool['n'], tasks_per_batch):
        t = torch.arange(s0, min(pool['n'], s0 + tasks_per_batch))
        B = len(t)
        batch = {k: pool[k][t] for k in ('c', 'sx', 'sy', 'sm')}
        batch['anchors'] = {K: {k: v[t] for k, v in pool['anchors'][K].items()} for K in (0, 1, 2)}
        batch['qx'] = pool['qx'][t].reshape(B * NQ, -1); batch['qy'] = pool['qy'][t].reshape(-1)
        batch['qm'] = qmt[t].reshape(B * NQ, -1); batch['tid'] = torch.arange(B).repeat_interleave(NQ)
        import devutil
        mu, lv = model(devutil.to_dev(batch, devutil.model_device(model)))
        mu, lv = mu.detach().cpu(), lv.detach().cpu()
        g = gauss_metrics(mu.double().numpy(), np.exp(lv.double().numpy()), batch['qy'].double().numpy())
        res.append({m: v.reshape(B, NQ).mean(1) for m, v in g.items()})
    return {m: np.concatenate([r[m] for r in res]) for m in res[0]}
