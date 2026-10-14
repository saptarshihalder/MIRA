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
import glob, hashlib, os, zlib
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
               gas=os.environ.get('MIRA_GAS_DIR', str(_RAW / 'gas_sensor_drift')),
               v6=os.environ.get('MIRA_V6_DIR', str(_RAW / 'v6')))
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


def load_beijing(folder, pollutant='PM2.5', dequant=False):
    """Log concentrations, stations as columns. dequant (protocol v4, amendment 1): each integer reading becomes
    reading + U(-1/2, 1/2) before the log, with noise fixed per pollutant; the few readings on finer grids (2013-2015
    only) are left as they are."""
    fs = sorted(glob.glob(str(Path(folder) / 'PRSA_Data_*_20130301-20170228.csv')))
    assert len(fs) == 12, fs
    dfs = [pd.read_csv(f) for f in fs]
    ts = pd.to_datetime(dfs[0][['year', 'month', 'day', 'hour']]).to_numpy()
    for d in dfs:
        assert (pd.to_datetime(d[['year', 'month', 'day', 'hour']]).to_numpy() == ts).all()
    V = np.column_stack([d[pollutant].to_numpy(float) for d in dfs])
    V[V <= 0] = np.nan
    seed = None
    if dequant:
        seed = [4041, zlib.crc32(pollutant.encode())]
        noise = np.random.default_rng(seed).uniform(-.5, .5, V.shape)
        V = np.where(np.isfinite(V) & np.isclose(V, np.round(V)), V + noise, V)
    L = np.log(V)
    return dict(L=L, ts=ts, stations=[d['station'].iloc[0] for d in dfs], files={Path(f).name: sha(f) for f in fs},
                dequant_seed=seed)


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
    if name in ('metr_la', 'pems_bay', 'intel'):                     # protocol v6 networks
        import realdata_v6
        return realdata_v6.episodes(name, raw, split, seed, n_source)
    if name in ('pems03', 'pems08', 'air_cn'):                       # protocol v7 networks
        import realdata_v7
        return realdata_v7.episodes(name, raw, split, seed, n_source)
    if name in ('dwd_t2m', 'aurn_no2', 'usgs_q'):                   # protocol v9 networks
        import realdata_v9
        return realdata_v9.episodes(name, raw, split, seed, n_source)
    if name in ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'):         # protocol v10 networks
        import realdata_v10
        return realdata_v10.episodes(name, raw, split, seed, n_source)
    if name in ('usgs_ny_f', 'usgs_wa_f', 'dwd_p_f', 'aurn_pm25_f'):  # protocol v16: v10 networks, new days and targets
        import realdata_v16
        return realdata_v16.episodes(name, raw, split, seed, n_source)
    if name in ('dwd_t2m_f', 'aurn_no2_f', 'usgs_q_f'):              # protocol v17: v9 networks, new days and targets
        import realdata_v17
        return realdata_v17.episodes(name, raw, split, seed, n_source)
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


def zpool(pool):
    """Protocol v10 scale-free (SF) pipeline: each episode's target standardized by its support mean and standard
    deviation (floor 1e-4 (1 + |mean|)), anchors refitted on the standardized target. 'yloc' and 'yscale' map back to raw
    units (score_model does this, so every reported metric stays in raw units). The inputs are already standardized
    per episode (to_pool), so the whole predictive becomes equivariant to affine maps of the target: no fixed scale
    (the .1 clamp of the backbone, the absolute ridges of the anchors) acts on the raw target any more."""
    sy = pool['sy'].double(); m = sy.mean(1); s = torch.maximum(sy.std(1), 1e-4 * (1 + m.abs()))
    out = dict(pool)
    out['sy'] = ((sy - m[:, None]) / s[:, None]).float()
    out['qy'] = ((pool['qy'].double() - m[:, None]) / s[:, None]).float()
    mu, sig = an.em_batched(pool['sx'].double().numpy(), out['sy'].double().numpy(), pool['sm'].double().numpy())
    out['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in (0, 1, 2)}
    out['yloc'], out['yscale'] = m, s
    return out


CF = NS // 2
CF_SEED = 20000                    # protocol v11: seed of the cross-fit masks (plus the condition)


def crossfit_masks(pool, extra, seed):
    """Protocol v11: masks of the support rows when they are predicted in the two-fold cross-fit: the support mask with
    `extra` further observed inputs dropped per row (never all). Shared by every model, TabPFN included."""
    rng = np.random.default_rng(seed)
    m = pool['sm'].numpy().copy()
    if extra:
        for t in range(m.shape[0]):
            for r in range(m.shape[1]):
                obs = np.flatnonzero(m[t, r])
                k = min(extra, len(obs) - 1)
                if k > 0:
                    m[t, r, rng.choice(obs, k, replace=False)] = 0
    return m


def crossfit_pool(pool):
    """Protocol v11: two-fold cross-fit of every episode's support. Episode 2t predicts support rows [0, CF) from rows
    [CF, NS); episode 2t+1 predicts rows [CF, NS) from rows [0, CF). Inputs keep the episode's standardization; the
    anchors are refitted on each half. Per-query outputs of the two reshape to (n, NS) in the original row order."""
    n = pool['n']
    halves = [(slice(CF, NS), slice(0, CF)), (slice(0, CF), slice(CF, NS))]
    sup = lambda k: torch.stack([pool[k][t, h[0]] for t in range(n) for h in halves], 0)
    qry = lambda k: torch.stack([pool[k][t, h[1]] for t in range(n) for h in halves], 0)
    out = dict(sx=sup('sx'), sy=sup('sy'), sm=sup('sm'), qx=qry('sx'), qy=qry('sy'), qnat=qry('sm'))
    out['c'] = torch.zeros(2 * n, pool['c'].shape[1], dtype=pool['c'].dtype)
    mu, sig = an.em_batched(out['sx'].double().numpy(), out['sy'].double().numpy(), out['sm'].double().numpy())
    out['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in (0, 1, 2)}
    out['n'] = 2 * n
    out['keys'] = [k for k in pool['keys'] for _ in halves]
    return out


def crossfit_query_masks(m):
    """(n, NS, P) cross-fit masks -> (2n, CF, P) query masks of crossfit_pool's episodes."""
    return np.stack([m[t, sl] for t in range(m.shape[0]) for sl in (slice(0, CF), slice(CF, NS))], 0)


def kfolds(K):
    """Protocol v15: K contiguous folds of the NS support rows, in order: (training rows, held-out rows). K = 2 gives the
    halves of crossfit_pool, in the same order."""
    size = NS // K
    assert size * K == NS, (NS, K)
    return [(np.r_[0:k * size, (k + 1) * size:NS], slice(k * size, (k + 1) * size)) for k in range(K)]


def crossfit_pool_k(pool, K):
    """Protocol v15: K-fold cross-fit of every episode's support (crossfit_pool for K folds). Episode K t + k predicts
    fold k from the other K - 1 folds; anchors are refitted on each training part. Per-query outputs reshape to (n, NS)
    in the original row order."""
    n, folds = pool['n'], kfolds(K)
    sup = lambda k: torch.stack([pool[k][t, torch.as_tensor(tr)] for t in range(n) for tr, _ in folds], 0)
    qry = lambda k: torch.stack([pool[k][t, te] for t in range(n) for _, te in folds], 0)
    out = dict(sx=sup('sx'), sy=sup('sy'), sm=sup('sm'), qx=qry('sx'), qy=qry('sy'), qnat=qry('sm'))
    out['c'] = torch.zeros(K * n, pool['c'].shape[1], dtype=pool['c'].dtype)
    mu, sig = an.em_batched(out['sx'].double().numpy(), out['sy'].double().numpy(), out['sm'].double().numpy())
    out['anchors'] = {A: {k: v.float() for k, v in an.fa_batched(mu, sig, A).items()} for A in (0, 1, 2)}
    out['n'] = K * n
    out['keys'] = [k for k in pool['keys'] for _ in folds]
    return out


def crossfit_query_masks_k(m, K):
    """(n, NS, P) cross-fit masks -> (K n, NS / K, P) query masks of crossfit_pool_k's episodes."""
    return np.stack([m[t, te] for t in range(m.shape[0]) for _, te in kfolds(K)], 0)


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
        mu, sig = fam.em_gaussian(sx, sy, sm, 40); f1 = fa_mod.fit_fa(mu, sig, 1)
        try:
            nm = nlfa.fit(sx, sy, sm)
        except np.linalg.LinAlgError:                              # singular NL-FA design (e.g. an all-missing sensor):
            nm = None                                              # undefined, recorded as NaN (protocol v6 amendment 2)
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
            acc_nl[idx] = nlfa.predict_nll(nm, qx[idx], qy[idx], S) if nm is not None else np.nan
        for k in gauss:
            for m in ('nll', 'se', 'cov', 'crps'):
                out[k][m][t] = acc[k][m].mean()
        out['nlfa']['nll'][t] = acc_nl.mean()
    return out


@torch.no_grad()
def score_model(model, pool, qm, tasks_per_batch=8, tta=1, per_query=False):
    """Per-episode metrics for a learned model under per-query masks qm.

    tta > 1 (protocol v9): the predictive is the equal-weight linear pool of the model's predictives under feature-
    embedding seeds 0..tta-1 (seed 0 is the single-model default), a test-time ensemble like TabPFN's n_estimators.
    per_query (protocol v11): also return 'q_nll' and 'q_mean' (n, queries) in raw target units, and (protocol v14)
    'q_var', the predictive variance."""
    from evaluate2 import gauss_metrics
    model.eval()
    res, pq = [], []
    NQ = pool['qx'].shape[1]                                        # 48 for panels, CF for cross-fit pools
    qmt = torch.as_tensor(qm, dtype=torch.float32)
    for s0 in range(0, pool['n'], tasks_per_batch):
        t = torch.arange(s0, min(pool['n'], s0 + tasks_per_batch))
        B = len(t)
        batch = {k: pool[k][t] for k in ('c', 'sx', 'sy', 'sm')}
        batch['anchors'] = {K: {k: v[t] for k, v in pool['anchors'][K].items()} for K in (0, 1, 2)}
        batch['qx'] = pool['qx'][t].reshape(B * NQ, -1); batch['qy'] = pool['qy'][t].reshape(-1)
        batch['qm'] = qmt[t].reshape(B * NQ, -1); batch['tid'] = torch.arange(B).repeat_interleave(NQ)
        import devutil
        if tta > 1:
            import upt
            dbatch = devutil.to_dev(batch, devutil.model_device(model))
            W, Mu, Var = [], [], []
            for k in range(tta):
                model.feature_seed = k
                lw, mu, lv = (v.detach().cpu().double().numpy() for v in upt.components(model, dbatch))
                W.append(np.exp(lw) / tta); Mu.append(mu); Var.append(np.exp(lv))
            model.feature_seed = 0
            W, Mu, Var = np.concatenate(W, 1), np.concatenate(Mu, 1), np.concatenate(Var, 1)
            g = upt.mixture_metrics(W, Mu, Var, batch['qy'].double().numpy())
            mean = (W * Mu).sum(1); var = (W * (Var + Mu ** 2)).sum(1) - mean ** 2
        elif hasattr(model, 'mixture'):                            # mixture output (upt.py): exact mixture metrics
            import upt
            lw, mu, lv = (v.detach().cpu().double().numpy() for v in model.mixture(devutil.to_dev(batch, devutil.model_device(model))))
            g = upt.mixture_metrics(np.exp(lw), mu, np.exp(lv), batch['qy'].double().numpy())
            mean = (np.exp(lw) * mu).sum(1); var = (np.exp(lw) * (np.exp(lv) + mu ** 2)).sum(1) - mean ** 2
        else:
            mu, lv = model(devutil.to_dev(batch, devutil.model_device(model)))
            mu, lv = mu.detach().cpu(), lv.detach().cpu()
            g = gauss_metrics(mu.double().numpy(), np.exp(lv.double().numpy()), batch['qy'].double().numpy())
            mean = mu.double().numpy(); var = np.exp(lv.double().numpy())
        res.append({m: v.reshape(B, NQ).mean(1) for m, v in g.items()})
        pq.append((g['nll'].reshape(B, NQ), mean.reshape(B, NQ), np.maximum(var, 0).reshape(B, NQ)))
    out = {m: np.concatenate([r[m] for r in res]) for m in res[0]}
    if per_query:
        out['q_nll'] = np.concatenate([p[0] for p in pq]); out['q_mean'] = np.concatenate([p[1] for p in pq])
        out['q_var'] = np.concatenate([p[2] for p in pq])          # protocol v14: predictive variance
    if 'yscale' in pool:                                           # SF pipeline: metrics back in raw target units
        s = pool['yscale'].double().numpy()
        out['nll'] = out['nll'] + np.log(s); out['crps'] = out['crps'] * s; out['se'] = out['se'] * s * s
        if per_query:
            out['q_nll'] = out['q_nll'] + np.log(s)[:, None]
            out['q_mean'] = pool['yloc'].double().numpy()[:, None] + s[:, None] * out['q_mean']
            out['q_var'] = out['q_var'] * (s ** 2)[:, None]
    return out
