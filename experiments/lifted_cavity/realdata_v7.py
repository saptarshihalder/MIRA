"""Protocol v7 sensor networks (in-context virtual sensing: one target sensor from 11 input sensors).

  pems03  PeMS district 3 (Sacramento) flow, 358 sensors, 5-min, 2018-09-01..11-30 (Song et al. 2020, STSGCN release).
  pems08  PeMS district 8 (San Bernardino) flow, 170 sensors, 5-min, 2016-07-01..08-31 (Guo et al. 2019, ASTGCN).
          Both are complete (no natural missingness): support rows get 20% simulated per-sensor dropout, as PEMS-BAY in
          v6. Values are log1p(flow). Inputs: the 11 sensors most correlated with the target over the SOURCE period
          (the released graphs are sparse); daily episodes.
  air_cn  China urban air quality (Zheng et al. 2015), hourly PM2.5, 2014-05..2015-04, the 401 stations outside Beijing
          (Beijing PM2.5 is used data in this project). Values are log PM2.5 (non-positive readings are missing); natural
          missingness. Inputs: the 11 nearest non-Beijing stations; weekly episodes.

Episodes: one window (day / week) of one target; 48 support + 48 query rows from the window's rows where the target
is observed (realdata._episode). Chronological split. Test targets: a seeded subset (24 traffic, 48 air). Keys
'window-start|target' so intervals cluster over network-windows. Files from tsl's ProcessedDatasets mirror (hashed).
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from realdata import _episode, NS, NQ

P_IN = 11
NETS = {   # start of data, step, source start, test start, test end, window days, support dropout, test targets
    'pems03': ('2018-09-01', '5min', '2018-09-01', '2018-11-01', '2018-12-01', 1, .2, 24),
    'pems08': ('2016-07-01', '5min', '2016-07-01', '2016-08-11', '2016-09-01', 1, .2, 24),
    'air_cn': (None, None, '2014-05-01', '2015-02-01', '2015-05-01', 7, 0., 48),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, folder):
    folder = Path(folder)
    start, step, s0, t0, t1, win, drop, k = NETS[name]
    if name in ('pems03', 'pems08'):
        f = folder / name / f'{name}.npz'
        X = np.load(f)['data'][..., 0].astype(float)                  # flow channel
        ts = pd.date_range(start, periods=len(X), freq=step).values.astype('datetime64[s]')
        V = np.log1p(np.where(X >= 0, X, np.nan))
        src = (ts >= np.datetime64(s0)) & (ts < np.datetime64(t0))
        C = np.array(pd.DataFrame(V[src]).corr().to_numpy(), copy=True)   # source period only
        np.fill_diagonal(C, -np.inf)
        C = np.nan_to_num(C, nan=-np.inf)
        nbr = np.argsort(-C, axis=1, kind='stable')[:, :P_IN]
        return dict(V=V, ts=ts, nbr=nbr, ids=[str(i) for i in range(V.shape[1])], files={f'{name}.npz': sha(f)})
    f = folder / 'air_quality' / 'full437.h5'
    pm = pd.read_hdf(f, 'pm25'); st = pd.read_hdf(f, 'stations')
    keep = [c for c in pm.columns if int(c) // 1000 != 1]               # drop Beijing (city 1)
    pm = pm[keep]
    V = np.array(pm.to_numpy(float), copy=True); V[V <= 0] = np.nan; V = np.log(V)
    xy = st.loc[[int(c) for c in keep], ['latitude', 'longitude']].to_numpy(float)
    xy = np.c_[xy[:, 0], xy[:, 1] * np.cos(np.deg2rad(xy[:, 0]))]      # local equirectangular projection
    d = ((xy[:, None] - xy[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1, kind='stable')[:, :P_IN]
    return dict(V=V, ts=pm.index.values.astype('datetime64[s]'), nbr=nbr, ids=[str(c) for c in keep], files={'full437.h5': sha(f)})


def episodes(name, raw, split, seed, n_source=None):
    rng = np.random.default_rng(seed)
    _, _, s0, t0, t1, win, drop, k = NETS[name]
    V, ts, nbr = raw['V'], raw['ts'], raw['nbr']
    lo, hi = (np.datetime64(s0), np.datetime64(t0)) if split == 'source' else (np.datetime64(t0), np.datetime64(t1))
    starts = np.arange(lo, hi, np.timedelta64(win, 'D')).astype('datetime64[s]')
    starts = starts[starts + np.timedelta64(win, 'D') <= hi]
    rows = {s: np.flatnonzero((ts >= s) & (ts < s + np.timedelta64(win, 'D'))) for s in starts}
    N = V.shape[1]
    out = []
    if split == 'test':
        targets = sorted(np.random.default_rng(seed + 7).choice(N, k, replace=False))
        for s in starts:
            for tgt in targets:
                idx = rows[s][~np.isnan(V[rows[s], tgt])]
                if len(idx) >= NS + NQ:
                    out.append(_episode(rng, V[:, nbr[tgt]], V[:, tgt], idx, drop) + (f'{str(s)[:10]}|{raw["ids"][tgt]}',))
    else:
        while len(out) < n_source:
            s, tgt = starts[rng.integers(len(starts))], int(rng.integers(N))
            idx = rows[s][~np.isnan(V[rows[s], tgt])]
            if len(idx) >= NS + NQ:
                out.append(_episode(rng, V[:, nbr[tgt]], V[:, tgt], idx, drop) + ('src',))
    return out
