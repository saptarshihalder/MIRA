"""Protocol v9 sensor networks: three new domains, test periods in 2026 (in-context virtual sensing: one target sensor
from 11 input sensors). Data: fetch_v9.py snapshots (<data>/<name>.npz).

  dwd_t2m   German weather stations (DWD), hourly 2 m air temperature in deg C. Inputs: the 11 nearest stations.
            Weekly episodes. Test 2026-04-01..2026-10-01.
  aurn_no2  UK AURN air-quality sites, hourly NO2; values log1p(ug/m3) (negative readings are missing). Inputs: the 11
            nearest sites. Weekly episodes. Test 2026-04-01..2026-10-01.
  usgs_q    Pennsylvania USGS stream gauges, 30-minute discharge; values log1p(ft3/s). Inputs: the 11 gauges most
            correlated with the target over the SOURCE period (river networks are not isotropic). 3-day episodes.
            Test 2026-07-01..2026-10-01.
The station networks are nearly complete, so support rows get 20% simulated per-sensor dropout (as PEMS-BAY in v6);
query rows keep the natural mask. Episodes: one window of one target; 48 support + 48 query rows from the window's rows
where the target is observed (realdata._episode). Chronological split. 48 seeded test targets per network. Keys
'window-start|target' so intervals cluster over network-windows.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from realdata import _episode, NS, NQ

P_IN = 11
NETS = {   # transform, source start, test start, test end, window days, support dropout, test targets, neighbours
    'dwd_t2m': ('raw', '2025-05-01', '2026-04-01', '2026-10-01', 7, .2, 48, 'distance'),
    'aurn_no2': ('log1p', '2025-01-01', '2026-04-01', '2026-10-01', 7, .2, 48, 'distance'),
    'usgs_q': ('log1p', '2026-03-01', '2026-07-01', '2026-10-01', 3, .2, 48, 'correlation'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, folder):
    f = Path(folder) / f'{name}.npz'
    d = np.load(f)
    tf, s0, t0 = NETS[name][:3]
    V = d['V'].astype(float)
    if tf == 'log1p':
        V = np.log1p(np.where(V >= 0, V, np.nan))
    ts = d['ts'].astype('datetime64[s]')
    if NETS[name][7] == 'correlation':
        src = (ts >= np.datetime64(s0)) & (ts < np.datetime64(t0))
        C = np.array(pd.DataFrame(V[src]).corr().to_numpy(), copy=True)       # source period only
        np.fill_diagonal(C, -np.inf)
        C = np.nan_to_num(C, nan=-np.inf)
        nbr = np.argsort(-C, axis=1, kind='stable')[:, :P_IN]
    else:
        lat, lon = d['lat'].astype(float), d['lon'].astype(float)
        xy = np.c_[lat, lon * np.cos(np.deg2rad(lat))]                      # local equirectangular projection
        dd = ((xy[:, None] - xy[None]) ** 2).sum(-1); np.fill_diagonal(dd, np.inf)
        nbr = np.argsort(dd, axis=1, kind='stable')[:, :P_IN]
    return dict(V=V, ts=ts, nbr=nbr, ids=[str(i) for i in d['ids']], files={f.name: sha(f)})


def episodes(name, raw, split, seed, n_source=None):
    rng = np.random.default_rng(seed)
    _, s0, t0, t1, win, drop, k, _ = NETS[name]
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
