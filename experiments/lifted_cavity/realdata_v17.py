"""Protocol v17 networks: the three v9 networks on days after TabPFN-3.5's release (2026-09-09), with new targets.

  dwd_t2m_f, aurn_no2_f  v9's dwd_t2m / aurn_no2: same stations, transforms and distance neighbours; test
                         2026-09-10..10-11, weekly episodes.
  usgs_q_f               v9's usgs_q: same gauges, transform and correlation neighbours (computed on the same
                         2026-03-01..07-01 source period of the same snapshot); test 2026-09-10..10-13, 3-day episodes.
Targets: 48 per network, drawn (seed + 7) from the sensors that were never a test target of v9 (v9's 48 targets per
network, sorted(default_rng(9091 + 7).choice(N, 48)), are excluded). Episode construction as realdata_v16 / v9.
Data: fetch_v17.py snapshots (<data>/<name>.npz).
"""
import numpy as np
import realdata_v9
from realdata import NS, NQ, _episode

V9_SEED = 9091
NETS = {   # transform, source start, test start, test end, window days, support dropout, test targets, neighbours
    'dwd_t2m_f': ('raw', '2025-05-01', '2026-09-10', '2026-10-11', 7, .2, 48, 'distance'),
    'aurn_no2_f': ('log1p', '2025-01-01', '2026-09-10', '2026-10-11', 7, .2, 48, 'distance'),
    'usgs_q_f': ('log1p', '2026-03-01', '2026-09-10', '2026-10-13', 3, .2, 48, 'correlation'),
}
SOURCE_END = {'dwd_t2m_f': '2026-04-01', 'aurn_no2_f': '2026-04-01', 'usgs_q_f': '2026-07-01'}   # v9's source periods


def load(name, folder):
    tf, s0, _, t1, win, drop, k, nb = NETS[name]
    old = realdata_v9.NETS
    realdata_v9.NETS = {name: (tf, s0, SOURCE_END[name], t1, win, drop, k, nb)}
    try:
        return realdata_v9.load(name, folder)
    finally:
        realdata_v9.NETS = old


def new_targets(N, seed, k=48):
    old = set(int(i) for i in np.random.default_rng(V9_SEED + 7).choice(N, k, replace=False))
    cand = np.array([i for i in range(N) if i not in old])
    return sorted(int(i) for i in np.random.default_rng(seed + 7).choice(cand, k, replace=False)), sorted(old)


def episodes(name, raw, split, seed, n_source=None):
    assert split == 'test', 'protocol v17 only evaluates (UPT-SF keeps its v9 fine-tuning)'
    rng = np.random.default_rng(seed)
    _, _, t0, t1, win, drop, k, _ = NETS[name]
    V, ts, nbr = raw['V'], raw['ts'], raw['nbr']
    lo, hi = np.datetime64(t0), np.datetime64(t1)
    starts = np.arange(lo, hi, np.timedelta64(win, 'D')).astype('datetime64[s]')
    starts = starts[starts + np.timedelta64(win, 'D') <= hi]
    rows = {s: np.flatnonzero((ts >= s) & (ts < s + np.timedelta64(win, 'D'))) for s in starts}
    targets, _ = new_targets(V.shape[1], seed, k)
    out = []
    for s in starts:
        for tgt in targets:
            idx = rows[s][~np.isnan(V[rows[s], tgt])]
            if len(idx) >= NS + NQ:
                out.append(_episode(rng, V[:, nbr[tgt]], V[:, tgt], idx, drop) + (f'{str(s)[:10]}|{raw["ids"][tgt]}',))
    return out
