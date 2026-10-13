"""Protocol v16 networks: the four v10 networks on days after TabPFN-3.5's release (2026-09-09), with new targets.

  usgs_ny_f, usgs_wa_f   v10's usgs_ny / usgs_wa: same gauges, transform and correlation neighbours (computed on the
                         same 2026-03-01..07-01 source period of the same snapshot); test 2026-09-10..10-13, 3-day episodes.
  dwd_p_f, aurn_pm25_f   v10's dwd_p / aurn_pm25: same stations, transform and distance neighbours; test
                         2026-09-10..10-11, weekly episodes.
Targets: 48 per network, drawn (seed + 7) from the sensors that were never a test target of v10 (v10's 48 targets per
network, sorted(default_rng(10101 + 7).choice(N, 48)), are excluded), so every episode is a new prediction task.
Everything else is realdata_v9's episode construction (48 support + 48 query rows, 20% support dropout).
Data: fetch_v16.py snapshots (<data>/<name>.npz).
"""
import numpy as np
import realdata_v9
from realdata import NS, NQ, _episode

V10_SEED = 10101
NETS = {   # transform, source start, test start, test end, window days, support dropout, test targets, neighbours
    'usgs_ny_f': ('log1p', '2026-03-01', '2026-09-10', '2026-10-13', 3, .2, 48, 'correlation'),
    'usgs_wa_f': ('log1p', '2026-03-01', '2026-09-10', '2026-10-13', 3, .2, 48, 'correlation'),
    'dwd_p_f': ('raw', '2025-05-01', '2026-09-10', '2026-10-11', 7, .2, 48, 'distance'),
    'aurn_pm25_f': ('log1p', '2025-01-01', '2026-09-10', '2026-10-11', 7, .2, 48, 'distance'),
}
# v10's source periods end where v10's test periods began; the neighbour selection must use exactly v10's source period
SOURCE_END = {'usgs_ny_f': '2026-07-01', 'usgs_wa_f': '2026-07-01', 'dwd_p_f': '2026-04-01', 'aurn_pm25_f': '2026-04-01'}


def load(name, folder):
    """realdata_v9.load with v10's source period for the neighbours (the table's third field is v10's test start)."""
    tf, s0, _, t1, win, drop, k, nb = NETS[name]
    old = realdata_v9.NETS
    realdata_v9.NETS = {name: (tf, s0, SOURCE_END[name], t1, win, drop, k, nb)}
    try:
        return realdata_v9.load(name, folder)
    finally:
        realdata_v9.NETS = old


def new_targets(N, seed, k=48):
    old = set(int(i) for i in np.random.default_rng(V10_SEED + 7).choice(N, k, replace=False))
    cand = np.array([i for i in range(N) if i not in old])
    return sorted(int(i) for i in np.random.default_rng(seed + 7).choice(cand, k, replace=False)), sorted(old)


def episodes(name, raw, split, seed, n_source=None):
    assert split == 'test', 'protocol v16 only evaluates (UPT-SF keeps its v10 fine-tuning)'
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
