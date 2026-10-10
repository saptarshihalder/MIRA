"""Protocol v6 sensor networks: in-context virtual sensing (predict one sensor from its 11 nearest neighbours).

  metr_la   METR-LA loop-detector speeds (Li et al. 2018): 207 sensors, 5-min, Mar-Jun 2012. Zero readings are
            missing (the standard convention); natural missingness in support and query rows.
  pems_bay  PEMS-BAY speeds (Li et al. 2018): 325 sensors, 5-min, Jan-Jun 2017. No natural missingness, so support
            rows get 20% simulated per-sensor dropout (as for Air Quality); queries are complete at extra = 0.
  intel     Intel Berkeley Research Lab motes (Bodik et al. 2004): temperature of 54 motes, readings averaged into
            5-min bins; readings outside [0, 50] C (failing motes) are missing. Natural missingness.

Episodes: one calendar day of one target sensor; inputs are the 11 sensors nearest to it by location (fixed per
target); 48 support + 48 query rows drawn from the day's rows where the target is observed (realdata._episode).
Chronological split (source before test). Test targets: all motes (intel) or a seeded subset of 24 sensors (traffic).
Keys are 'day|target', so intervals cluster over days. Sources: tsl's ProcessedDatasets mirror (traffic) and
db.csail.mit.edu/labdata (intel); file hashes are recorded with every panel.
"""
import gzip, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from realdata import _episode, NS, NQ

P_IN = 11
SPLIT = {'metr_la': ('2012-03-01', '2012-05-16', '2012-06-28'),       # source start, test start, test end
         'pems_bay': ('2017-01-01', '2017-05-01', '2017-07-01'),
         'intel': ('2004-02-28', '2004-03-12', '2004-03-25')}   # motes die after ~Mar 23 (availability only)
SUPPORT_DROPOUT = {'metr_la': 0., 'pems_bay': .2, 'intel': 0.}
N_TEST_TARGETS = {'metr_la': 24, 'pems_bay': 24, 'intel': None}
EXTRA = (0, 3, 6)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, folder):
    folder = Path(folder)
    if name in ('metr_la', 'pems_bay'):
        z = np.load(folder / f'{name}.npz')
        V = z['X'].astype(float)
        V[V <= 0] = np.nan
        ids = [str(i) for i in z['ids']]
        if name == 'metr_la':
            loc = pd.read_csv(folder / 'metr_la' / 'sensor_locations_la.csv')
            pos = loc.set_index(loc.sensor_id.astype(str))[['latitude', 'longitude']]
        else:
            loc = pd.read_csv(folder / 'pems_bay' / 'sensor_locations_bay.csv', header=None, names=['sensor_id', 'latitude', 'longitude'])
            pos = loc.set_index(loc.sensor_id.astype(str))[['latitude', 'longitude']]
        xy = pos.loc[ids].to_numpy(float)
        return dict(V=V, ts=z['ts'].astype('datetime64[s]'), xy=xy, ids=ids, files={f'{name}.npz': sha(folder / f'{name}.npz')})
    f = folder / 'intel_data.txt.gz'
    df = pd.read_csv(f, sep=r'\s+', names='date time epoch mote temp hum light volt'.split(), header=None, on_bad_lines='skip')
    loc = pd.read_csv(folder / 'mote_locs.txt', sep=r'\s+', names=['mote', 'x', 'y'], header=None)
    df = df[df.mote.isin(loc.mote) & df.temp.between(0, 50)]
    t = pd.to_datetime(df.date + ' ' + df.time, errors='coerce', format='mixed')
    df = df.assign(bin=t.dt.floor('5min')).dropna(subset=['bin'])
    W = df.groupby(['bin', 'mote']).temp.mean().unstack()
    full = pd.date_range(SPLIT['intel'][0], SPLIT['intel'][2], freq='5min', inclusive='left')
    W = W.reindex(index=full, columns=sorted(loc.mote))
    xy = loc.set_index('mote').loc[W.columns, ['x', 'y']].to_numpy(float)
    return dict(V=W.to_numpy(float), ts=W.index.values.astype('datetime64[s]'), xy=xy, ids=[str(m) for m in W.columns],
                files={'intel_data.txt.gz': sha(f), 'mote_locs.txt': sha(folder / 'mote_locs.txt')})


def neighbours(xy, tgt):
    d = ((xy - xy[tgt]) ** 2).sum(1)
    d[tgt] = np.inf
    return list(np.argsort(d, kind='stable')[:P_IN])


def episodes(name, raw, split, seed, n_source=None):
    rng = np.random.default_rng(seed)
    V, ts, xy = raw['V'], raw['ts'], raw['xy']
    s0, t0, t1 = (np.datetime64(x) for x in SPLIT[name])
    lo, hi = (s0, t0) if split == 'source' else (t0, t1)
    days = np.arange(lo, hi, np.timedelta64(1, 'D')).astype('datetime64[s]')
    day_of = ts.astype('datetime64[D]').astype('datetime64[s]')
    rows = {d: np.flatnonzero(day_of == d) for d in days}
    N = V.shape[1]
    out = []
    if split == 'test':
        k = N_TEST_TARGETS[name]
        targets = sorted(np.random.default_rng(seed + 7).choice(N, k, replace=False)) if k else range(N)
        for d in days:
            for tgt in targets:
                idx = rows[d][~np.isnan(V[rows[d], tgt])]
                if len(idx) >= NS + NQ:
                    out.append(_episode(rng, V[:, neighbours(xy, tgt)], V[:, tgt], idx, SUPPORT_DROPOUT[name])
                               + (f'{str(d)[:10]}|{raw["ids"][tgt]}',))
    else:
        while len(out) < n_source:
            d, tgt = days[rng.integers(len(days))], int(rng.integers(N))
            idx = rows[d][~np.isnan(V[rows[d], tgt])]
            if len(idx) >= NS + NQ:
                out.append(_episode(rng, V[:, neighbours(xy, tgt)], V[:, tgt], idx, SUPPORT_DROPOUT[name]) + ('src',))
    return out
