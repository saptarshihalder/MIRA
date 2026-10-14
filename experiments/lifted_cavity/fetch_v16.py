"""Protocol v16 data: the v10 snapshots, with the river networks extended to 2026-10-13 (UTC, exclusive).

    python fetch_v16.py --v10 <folder with the v10 snapshots> --out data_snapshots/v16

usgs_ny_f, usgs_wa_f: the v10 snapshot (same gauges, same order, same 30-minute UTC grid) plus 2026-10-01..10-13 from
the legacy NWIS instantaneous-value service, 20 gauges per request, 6 requests in parallel (parse_rdb and place of
fetch_v9; a gauge with no new data keeps NaN). dwd_p_f, aurn_pm25_f: the v10 snapshots unchanged (they already run to
2026-10-11). Writes <name>.npz (V, ts, ids, lat, lon) and <name>.json (provenance, content digest of V).
"""
import argparse, hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from fetch_v9 import UA, parse_rdb, place

IV = 'https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&parameterCd=00060&startDT=2026-09-30&endDT=2026-10-13&sites='
EXT = ('2026-10-01', '2026-10-13')
BASE = {'usgs_ny_f': 'usgs_ny', 'usgs_wa_f': 'usgs_wa', 'dwd_p_f': 'dwd_p', 'aurn_pm25_f': 'aurn_pm25'}


def digest(V):
    return hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest()[:16]


def fetch(chunk):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(IV + ','.join(chunk), headers=UA), timeout=120) as r:
                return parse_rdb(r.read().decode('utf-8', 'replace'))
        except Exception as e:
            err = e; time.sleep(5 * (attempt + 1))
    print('FAILED chunk', chunk[0], repr(err)[:120], flush=True)
    return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, base in BASE.items():
        d = np.load(Path(a.v10) / f'{base}.npz', allow_pickle=True)
        V, ts, ids = d['V'], d['ts'].astype('datetime64[s]'), [str(i) for i in d['ids']]
        meta = dict(base=base, base_digest=digest(V), base_rows=int(len(ts)))
        if name.startswith('usgs'):
            t2 = pd.date_range(*EXT, freq='30min', inclusive='left').values.astype('datetime64[s]')
            t0 = time.time()
            chunks = [ids[i:i + 20] for i in range(0, len(ids), 20)]
            with ThreadPoolExecutor(6) as ex:
                parts = list(ex.map(fetch, chunks))
            data = {}
            for p in parts:
                data.update(p)
            V2 = np.stack([place(t2, *data[s]) if s in data else np.full(len(t2), np.nan, 'float32') for s in ids], 1)
            V, ts = np.concatenate([V, V2.astype(V.dtype)]), np.concatenate([ts, t2])
            meta.update(extension=EXT, url=IV, gauges_with_new_data=int(sum(s in data for s in ids)), gauges=len(ids),
                        new_finite_frac=float(np.isfinite(V2).mean()), seconds=round(time.time() - t0, 1))
        np.savez_compressed(out / f'{name}.npz', V=V, ts=ts, ids=d['ids'], lat=d['lat'], lon=d['lon'])
        meta.update(rows=int(len(ts)), first=str(ts[0]), last=str(ts[-1]), digest=digest(V))
        (out / f'{name}.json').write_text(json.dumps(meta, indent=1))
        print(name, meta, flush=True)


if __name__ == '__main__':
    main()
