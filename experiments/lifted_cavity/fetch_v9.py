"""Protocol v9 data: three sensor networks from three new domains whose test periods lie in 2026, after the release of
every pretrained model compared here (TabPFN v2: January 2025).

    python fetch_v9.py --out <data_v9> [dwd_t2m aurn_no2 usgs_q]

Writes <out>/<name>.npz with V (T, N) float32 (NaN = missing; raw units), ts (T,) datetime64[s] UTC, ids, lat, lon (N,),
and <out>/<name>.json (source URLs, download time, station counts). The snapshot is frozen by the SHA-256 of the npz,
which every panel records (recent data are provisional and their publishers may revise them later).

  dwd_t2m   Deutscher Wetterdienst, hourly 2 m air temperature (TT_TU, deg C), 'recent' archive of the hourly station
            network (opendata.dwd.de, CC BY 4.0). -999 = missing.
  aurn_no2  UK Automatic Urban and Rural Network (Defra), hourly NO2 (ug/m3), 2025 and 2026 site files (openair R_data,
            Open Government Licence). Sites with an 'ongoing' NO2 record.
  usgs_q    USGS NWIS instantaneous stream discharge (parameter 00060, ft3/s), active Pennsylvania stream gauges,
            2026-03-01..2026-09-30, on a 30-minute UTC grid (values stamped at :00 and :30). Negative values = missing.

Stations are kept when at least 75% of the grid is observed in both the source and the test period (realdata_v9.NETS).
"""
import argparse, io, json, re, time, urllib.request, warnings, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0 (MIRA research data download)'}
DWD = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/'
AURN = 'https://uk-air.defra.gov.uk/openair/R_data/'
USGS_SITES = ('https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=pa&parameterCd=00060&siteType=ST'
              '&siteStatus=active&hasDataTypeCd=iv')
USGS_IV = 'https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&parameterCd=00060&startDT=2026-03-01&endDT=2026-09-30&sites='
GRID = {'dwd_t2m': ('2025-04-01', '2026-10-11', 'h'), 'aurn_no2': ('2025-01-01', '2026-10-11', 'h'),
        'usgs_q': ('2026-03-01', '2026-10-01', '30min')}
KEEP = .75


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def grid(name):
    a, b, f = GRID[name]
    return pd.date_range(a, b, freq=f, inclusive='left').values.astype('datetime64[s]')


def place(ts, t, v):
    """Values v at times t on the grid ts (exact matches only; duplicates keep the first)."""
    out = np.full(len(ts), np.nan, 'float32')
    i = np.searchsorted(ts, t)
    ok = (i < len(ts)) & (ts[np.minimum(i, len(ts) - 1)] == t) & np.isfinite(v)
    i, v = i[ok], v[ok]
    _, first = np.unique(i, return_index=True)
    out[i[first]] = v[first]
    return out


def dwd(ts):
    idx = get(DWD).decode('latin-1')
    zips = sorted(set(re.findall(r'stundenwerte_TU_\d+_akt\.zip', idx)))
    meta = {}
    for line in get(DWD + 'TU_Stundenwerte_Beschreibung_Stationen.txt').decode('latin-1').splitlines()[2:]:
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            meta[p[0].zfill(5)] = (float(p[4]), float(p[5]))

    def one(z):
        sid = z.split('_')[2]
        zf = zipfile.ZipFile(io.BytesIO(get(DWD + z)))
        df = pd.read_csv(zf.open([n for n in zf.namelist() if n.startswith('produkt')][0]), sep=';', skipinitialspace=True)
        t = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H').values.astype('datetime64[s]')
        v = np.array(df['TT_TU'], float); v[v <= -999] = np.nan
        return sid, place(ts, t, v)
    with ThreadPoolExecutor(16) as ex:
        cols = [c for c in ex.map(one, zips) if c[0] in meta]
    return [c[0] for c in cols], np.stack([c[1] for c in cols], 1), [meta[c[0]] for c in cols], [DWD]


def aurn(ts):
    import rdata
    warnings.filterwarnings('ignore', module='rdata')
    m = rdata.read_rda(io.BytesIO(get(AURN + 'AURN_metadata.RData')))['AURN_metadata']
    m = m[(m['parameter'] == 'NO2') & (m['end_date'] == 'ongoing')].drop_duplicates('site_id')
    sites = {str(r.site_id): (float(r.latitude), float(r.longitude)) for r in m.itertuples()}

    def one(sid):
        parts = []
        for y in (2025, 2026):
            try:
                d = rdata.read_rda(io.BytesIO(get(f'{AURN}{sid}_{y}.RData')))[f'{sid}_{y}']
            except Exception:
                continue
            if 'NO2' in d.columns:
                parts.append((np.asarray(d['date'], float).astype('int64').astype('datetime64[s]'), np.asarray(d['NO2'], float)))
        if not parts:
            return sid, None
        return sid, place(ts, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    with ThreadPoolExecutor(16) as ex:
        cols = [c for c in ex.map(one, sorted(sites)) if c[1] is not None]
    return [c[0] for c in cols], np.stack([c[1] for c in cols], 1), [sites[c[0]] for c in cols], [AURN]


def parse_rdb(text):
    """Site blocks of an NWIS instantaneous-value RDB file -> {site: (UTC times, values)}; first 00060 series per site."""
    out, cols = {}, None
    for line in text.splitlines():
        if not line or line[0] == '#':
            cols = None if line.startswith('#') else cols
            continue
        f = line.split('\t')
        if f[0] == 'agency_cd':
            cols = f; vi = next(i for i, c in enumerate(f) if c.endswith('_00060')); continue
        if cols is None or f[0] != 'USGS' or len(f) <= vi:
            continue
        off = {'EST': 5, 'EDT': 4, 'CST': 6, 'CDT': 5, 'MST': 7, 'MDT': 6, 'PST': 8, 'PDT': 7, 'AKST': 9, 'AKDT': 8,
               'HST': 10, 'UTC': 0}.get(f[3])                 # hours behind UTC (v10 amendment 1: all US zones)
        if off is None:
            continue
        try:
            v = float(f[vi])
        except ValueError:
            continue
        t = np.datetime64(f[2].replace(' ', 'T')) + np.timedelta64(off, 'h')
        out.setdefault(f[1], ([], []))
        out[f[1]][0].append(t); out[f[1]][1].append(v if v >= 0 else np.nan)
    return {k: (np.array(t, 'datetime64[s]'), np.array(v)) for k, (t, v) in out.items()}


def usgs(ts):
    rows = [l.split('\t') for l in get(USGS_SITES).decode().splitlines() if l.startswith('USGS\t')]
    sites = {r[1]: (float(r[4]), float(r[5])) for r in rows}
    ids = sorted(sites)

    def one(site):                                  # one gauge per request; a gauge the service cannot serve is skipped
        try:
            return parse_rdb(get(USGS_IV + site, tries=3).decode())
        except Exception:
            return {}
    data = {}
    with ThreadPoolExecutor(8) as ex:
        for d in ex.map(one, ids):
            data.update(d)
    keep = [s for s in ids if s in data and len(data[s][0])]
    return keep, np.stack([place(ts, *data[s]) for s in keep], 1), [sites[s] for s in keep], [USGS_SITES, USGS_IV]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('names', nargs='*', default=list(GRID))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import realdata_v9
    for name in a.names:
        if (out / f'{name}.npz').exists():
            print(name, 'exists'); continue
        t0 = time.time(); ts = grid(name)
        ids, V, ll, urls = dict(dwd_t2m=dwd, aurn_no2=aurn, usgs_q=usgs)[name](ts)
        n0 = len(ids)
        _, s0, t1, t2 = realdata_v9.NETS[name][:4]
        cov = [np.isfinite(V[(ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))]).mean(0) for lo, hi in ((s0, t1), (t1, t2))]
        k = (cov[0] >= KEEP) & (cov[1] >= KEEP)
        ll = np.array(ll)[k]
        np.savez_compressed(out / f'{name}.npz', V=V[:, k], ts=ts, ids=np.array(ids)[k], lat=ll[:, 0], lon=ll[:, 1])
        info = dict(name=name, urls=urls, downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stations_found=n0,
                    stations_kept=int(k.sum()), observed_frac=float(np.isfinite(V[:, k]).mean()), seconds=round(time.time() - t0))
        (out / f'{name}.json').write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
