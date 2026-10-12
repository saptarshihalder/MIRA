"""Protocol v10 data: four fresh sensor networks (test periods in 2026), same snapshot format as fetch_v9.py.

    python fetch_v10.py --out <data_v10> [usgs_ny usgs_wa dwd_p aurn_pm25]

  usgs_ny    USGS NWIS discharge (00060), active New York stream gauges, 2026-03-01..2026-09-30, 30-minute UTC grid.
  usgs_wa    the same for Washington State (snowmelt- and glacier-fed rivers).
  dwd_p      Deutscher Wetterdienst hourly station-level air pressure (P0, hPa), 'recent' archive; -999 = missing.
  aurn_pm25  UK AURN hourly PM2.5 (ug/m3), 2025 and 2026 site files; sites with an 'ongoing' PM2.5 record.
Stations are kept when at least 75% of the grid is observed in both the source and the test period (realdata_v10.NETS).
"""
import argparse, io, json, re, time, warnings, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from fetch_v9 import get, place, parse_rdb, AURN, KEEP

DWD_P = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/pressure/recent/'
USGS_SITES = ('https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={}&parameterCd=00060&siteType=ST'
              '&siteStatus=active&hasDataTypeCd=iv')
USGS_IV = 'https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&parameterCd=00060&startDT=2026-03-01&endDT=2026-09-30&sites='
USGS_OGC = ('https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items?f=csv&limit=50000&monitoring_location_id=USGS-{site}'
            '&parameter_code=00060&time={a}T00:00:00Z/{b}T00:00:00Z&properties=time,value')
GRID = {'usgs_ny': ('2026-03-01', '2026-10-01', '30min'), 'usgs_wa': ('2026-03-01', '2026-10-01', '30min'),
        'dwd_p': ('2025-04-01', '2026-10-11', 'h'), 'aurn_pm25': ('2025-01-01', '2026-10-11', 'h')}


def grid(name):
    a, b, f = GRID[name]
    return pd.date_range(a, b, freq=f, inclusive='left').values.astype('datetime64[s]')


def dwd_pressure(ts):
    idx = get(DWD_P).decode('latin-1')
    zips = sorted(set(re.findall(r'stundenwerte_P0_\d+_akt\.zip', idx)))
    meta = {}
    for line in get(DWD_P + 'P0_Stundenwerte_Beschreibung_Stationen.txt').decode('latin-1').splitlines()[2:]:
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            meta[p[0].zfill(5)] = (float(p[4]), float(p[5]))

    def one(z):
        sid = z.split('_')[2]
        zf = zipfile.ZipFile(io.BytesIO(get(DWD_P + z)))
        df = pd.read_csv(zf.open([n for n in zf.namelist() if n.startswith('produkt')][0]), sep=';', skipinitialspace=True)
        t = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H').values.astype('datetime64[s]')
        v = np.array(df['P0'], float); v[v <= -999] = np.nan
        return sid, place(ts, t, v)
    with ThreadPoolExecutor(16) as ex:
        cols = [c for c in ex.map(one, zips) if c[0] in meta]
    return [c[0] for c in cols], np.stack([c[1] for c in cols], 1), [meta[c[0]] for c in cols], [DWD_P]


def aurn(ts, param):
    import rdata
    warnings.filterwarnings('ignore', module='rdata')
    m = rdata.read_rda(io.BytesIO(get(AURN + 'AURN_metadata.RData')))['AURN_metadata']
    m = m[(m['parameter'] == param) & (m['end_date'] == 'ongoing')].drop_duplicates('site_id')
    sites = {str(r.site_id): (float(r.latitude), float(r.longitude)) for r in m.itertuples()}

    def one(sid):
        parts = []
        for y in (2025, 2026):
            try:
                d = rdata.read_rda(io.BytesIO(get(f'{AURN}{sid}_{y}.RData')))[f'{sid}_{y}']
            except Exception:
                continue
            if param in d.columns:
                parts.append((np.asarray(d['date'], float).astype('int64').astype('datetime64[s]'), np.asarray(d[param], float)))
        if not parts:
            return sid, None
        return sid, place(ts, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    with ThreadPoolExecutor(16) as ex:
        cols = [c for c in ex.map(one, sorted(sites)) if c[1] is not None]
    return [c[0] for c in cols], np.stack([c[1] for c in cols], 1), [sites[c[0]] for c in cols], [AURN]


def usgs(ts, state):
    rows = [l.split('\t') for l in get(USGS_SITES.format(state)).decode().splitlines() if l.startswith('USGS\t')]
    sites = {r[1]: (float(r[4]), float(r[5])) for r in rows}
    ids = sorted(sites)

    def one(site):          # v10 amendment 2: legacy NWIS first, then the USGS Water Data OGC API (same data, UTC)
        for attempt in range(3):
            try:
                return site, parse_rdb(get(USGS_IV + site, tries=1).decode())
            except Exception:
                pass
            try:               # one request for the whole period (the API allows 1,000 requests per hour per address);
                df = pd.read_csv(io.StringIO(get(USGS_OGC.format(site=site, a='2026-03-01', b='2026-10-01'), tries=2).decode()))
                if len(df) >= 50000:                       # a 5-minute record hit the page limit: fetch the rest
                    rest = pd.read_csv(io.StringIO(get(USGS_OGC.format(site=site, a='2026-07-01', b='2026-10-01'), tries=2).decode()))
                    df = pd.concat([df, rest])
                if len(df):
                    t = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None).values.astype('datetime64[s]')
                    v = pd.to_numeric(df['value'], errors='coerce').to_numpy(float).copy()
                    v[v < 0] = np.nan
                    return site, {site: (t, v)}
                return site, {}
            except Exception:
                time.sleep(15 * (attempt + 1))
        return site, None
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(one, ids))
    for site in [k for k, v in res.items() if v is None]:   # second, sequential pass for the gauges that failed
        res[site] = one(site)[1]
    FAILED[0] = sum(v is None for v in res.values())
    data = {}
    for v in res.values():
        data.update(v or {})
    keep = [s for s in ids if s in data and len(data[s][0])]
    return keep, np.stack([place(ts, *data[s]) for s in keep], 1), [sites[s] for s in keep], [USGS_SITES.format(state), USGS_IV]


FAILED = [0]
FETCH = {'usgs_ny': lambda ts: usgs(ts, 'ny'), 'usgs_wa': lambda ts: usgs(ts, 'wa'), 'dwd_p': dwd_pressure,
         'aurn_pm25': lambda ts: aurn(ts, 'PM2.5')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('names', nargs='*', default=list(GRID))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import realdata_v10
    for name in a.names:
        if (out / f'{name}.npz').exists():
            print(name, 'exists'); continue
        t0 = time.time(); ts = grid(name)
        ids, V, ll, urls = FETCH[name](ts)
        n0 = len(ids)
        _, s0, t1, t2 = realdata_v10.NETS[name][:4]
        cov = [np.isfinite(V[(ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))]).mean(0) for lo, hi in ((s0, t1), (t1, t2))]
        k = (cov[0] >= KEEP) & (cov[1] >= KEEP)
        ll = np.array(ll)[k]
        np.savez_compressed(out / f'{name}.npz', V=V[:, k], ts=ts, ids=np.array(ids)[k], lat=ll[:, 0], lon=ll[:, 1])
        info = dict(name=name, urls=urls, downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stations_found=n0,
                    stations_kept=int(k.sum()), failed_requests=FAILED[0], observed_frac=float(np.isfinite(V[:, k]).mean()), seconds=round(time.time() - t0))
        (out / f'{name}.json').write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
