"""Protocol v10 sensor networks: four fresh networks, test periods in 2026. Data: fetch_v10.py snapshots
(<data>/<name>.npz). Episode construction is exactly realdata_v9's (48 support + 48 query rows, 20% simulated support
dropout, 48 seeded test targets, keys 'window-start|target').

  usgs_ny    New York stream gauges, 30-minute discharge, log1p(ft3/s); the 11 gauges most correlated over the source
             period; 3-day episodes; source 2026-03-01..07-01, test 2026-07-01..10-01.
  usgs_wa    the same for Washington State.
  dwd_p      German weather stations, hourly station-level pressure in hPa; the 11 nearest stations; weekly episodes;
             source 2025-05-01..2026-04-01, test 2026-04-01..10-01.
  aurn_pm25  UK AURN sites, hourly PM2.5, log1p(ug/m3) (negative readings are missing); the 11 nearest sites; weekly;
             source 2025-01-01..2026-04-01, test 2026-04-01..10-01.
"""
import realdata_v9

NETS = {   # transform, source start, test start, test end, window days, support dropout, test targets, neighbours
    'usgs_ny': ('log1p', '2026-03-01', '2026-07-01', '2026-10-01', 3, .2, 48, 'correlation'),
    'usgs_wa': ('log1p', '2026-03-01', '2026-07-01', '2026-10-01', 3, .2, 48, 'correlation'),
    'dwd_p': ('raw', '2025-05-01', '2026-04-01', '2026-10-01', 7, .2, 48, 'distance'),
    'aurn_pm25': ('log1p', '2025-01-01', '2026-04-01', '2026-10-01', 7, .2, 48, 'distance'),
}


def _with_nets(fn, *args, **kw):
    """Run a realdata_v9 function with this protocol's network table (same code path as v9)."""
    old = realdata_v9.NETS
    realdata_v9.NETS = NETS
    try:
        return fn(*args, **kw)
    finally:
        realdata_v9.NETS = old


def load(name, folder):
    return _with_nets(realdata_v9.load, name, folder)


def episodes(name, raw, split, seed, n_source=None):
    return _with_nets(realdata_v9.episodes, name, raw, split, seed, n_source)
