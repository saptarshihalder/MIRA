"""Protocol v17 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V17.md): the frozen v15 pool (P3) vs TabPFN-3.5 on every fresh,
post-release task of v16 (four v10 networks) and v17 (three v9 networks), pooled.

python confirm17.py --v10 .../mira_v10 --v9 .../mira_v9 --out <json>
Statistics as v9-v16 (per-episode metrics, paired intervals cluster-robust over network-windows; gain = TabPFN-3.5 -
pool), except that the combined endpoints G1-G2 use 97.5% intervals: v16 already looked at four of the seven
networks, and the second look pays with a Bonferroni split of the 5% level. Episodes on which TabPFN-3.5 is undefined
are excluded. Secondary: v17's three networks alone (95%), each network, the v13 rule, UPT-SF alone.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--v9', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
V10, V9 = Path(a.v10), Path(a.v9)
NETS = {'usgs_ny_f': (V10, 16161, 'usgs_ny'), 'usgs_wa_f': (V10, 16161, 'usgs_wa'), 'dwd_p_f': (V10, 16161, 'dwd_p'),
        'aurn_pm25_f': (V10, 16161, 'aurn_pm25'),
        'dwd_t2m_f': (V9, 17171, 'dwd_t2m'), 'aurn_no2_f': (V9, 17171, 'aurn_no2'), 'usgs_q_f': (V9, 17171, 'usgs_q')}
V17 = ('dwd_t2m_f', 'aurn_no2_f', 'usgs_q_f')
TAB, SEL, REF = 'tabpfn_v3_5', 'pool15_P3', 'pool15_v13ref'
Z95, Z975 = 1.959964, 2.241403


def cells(net, run, e, metric='nll'):
    root, seed, base = NETS[net]
    if run == 'upt':
        return np.mean([np.load(root / 'runs_real' / base / f'upt_L_sfm_s{s}_ft' / f'pq_real_{net}_s{seed}_tta8.npz')[f'e{e}_nll'].mean(1)
                        for s in (1, 2, 3)], 0)
    return np.load(root / 'runs_real' / net / run / f'cells_real_{net}_s{seed}.npz')[f'e{e}_{metric}'].astype(float)


def keys(nets):
    out = []
    for n in nets:
        root, seed, _ = NETS[n]
        out += [f'{n}|{k.split("|")[0]}' for k in torch.load(root / 'panels' / f'real_{n}_s{seed}.pt', weights_only=False)['pool']['keys']]
    return np.array(out)


def paired(base, new, clusters, z, margin=.01):
    ok = np.isfinite(base) & np.isfinite(new)
    d = (base - new)[ok]; c = clusters[ok]
    d = np.array([d[c == k].mean() for k in sorted(set(c))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - z * se, m + z * se
    return dict(gain=float(m), lo=float(lo), hi=float(hi), level=0.975 if z == Z975 else 0.95, n=int(len(d)), episodes=int(ok.sum()),
                excluded=int((~ok).sum()), passed=bool(m >= margin and lo > 0))


def compare(base, new, nets, e, z=Z95, metric='nll', margin=.01, defined_only=True):
    b, p = [], []
    for n in nets:
        ok = np.isfinite(cells(n, TAB, e)) if defined_only else True
        b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, new, e, metric), np.nan))
    return paired(np.concatenate(b), np.concatenate(p), keys(nets), z, margin)


ALL = tuple(NETS)
res = {}
res['G1_pool_vs_tabpfn35_nll_e0_all7'] = compare(TAB, SEL, ALL, 0, Z975)
res['G2_pool_vs_tabpfn35_nll_e3_all7'] = compare(TAB, SEL, ALL, 3, Z975)
res['G2_pool_vs_tabpfn35_nll_e6_all7'] = compare(TAB, SEL, ALL, 6, Z975)
sec = {}
for e in (0, 3, 6):
    sec[f'v17only|pool_vs_tabpfn35_nll_e{e}'] = compare(TAB, SEL, V17, e)
    sec[f'all7|pool_vs_v13rule_nll_e{e}'] = compare(REF, SEL, ALL, e, margin=0.)
    sec[f'all7|pool_vs_uptsf_nll_e{e}'] = compare('upt', SEL, ALL, e, defined_only=False)
    sec[f'all7|pool_vs_tabpfn35_se_e{e}'] = compare(TAB, SEL, ALL, e, metric='se', margin=0.)
    for n in V17:
        sec[f'{n}|pool_vs_tabpfn35_nll_e{e}'] = compare(TAB, SEL, (n,), e)
res['secondary'] = sec
desc = {}
for n in ALL:
    for e in (0, 3, 6):
        ok = np.isfinite(cells(n, TAB, e))
        desc[f'{n}|e{e}'] = dict(episodes=int(ok.size), tabpfn35_undefined=int((~ok).sum()),
                                 **{nm: round(float(cells(n, run, e)[ok].mean()), 4) for nm, run in
                                    (('tabpfn35', TAB), ('uptsf', 'upt'), ('pool_v13rule', REF), ('pool_P3', SEL))},
                                 w_P3=round(float(cells(n, SEL, e, 'w')[ok].mean()), 3))
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and 'gain' in v:
        print(f"{k:42s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] ({v['level']:.1%}) n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}")
for k, v in desc.items():
    print(k, v)
