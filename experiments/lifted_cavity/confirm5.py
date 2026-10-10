"""Protocol v5 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V5.md), computed mechanically from replayed per-query predictions.

python site_pool.py predict --panel ../../cache/real_beijing_<p>_s4041.pt --runs <runs_real/beijing_<p>/{pfn_s1_ft,lct_s1_ft[,pfn_s2_ft]}>
python confirm5.py
"""
import argparse, json
from pathlib import Path
import numpy as np
from site_pool import episode_nll, pooled, lin_pool_nll

NEW = ('beijing_pm10', 'beijing_so2', 'beijing_o3')
ap = argparse.ArgumentParser()
ap.add_argument('--preds', default='../../cache/site_pool')
ap.add_argument('--out', default='../../artifacts/reports/lifted_cavity_paper/runs/confirm_v5.json')
a = ap.parse_args()


def paired(base, new, clusters, margin=.01, noninf=None):
    """95% decision (protocol v5) and 97.5% decision (amendment 1: Bonferroni over the two pooling rules)."""
    d = base - new
    if len(clusters) != len(d):
        raise ValueError('Episode and cluster counts differ.')
    d = np.array([d[clusters == k].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    out = dict(gain=float(m), n=int(len(d)))
    for tag, z in (('', 1.959963984540054), ('_975', 2.241402727604947)):
        lo, hi = m - z * se, m + z * se
        out.update({f'lo{tag}': float(lo), f'hi{tag}': float(hi),
                    f'passed{tag}': bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)})
    return out


def collect(e):
    """Per-episode NLL over the three new pollutants (concatenated) and week keys; None where a run is missing."""
    out, weeks = {}, []
    for p in NEW:
        P = dict(np.load(Path(a.preds) / f'preds_real_{p}_s4041.npz'))
        weeks.append(P['weeks'])
        have = lambda r: f'{r}|e{e}|mu' in P
        for name, fn in [('pfn', lambda: episode_nll(P, 'pfn_s1_ft', e)), ('lct', lambda: episode_nll(P, 'lct_s1_ft', e)),
                         ('pool', lambda: pooled(P, 'lct_s1_ft', 'pfn_s1_ft', e)[0]),
                         ('slack', lambda: pooled(P, 'lct_s1_ft', 'pfn_s1_ft', e)[1]),
                         ('lin', lambda: pooled(P, 'lct_s1_ft', 'pfn_s1_ft', e, 'lin')[0]),
                         ('pool_pfn2', lambda: pooled(P, 'pfn_s1_ft', 'pfn_s2_ft', e)[0] if have('pfn_s2_ft') else None),
                         ('lin_pfn2', lambda: pooled(P, 'pfn_s1_ft', 'pfn_s2_ft', e, 'lin')[0] if have('pfn_s2_ft') else None)]:
            out.setdefault(name, []).append(fn())
        out.setdefault('pollutant', []).append(np.full(len(P['weeks']), p))
    cat = {k: (np.concatenate(v) if all(x is not None for x in v) else None) for k, v in out.items()}
    return cat, np.concatenate(weeks)


res = {}
c, weeks = collect(0)
res['E13_bjnew_pool_noninf_pfnft'] = paired(c['pfn'], c['pool'], weeks, noninf=-.02)
res['E14_bjnew_pool_vs_pfnft'] = paired(c['pfn'], c['pool'], weeks)
res['E15_bjnew_pool_vs_lctft'] = paired(c['lct'], c['pool'], weeks)
res['E16_bjnew_pool_vs_pfnpool'] = (paired(c['pool_pfn2'], c['pool'], weeks) if c['pool_pfn2'] is not None
                                    else dict(status='not run'))
res['E13L_bjnew_linpool_noninf_pfnft'] = paired(c['pfn'], c['lin'], weeks, noninf=-.02)
res['E14L_bjnew_linpool_vs_pfnft'] = paired(c['pfn'], c['lin'], weeks)
res['E15L_bjnew_linpool_vs_lctft'] = paired(c['lct'], c['lin'], weeks)
res['E16L_bjnew_linpool_vs_pfnlinpool'] = (paired(c['lin_pfn2'], c['lin'], weeks) if c['lin_pfn2'] is not None
                                           else dict(status='not run'))
desc = {}
for e in (0, 3, 6):
    ce, wk = collect(e)
    desc[f'e{e}'] = {k: float(ce[k].mean()) for k in ('pfn', 'lct', 'pool', 'lin', 'slack', 'pool_pfn2', 'lin_pfn2') if ce[k] is not None}
    for p in NEW:
        s = ce['pollutant'] == p
        desc[f'e{e}|{p}'] = {k: float(ce[k][s].mean()) for k in ('pfn', 'lct', 'pool', 'lin', 'slack', 'pool_pfn2', 'lin_pfn2') if ce[k] is not None}
    desc[f'e{e}|pool_vs_pfn'] = paired(ce['pfn'], ce['pool'], wk)
    desc[f'e{e}|linpool_vs_pfn'] = paired(ce['pfn'], ce['lin'], wk)
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if k.startswith('E'):
        print(f"{k:34s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             f" | 97.5% [{v['lo_975']:+.4f}, {v['hi_975']:+.4f}] -> {'PASS' if v['passed_975'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, {m: round(x, 4) for m, x in v.items()} if 'gain' not in v else f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
