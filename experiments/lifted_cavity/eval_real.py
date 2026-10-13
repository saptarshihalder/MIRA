"""Evaluate closed forms and learned models on the fixed TEST episodes of a real dataset.

python eval_real.py build --dataset beijing --seed 2027 --out cache/real_beijing_s2027.pt
python eval_real.py score --panel cache/real_beijing_s2027.pt --runs runs/lift1_s1,runs_real/beijing/lift1_s1_ft,...
Conditions: extra dropped sensors per query on top of the natural mask (airq: 0-3, beijing: 0,3,6, gas: 0,4,8,12).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import realdata
from finetune_real import load_raw
from train import build

EXTRA = {'airq_co': (0, 1, 2, 3), 'airq_no2': (0, 1, 2, 3), 'beijing': (0, 3, 6), 'beijing_no2': (0, 3, 6), 'beijing_co': (0, 3, 6),
         'beijing_pm10': (0, 3, 6), 'beijing_so2': (0, 3, 6), 'beijing_o3': (0, 3, 6), 'gas': (0, 4, 8, 12),
         'metr_la': (0, 3, 6), 'pems_bay': (0, 3, 6), 'intel': (0, 3, 6), 'pems03': (0, 3, 6), 'pems08': (0, 3, 6),
         'air_cn': (0, 3, 6), 'dwd_t2m': (0, 3, 6), 'aurn_no2': (0, 3, 6), 'usgs_q': (0, 3, 6),
         'usgs_ny': (0, 3, 6), 'usgs_wa': (0, 3, 6), 'dwd_p': (0, 3, 6), 'aurn_pm25': (0, 3, 6),
         'usgs_ny_f': (0, 3, 6), 'usgs_wa_f': (0, 3, 6), 'dwd_p_f': (0, 3, 6), 'aurn_pm25_f': (0, 3, 6)}


def build_panel(a):
    raw = load_raw(a.dataset, dict(airq=a.airq, beijing=a.beijing, gas=a.gas, v6=a.v6))
    eps = realdata.episodes(a.dataset, raw, 'test', a.seed)
    pool = realdata.to_pool(eps)
    conds, refs = {}, {}
    t0 = time.time()
    for e in EXTRA[a.dataset]:
        qm = realdata.condition_masks(pool, e, a.seed + 100 + e)
        conds[e] = qm
        refs[e] = realdata.closed_forms(pool, qm)
    meta = dict(dataset=a.dataset, seed=a.seed, episodes=pool['n'], files=raw['files'], dequant_seed=raw.get('dequant_seed'), seconds=time.time() - t0,
                natural_query_missing_frac=float(1 - pool['qnat'].mean()))
    torch.save(dict(pool=pool, conds=conds, refs=refs, meta=meta), a.out)
    print(json.dumps(meta)[:400])
    for e in refs:
        print(f'extra={e}', {k: round(float(np.nanmean(v['nll'])), 4) for k, v in refs[e].items()})


def score_runs(a):
    panel = torch.load(a.panel, weights_only=False)
    tag = Path(a.panel).stem
    torch.set_num_threads(1)
    for run in [Path(r) for r in a.runs.split(',') if r]:
        meta = json.loads((run / 'train.json').read_text())
        model = build(meta['model'], meta.get('repo'), meta)
        model.load_state_dict(torch.load(run / 'model.pt', weights_only=True, map_location='cpu'))
        if a.site_float64:                                             # protocol v7 UPT64: double-precision sites
            model.site_dtype = torch.float64
        ft = meta.get('finetune', {})
        from finetune_real import set_lv_bound
        set_lv_bound(model, ft.get('lv_bound', 5.))                    # protocol v10 SF runs record lv_bound and yz
        pool = realdata.zpool(panel['pool']) if ft.get('yz') else panel['pool']
        import devutil
        model.to(devutil.pick(a.device))
        sfx = ('_f64' if a.site_float64 else '') + (f'_tta{a.tta}' if a.tta > 1 else '')
        if a.cf_k:                                                     # protocol v15: K-fold cross-fit only
            K, n = a.cf_k, panel['pool']['n']
            cf = realdata.crossfit_pool_k(panel['pool'], K)
            cf = realdata.zpool(cf) if ft.get('yz') else cf
            out = {}
            for e in panel['conds']:
                cqm = realdata.crossfit_query_masks_k(realdata.crossfit_masks(panel['pool'], e, realdata.CF_SEED + e), K)
                r = realdata.score_model(model, cf, cqm, tta=a.tta, per_query=True)
                for m in ('nll', 'mean', 'var'):
                    out[f'cf_e{e}_{m}'] = r[f'q_{m}'].reshape(n, -1)
            np.savez_compressed(run / f'pqk{K}_{tag}{sfx}.npz', **out)
            print(run, f'{K}-fold cross-fit written', flush=True)
            continue
        cells = {} if a.per_query else {e: realdata.score_model(model, pool, qm, tta=a.tta) for e, qm in panel['conds'].items()}
        if a.per_query:                                                # protocol v11
            pq, var = {}, {}
            cf = realdata.crossfit_pool(panel['pool'])
            cf = realdata.zpool(cf) if ft.get('yz') else cf
            for e, qm in panel['conds'].items():
                r = realdata.score_model(model, pool, qm, tta=a.tta, per_query=True)
                pq[f'e{e}_nll'], pq[f'e{e}_mean'], var[f'e{e}_var'] = r['q_nll'], r['q_mean'], r['q_var']
                cqm = realdata.crossfit_query_masks(realdata.crossfit_masks(panel['pool'], e, realdata.CF_SEED + e))
                r = realdata.score_model(model, cf, cqm, tta=a.tta, per_query=True)
                pq[f'cf_e{e}_nll'] = r['q_nll'].reshape(panel['pool']['n'], -1)
                pq[f'cf_e{e}_mean'] = r['q_mean'].reshape(panel['pool']['n'], -1)
                var[f'cf_e{e}_var'] = r['q_var'].reshape(panel['pool']['n'], -1)
            if a.pq_var:                                               # protocol v14: + variances, in pqv_; pq_ of
                np.savez_compressed(run / f'pqv_{tag}{sfx}.npz', **pq, **var)   # record is written only if missing
                if (run / f'pq_{tag}{sfx}.npz').exists():
                    old = np.load(run / f'pq_{tag}{sfx}.npz')
                    print(run, 'pq of record kept; max |mean - record| =',
                          max(float(np.nanmax(np.abs(old[k] - pq[k]))) for k in pq if k.endswith('_mean')), flush=True)
                    continue
            np.savez_compressed(run / f'pq_{tag}{sfx}.npz', **pq)
            print(run, 'per-query written', flush=True)
            continue
        np.savez_compressed(run / f'cells_{tag}{sfx}.npz', **{f'e{e}_{m}': v for e, d in cells.items() for m, v in d.items()})
        print(run, {f'e{e}': round(float(d['nll'].mean()), 4) for e, d in cells.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['build', 'score'])
    ap.add_argument('--dataset'); ap.add_argument('--seed', type=int, default=2027); ap.add_argument('--out')
    ap.add_argument('--panel'); ap.add_argument('--runs', default=''); ap.add_argument('--device', default='cpu')
    ap.add_argument('--per-query', action='store_true', help='protocol v11: also write per-query NLL and mean on the test '
                    'queries and on the two-fold cross-fit of the support (pq_<tag><sfx>.npz)')
    ap.add_argument('--cf-k', type=int, default=0, help='protocol v15: write only the K-fold cross-fit of the support '
                    '(per-query NLL, mean, variance) to pqk<K>_<tag><sfx>.npz')
    ap.add_argument('--pq-var', action='store_true', help='protocol v14 (with --per-query): also write the predictive '
                    'variances, to pqv_<tag><sfx>.npz; an existing pq_ file of record is not overwritten')
    ap.add_argument('--tta', type=int, default=1, help='protocol v9: test-time ensemble over this many feature-embedding '
                    'seeds (cells_<tag>_tta<k>.npz)')
    ap.add_argument('--site-float64', action='store_true', help='UPT: lifted-site algebra in float64 (cells_<tag>_f64.npz)')
    ap.add_argument('--airq', default=realdata.DEFAULT['airq'])
    ap.add_argument('--beijing', default=realdata.DEFAULT['beijing'])
    ap.add_argument('--gas', default=realdata.DEFAULT['gas'])
    ap.add_argument('--v6', default=realdata.DEFAULT['v6'])
    a = ap.parse_args()
    build_panel(a) if a.what == 'build' else score_runs(a)
