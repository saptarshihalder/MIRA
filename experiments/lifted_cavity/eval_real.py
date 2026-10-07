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
         'beijing_pm10': (0, 3, 6), 'beijing_so2': (0, 3, 6), 'beijing_o3': (0, 3, 6), 'gas': (0, 4, 8, 12)}


def build_panel(a):
    raw = load_raw(a.dataset, dict(airq=a.airq, beijing=a.beijing, gas=a.gas))
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
        import devutil
        model.to(devutil.pick(a.device))
        cells = {e: realdata.score_model(model, panel['pool'], qm) for e, qm in panel['conds'].items()}
        np.savez_compressed(run / f'cells_{tag}.npz', **{f'e{e}_{m}': v for e, d in cells.items() for m, v in d.items()})
        print(run, {f'e{e}': round(float(d['nll'].mean()), 4) for e, d in cells.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['build', 'score'])
    ap.add_argument('--dataset'); ap.add_argument('--seed', type=int, default=2027); ap.add_argument('--out')
    ap.add_argument('--panel'); ap.add_argument('--runs', default=''); ap.add_argument('--device', default='cpu')
    ap.add_argument('--airq', default=realdata.DEFAULT['airq'])
    ap.add_argument('--beijing', default=realdata.DEFAULT['beijing'])
    ap.add_argument('--gas', default=realdata.DEFAULT['gas'])
    a = ap.parse_args()
    build_panel(a) if a.what == 'build' else score_runs(a)
