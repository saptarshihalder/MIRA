"""Evaluate trained models and closed-form references on a fixed fresh panel, by number of missing sensors.

python evaluate.py panel  --seed 20261007 --tasks 256 [--nonlin .4]      # builds panel + references (oracle etc.)
python evaluate.py model  --run runs/lift1_s1 --panel cache/panel_...pt   # scores one trained model
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import data, family as fam, fa as fm, models
from train import build, CACHE

KMISS = (0, 1, 2, 3)


def banks(P=5):
    return {k: fam.mask_bank(k, P) for k in KMISS if k < P}


def build_panel(args):
    pool = data.build_pool(args.tasks, args.seed, nonlin=args.nonlin, support_rows=args.support, keep_truth=True,
                           support_missing=args.support_missing, P=args.sensors)
    refs = {}
    for k, bank in banks(args.sensors).items():
        names = ('oracle', 'oracle_gauss', 'pop_linear', 'ridge_repo', 'ridge_cc', 'em_gauss', 'fa1')
        cells = {n: np.zeros((args.tasks, len(bank))) for n in names}
        for t in range(args.tasks):
            sx, sy, sm = (pool[k_][t].double().numpy() for k_ in ('sx', 'sy', 'sm'))
            qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
            mu, sig = fam.em_gaussian(sx, sy, sm, 40)
            f1 = fm.fit_fa(mu, sig, 1)
            n = len(sy)
            for i, m in enumerate(bank):
                S = np.flatnonzero(m)
                nll, pm, pv = fam.oracle(pool['prm'], t, S, pool['raw_qx'][t], qy)
                cells['oracle'][t, i] = nll.mean(); cells['oracle_gauss'][t, i] = fam.gnll(pm, pv, qy).mean()
                beta, b0, resid = fam.pop_linear(pool['prm'], t, S)
                cells['pop_linear'][t, i] = fam.gnll((pool['raw_qx'][t][:, S] - b0) @ beta, np.full(len(qy), resid), qy).mean()
                cells['ridge_repo'][t, i] = fam.gnll(*fam.ridge_repo(sx, sy, qx, S), qy).mean()
                cells['ridge_cc'][t, i] = fam.gnll(*fam.ridge_cc(sx, sy, sm.astype(bool), qx, S), qy).mean()
                cells['em_gauss'][t, i] = fam.gnll(*fam.em_conditional(mu, sig, qx, S, n), qy).mean()
                cells['fa1'][t, i] = fam.gnll(*fm.lifted_poe(f1, qx, S, n), qy).mean()
        refs[k] = cells
    pool.pop('raw_qx'); pool.pop('prm')
    path = Path(args.cache) / (f'panel_s{args.seed}_n{args.tasks}_nl{args.nonlin}_sr{args.support}_sm{args.support_missing}'
                               + (f'_p{args.sensors}' if args.sensors != 5 else '') + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pool=pool, refs=refs, meta=vars(args)), path)
    print(path)
    print(json.dumps({k: {n: float(v.mean()) for n, v in cells.items()} for k, cells in refs.items()}, indent=1))


@torch.no_grad()
def score(model, pool):
    model.eval()
    out = {}
    for k, bank in banks(pool['qx'].shape[-1]).items():
        cells = []
        for tids, batch in data.eval_batches(pool, bank, tasks_per_batch=8):
            mu, lv = model(batch)
            cells.append(models.gauss_nll(mu, lv, batch['qy']).reshape(len(tids), len(bank), -1).mean(-1).double().numpy())
        out[k] = np.concatenate(cells)
    return out


def score_run(args):
    run = Path(args.run)
    meta = json.loads((run / 'train.json').read_text())
    model = build(meta['model'], meta['repo'])
    model.load_state_dict(torch.load(run / 'model.pt', weights_only=True))
    panel = torch.load(args.panel, weights_only=False)
    cells = score(model, panel['pool'])
    tag = Path(args.panel).stem
    np.savez_compressed(run / f'cells_{tag}.npz', **{f'k{k}': v for k, v in cells.items()})
    summary = {f'k{k}': float(v.mean()) for k, v in cells.items()}
    (run / f'score_{tag}.json').write_text(json.dumps(summary, indent=1))
    print(run.name, json.dumps(summary))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['panel', 'model'])
    ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--tasks', type=int, default=256)
    ap.add_argument('--nonlin', type=float, default=.4)
    ap.add_argument('--support', type=int, default=48)
    ap.add_argument('--support-missing', type=float, default=.2)
    ap.add_argument('--sensors', type=int, default=5)
    ap.add_argument('--cache', default=str(CACHE))
    ap.add_argument('--run'); ap.add_argument('--panel')
    a = ap.parse_args()
    torch.set_num_threads(1)
    build_panel(a) if a.what == 'panel' else score_run(a)
