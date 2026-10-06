"""Protocol v2 evaluation: synthetic panels with every reference, and secondary metrics for Gaussian predictives.

python evaluate2.py panel --seed 20261201 --tasks 256 [--sensors 5 --nonlin .4 --support 48]
python evaluate2.py model --run runs/lift1_s1 --panel cache/v2_....pt
Metrics per (task, mask): NLL (primary); squared error, 90% central-interval coverage and CRPS for methods with a
Gaussian predictive (closed forms other than NL-FA, and all learned models). Mask banks: all patterns with k missing
sensors, or 20 seeded patterns when there are more.
"""
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
import data, family as fam, fa as fm, models, nlfa
from train import build, CACHE

KMISS = (0, 1, 2, 3)
MAXMASKS = 20


def banks(P, seed=0):
    out = {}
    for k in KMISS:
        if k >= P:
            continue
        combos = list(itertools.combinations(range(P), k))
        if len(combos) > MAXMASKS:
            rng = np.random.default_rng(seed + 1000 * k + P)
            combos = [combos[i] for i in sorted(rng.choice(len(combos), MAXMASKS, replace=False))]
        m = np.ones((len(combos), P), dtype='float32')
        for i, c in enumerate(combos):
            m[i, list(c)] = 0
        out[k] = m
    return out


def gauss_metrics(mu, var, y):
    sd = np.sqrt(var); z = (y - mu) / sd
    nll = .5 * (np.log(2 * np.pi) + np.log(var) + z * z)
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    return dict(nll=nll, se=(y - mu) ** 2, cov=(np.abs(z) <= 1.6448536269514722).astype(float), crps=crps)


def build_panel(a):
    pool = data.build_pool(a.tasks, a.seed, nonlin=a.nonlin, support_rows=a.support, keep_truth=True,
                           support_missing=a.support_missing, P=a.sensors)
    B = banks(a.sensors, a.seed)
    gauss = ('pop_linear', 'ridge_repo', 'ridge_cc', 'blr_cc', 'em_gauss', 'fa1')
    refs = {}
    for k, bank in B.items():
        cells = {n: {m: np.full((a.tasks, len(bank)), np.nan) for m in ('nll', 'se', 'cov', 'crps')} for n in gauss + ('oracle', 'nlfa')}
        for t in range(a.tasks):
            sx, sy, sm = (pool[k_][t].double().numpy() for k_ in ('sx', 'sy', 'sm'))
            qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
            mu, sig = fam.em_gaussian(sx, sy, sm, 40)
            f1 = fm.fit_fa(mu, sig, 1)
            nm = nlfa.fit(sx, sy, sm)
            n = len(sy)
            for i, msk in enumerate(bank):
                S = np.flatnonzero(msk)
                nll, pm, pv = fam.oracle(pool['prm'], t, S, pool['raw_qx'][t], qy)
                cells['oracle']['nll'][t, i] = nll.mean()
                cells['nlfa']['nll'][t, i] = nlfa.predict_nll(nm, qx, qy, S).mean()
                beta, b0, resid = fam.pop_linear(pool['prm'], t, S)
                preds = dict(pop_linear=((pool['raw_qx'][t][:, S] - b0) @ beta, np.full(len(qy), resid)),
                             ridge_repo=fam.ridge_repo(sx, sy, qx, S), ridge_cc=fam.ridge_cc(sx, sy, sm.astype(bool), qx, S),
                             blr_cc=fam.blr_cc(sx, sy, sm.astype(bool), qx, S), em_gauss=fam.em_conditional(mu, sig, qx, S, n),
                             fa1=fm.lifted_poe(f1, qx, S, n))
                for name, (pmu, pvar) in preds.items():
                    for m, v in gauss_metrics(pmu, pvar, qy).items():
                        cells[name][m][t, i] = v.mean()
        refs[k] = cells
    pool.pop('raw_qx')
    meta = dict(vars(a), masks={k: int(len(b)) for k, b in B.items()})
    path = Path(a.cache) / f'v2_s{a.seed}_n{a.tasks}_p{a.sensors}_nl{a.nonlin}_sr{a.support}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pool=pool, refs=refs, banks=B, meta=meta), path)
    print(path)
    print(json.dumps({k: {n: round(float(np.nanmean(v['nll'])), 4) for n, v in c.items()} for k, c in refs.items()}))


@torch.no_grad()
def score(model, pool, B):
    model.eval()
    out = {}
    for k, bank in B.items():
        acc = {m: [] for m in ('nll', 'se', 'cov', 'crps')}
        for tids, batch in data.eval_batches(pool, bank, tasks_per_batch=8):
            mu, lv = model(batch)
            g = gauss_metrics(mu.double().numpy(), np.exp(lv.double().numpy()), batch['qy'].double().numpy())
            for m, v in g.items():
                acc[m].append(v.reshape(len(tids), len(bank), -1).mean(-1))
        out[k] = {m: np.concatenate(v) for m, v in acc.items()}
    return out


def score_run(a):
    run = Path(a.run)
    meta = json.loads((run / 'train.json').read_text())
    model = build(meta['model'], meta.get('repo'), meta)
    model.load_state_dict(torch.load(run / 'model.pt', weights_only=True))
    panel = torch.load(a.panel, weights_only=False)
    cells = score(model, panel['pool'], panel['banks'])
    tag = Path(a.panel).stem
    np.savez_compressed(run / f'cells_{tag}.npz', **{f'k{k}_{m}': v for k, d in cells.items() for m, v in d.items()})
    print(run.name, tag, {f'k{k}': round(float(d['nll'].mean()), 4) for k, d in cells.items()}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['panel', 'model'])
    ap.add_argument('--seed', type=int); ap.add_argument('--tasks', type=int, default=256)
    ap.add_argument('--nonlin', type=float, default=.4); ap.add_argument('--support', type=int, default=48)
    ap.add_argument('--support-missing', type=float, default=.2); ap.add_argument('--sensors', type=int, default=5)
    ap.add_argument('--cache', default=str(CACHE))
    ap.add_argument('--run'); ap.add_argument('--panel')
    a = ap.parse_args()
    torch.set_num_threads(1)
    build_panel(a) if a.what == 'panel' else score_run(a)
