"""Protocol v14: TabPFN-3.5 combined with UPT-SF beyond a linear pool.

    python tabpfn_feat.py --root <runs root> --net dwd_t2m --panel-seed 9091 --variants alone,uptsite [--stride 2]
    python tabpfn_feat.py --root <runs root> --net dwd_t2m --panel-seed 9091 --variants uptcol,uptcol2,uptres [--stride 2]

UPT-SF (`upt_L_sfm`, tta8) enters through its per-query predictive mean m and variance v (pqv_<tag>_tta8.npz,
eval_real.py --per-query --pq-var), seed-averaged as the equal mixture of seeds 1-3. Support rows always use UPT-SF's
two-fold cross-fitted predictive (each half predicted from the other half under the shared cross-fit masks of v11), so
no row's own label enters what UPT-SF says about it. Variants:
- alone    TabPFN-3.5 alone, the calls of tabpfn_colab.py (one main call for all conditions; two cross-fit calls).
- uptcol   (F1) + one input column: m.
- uptcol2  (F3) + two input columns: m and log sd(v), so TabPFN can learn when to trust UPT-SF.
- uptres   (F4) TabPFN models the residual y - m (columns m, log sd); the predictive of y is m + residual.
- uptsite  (F5) UPT-SF as a Gaussian site on TabPFN-3.5's predictive: within each bar of TabPFN's bar distribution,
           p(y) is proportional to p_TabPFN(y)^a N(y; m, v)^b (bar centres; TabPFN's half-normal tails kept), i.e. the
           bar logits shift by (a - 1) log density + b log N. (a, b) is chosen per episode and condition on the grid
           A x B by the summed log score of the two-fold cross-fit of the support (TabPFN's cross-fit calls of
           tabpfn_colab.py, UPT-SF's cross-fitted m, v); (1, 0) is TabPFN alone and wins ties.
Everything else as tabpfn_colab.py: 8 estimators, random state 0, NaN for missing inputs, constant-support episodes
undefined (all variants), CRPS and 90% coverage from quantiles at (i + .5) / 100 and .05, .95.
--stride k scores episodes 0, k, 2k, ... only (development); the others are NaN and flagged 'skipped'.
Outputs: runs_real/<net>/tabpfn_<version>_<variant>[_s<k>]/cells_<tag>.npz (+ pq_<tag>.npz for alone).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import realdata
from tabpfn_colab import N_ESTIMATORS, QN, QUANT, RANDOM_STATE, make_regressor, tabpfn_nll, task_arrays

SEEDS = (1, 2, 3)
A_GRID = (1.0, 0.8, 0.6)
B_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
COLUMN = ('uptcol', 'uptcol2', 'uptres')
SITE = ('alone', 'uptsite')


def upt_moments(R, model, tag, sfx, E):
    """Seed-averaged UPT-SF predictive mean and variance (the equal mixture of the seeds), test and cross-fit."""
    Z = [np.load(R / f'{model}_s{s}_ft' / f'pqv_{tag}{sfx}.npz') for s in SEEDS]
    out = {}
    for p in ('', 'cf_'):
        for e in E:
            m = np.mean([z[f'{p}e{e}_mean'] for z in Z], 0)
            v = np.mean([z[f'{p}e{e}_var'] + z[f'{p}e{e}_mean'] ** 2 for z in Z], 0) - m ** 2
            floor = (1e-4 * (1 + np.abs(m))) ** 2
            out[f'{p}e{e}'] = (m, np.maximum(v, floor))
    return out


def full_predict(reg, Xs, ys, Xq):
    """Fit in context; return TabPFN's raw-unit criterion and float64 logits for the query rows."""
    reg.fit(Xs, ys)
    out = reg.predict(Xq, output_type='full')
    crit, logits = out['criterion'], torch.as_tensor(out['logits'])
    crit = crit.to(logits.device)
    logits = logits.to(dtype=crit.borders.dtype)
    m = crit.mean(logits).double().cpu().numpy().reshape(-1)
    mean = np.asarray(out['mean'], dtype=float)
    if not np.allclose(m, mean, atol=1e-3 * (1 + np.abs(mean).max())):
        raise RuntimeError('TabPFN criterion is not in raw target units')
    return crit, logits


def centres(crit):
    c = (crit.borders[:-1] + crit.bucket_widths / 2).clone()
    hn = (crit.halfnormal_with_p_weight_before(crit.bucket_widths[0]), crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1]))
    c[0] = -hn[0].mean + crit.borders[1]; c[-1] = hn[1].mean + crit.borders[-2]
    return c


def fused(crit, logits, m, v, a, b):
    """Bar logits of p_TabPFN^a N(m, v)^b (a = 1, b = 0: TabPFN unchanged)."""
    if a == 1.0 and b == 0.0:
        return logits
    lw = torch.log(crit.bucket_widths)
    lp = torch.log_softmax(logits, -1)
    out = a * (lp - lw) + lw
    if b:
        c = centres(crit)
        mt = torch.as_tensor(m, dtype=logits.dtype, device=logits.device)[:, None]
        vt = torch.as_tensor(v, dtype=logits.dtype, device=logits.device)[:, None]
        out = out + b * (-0.5 * torch.log(2 * np.pi * vt) - (c[None] - mt) ** 2 / (2 * vt))
    return out


def scores(crit, logits, y):
    """Per-row NLL, mean, CRPS, 90% coverage of the bar distribution `logits` at y."""
    with torch.no_grad():
        yt = torch.as_tensor(y, dtype=logits.dtype, device=logits.device)
        nll = crit(logits, yt).double().cpu().numpy().reshape(-1)
        mean = crit.mean(logits).double().cpu().numpy().reshape(-1)
        Q = torch.stack([crit.icdf(logits, q) for q in QUANT], -1).double().cpu().numpy()
    u, p = np.asarray(y, float)[:, None] - Q[:, :QN], np.array(QUANT[:QN])
    crps = 2 * (u * (p - (u < 0))).mean(1)
    cov = ((y >= Q[:, QN]) & (y <= Q[:, QN + 1])).astype(float)
    return nll, mean, crps, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--net', required=True); ap.add_argument('--panel-seed', type=int, required=True)
    ap.add_argument('--variants', required=True, help='comma list of ' + ', '.join(SITE + COLUMN))
    ap.add_argument('--version', default='v3.5'); ap.add_argument('--model', default='upt_L_sfm'); ap.add_argument('--sfx', default='_tta8')
    ap.add_argument('--stride', type=int, default=1); ap.add_argument('--limit', type=int, default=None, help='smoke test: first LIMIT episodes')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    V = a.variants.split(',')
    assert set(V) <= set(SITE + COLUMN), V
    R = Path(a.root) / 'runs_real' / a.net
    tag = f'real_{a.net}_s{a.panel_seed}'
    panel = torch.load(Path(a.root) / 'panels' / f'{tag}.pt', weights_only=False); pool = panel['pool']
    E = list(panel['conds'])
    U = upt_moments(R, a.model, tag, a.sfx, E)
    NS, CF = realdata.NS, realdata.CF
    cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
    n = pool['n'] if a.limit is None else min(a.limit, pool['n'])
    vname = 'tabpfn_' + a.version.replace('.', '_')
    dirs = {v: R / (f'{vname}_{v}' if v != 'alone' else vname) for v in V}
    if a.stride > 1:
        dirs = {v: d.with_name(d.name + f'_s{a.stride}') for v, d in dirs.items()}
    stem = f'cells_{tag}' if a.limit is None else f'partial_cells_{tag}_limit{a.limit}'
    if all((d / f'{stem}.npz').exists() for d in dirs.values()):
        print('done already', flush=True); return
    reg = make_regressor(a.device, a.version)
    M = ('nll', 'se', 'crps', 'cov')
    res = {v: {f'e{e}_{m}': np.full(pool['n'], np.nan) for e in E for m in M} for v in V}
    if 'uptsite' in V:
        for e in E:
            res['uptsite'][f'e{e}_a'] = np.full(pool['n'], np.nan); res['uptsite'][f'e{e}_b'] = np.full(pool['n'], np.nan)
    pq = {f'{p}e{e}_{m}': np.full((pool['n'], NS), np.nan) for p in ('', 'cf_') for e in E for m in ('nll', 'mean')} if 'alone' in V else None
    und = np.zeros(pool['n'], bool); skipped = np.ones(pool['n'], bool)
    t0, calls = time.time(), 0
    for t in range(n):
        if t % a.stride:
            continue
        skipped[t] = False
        Xs, ys, qx, qy = task_arrays(pool, t)
        if np.ptp(ys) == 0:                       # protocol v6 amendment 3: TabPFN undefined on a constant support target
            und[t] = True; continue
        sx = pool['sx'][t].numpy().astype(float)
        Xq = {e: np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan) for e in E}
        if set(V) & set(SITE):                    # one main call for every condition, and the two cross-fit calls
            crit, lg = full_predict(reg, Xs, ys, np.concatenate([Xq[e] for e in E])); calls += 1
            nq = len(qy)
            cf = {e: [] for e in E}               # (crit, logits, rows, y) per half
            for tr, te in ((slice(CF, NS), slice(0, CF)), (slice(0, CF), slice(CF, NS))):
                if np.ptp(ys[tr]) == 0:
                    continue
                Xb = np.concatenate([np.where(cfm[e][t, te] > 0, sx[te], np.nan) for e in E])
                c2, l2 = full_predict(reg, Xs[tr], ys[tr], Xb); calls += 1
                k = te.stop - te.start
                for i, e in enumerate(E):
                    cf[e].append((c2, l2[i * k:(i + 1) * k], te, ys[te]))
            for i, e in enumerate(E):
                L = lg[i * nq:(i + 1) * nq]
                m, v = (x[t] for x in U[f'e{e}'])
                if 'alone' in V:
                    nll, mu, crps, cov = scores(crit, L, qy)
                    r = res['alone']
                    r[f'e{e}_nll'][t], r[f'e{e}_se'][t], r[f'e{e}_crps'][t], r[f'e{e}_cov'][t] = nll.mean(), ((mu - qy) ** 2).mean(), crps.mean(), cov.mean()
                    pq[f'e{e}_nll'][t], pq[f'e{e}_mean'][t] = nll, mu
                    for c2, l2, te, yt in cf[e]:
                        nl2, mu2, _, _ = scores(c2, l2, yt)
                        pq[f'cf_e{e}_nll'][t, te], pq[f'cf_e{e}_mean'][t, te] = nl2, mu2
                if 'uptsite' in V:
                    best, ab = -np.inf, (1.0, 0.0)
                    cm, cv = (x[t] for x in U[f'cf_e{e}'])
                    if cf[e]:
                        for aa in A_GRID:
                            for bb in B_GRID:
                                s = 0.
                                for c2, l2, te, yt in cf[e]:
                                    with torch.no_grad():
                                        f = fused(c2, l2, cm[te], cv[te], aa, bb)
                                        s -= float(c2(f, torch.as_tensor(yt, dtype=f.dtype, device=f.device)).double().sum())
                                if np.isfinite(s) and s > best + 1e-9:
                                    best, ab = s, (aa, bb)
                    with torch.no_grad():
                        f = fused(crit, L, m, v, *ab)
                    nll, mu, crps, cov = scores(crit, f, qy)
                    r = res['uptsite']
                    r[f'e{e}_nll'][t], r[f'e{e}_se'][t], r[f'e{e}_crps'][t], r[f'e{e}_cov'][t] = nll.mean(), ((mu - qy) ** 2).mean(), crps.mean(), cov.mean()
                    r[f'e{e}_a'][t], r[f'e{e}_b'][t] = ab
        for e in E:                               # column variants: one call per condition and variant
            m, v = (x[t] for x in U[f'e{e}'])
            cm, cv = (x[t] for x in U[f'cf_e{e}'])
            for vv in set(V) & set(COLUMN):
                cols_s = [cm] if vv == 'uptcol' else [cm, 0.5 * np.log(cv)]
                cols_q = [m] if vv == 'uptcol' else [m, 0.5 * np.log(v)]
                S = np.concatenate([Xs] + [c[:, None] for c in cols_s], 1)
                Q = np.concatenate([Xq[e]] + [c[:, None] for c in cols_q], 1)
                if vv == 'uptres':
                    nll, mu, crps, cov = tabpfn_nll(reg, S, ys - cm, Q, qy - m, quant=True); mu = mu + m
                else:
                    nll, mu, crps, cov = tabpfn_nll(reg, S, ys, Q, qy, quant=True)
                calls += 1
                r = res[vv]
                r[f'e{e}_nll'][t], r[f'e{e}_se'][t], r[f'e{e}_crps'][t], r[f'e{e}_cov'][t] = nll.mean(), ((mu - qy) ** 2).mean(), crps.mean(), cov.mean()
        done = (~skipped[:t + 1]).sum()
        if done % 50 == 0:
            print(a.net, t + 1, n, f'{time.time() - t0:.0f}s', {v: round(float(np.nanmean(res[v]['e0_nll'][:t + 1])), 3) for v in V}, flush=True)
    meta = dict(version=a.version, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, column=f'{a.model}{a.sfx} seeds {SEEDS}',
                stride=a.stride, episodes=int((~skipped).sum()), calls=calls, seconds=time.time() - t0, undefined=int(und.sum()),
                grid=dict(a=A_GRID, b=B_GRID))
    if torch.cuda.is_available():
        meta['gpu'] = torch.cuda.get_device_name(0)
    for v, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        r = res[v]; r['undefined'] = und; r['skipped'] = skipped
        ok = ~und & ~skipped
        ok[n:] = False
        assert all(np.isfinite(r[k][ok]).all() for k in r if k not in ('undefined', 'skipped')), v
        tmp = d / f'{stem}.tmp.npz'; np.savez_compressed(tmp, **r); tmp.replace(d / f'{stem}.npz')
        if v == 'alone' and a.limit is None:
            tmp = d / f'pq_{tag}.tmp.npz'; np.savez_compressed(tmp, **pq); tmp.replace(d / f'pq_{tag}.npz')
        (d / 'train.json').write_text(json.dumps(dict(meta, variant=v), indent=1))
    print(a.net, 'done', {v: {e: round(float(np.nanmean(res[v][f'e{e}_nll'])), 4) for e in E} for v in V}, flush=True)


if __name__ == '__main__':
    main()
