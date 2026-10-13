"""Score the pretrained TabPFN v2 regressor on the evaluation panels (run where Hugging Face is reachable, e.g. Colab).

    pip install tabpfn
    python tabpfn_colab.py --panels-dir $MIRA_LC_CACHE --out runs/tabpfn_v2          # every panel in the folder
    python tabpfn_colab.py --panel cache/A.pt cache/B.pt --out runs/tabpfn_v2        # these panels, in this order

colab_v4.py calls this for protocol v4. For each task, TabPFN is fitted in context on the support rows (missing entries
as NaN, which TabPFN handles natively) and predicts the query rows with unobserved sensors set to NaN. Query rows never
attend to each other in TabPFN, so every mask of a task (synthetic panels) or every condition of an episode (real
panels) is predicted in one call. NLL is TabPFN's own predictive density (bar distribution in raw target units, checked
against its predictive mean), so it is directly comparable with the NLL of every other method. Results use the cells
format of every other run. The model version is pinned with --version (default v2, the ungated TabPFN v2 weights;
protocol v13: v3.5, which needs the license token TABPFN_TOKEN).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from score_validation import TABPFN_METRICS, expected_scores, load_scores, validate_arrays


N_ESTIMATORS, RANDOM_STATE = 8, 0                   # fixed by protocol v4
QN = 100                                            # protocol v9 --crps: CRPS from TabPFN's quantiles at (i + .5) / QN
QUANT = [(i + .5) / QN for i in range(QN)] + [.05, .95]


def make_regressor(device, version):
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion(version), device=device, n_estimators=N_ESTIMATORS,
                                                          random_state=RANDOM_STATE)
    except (ImportError, AttributeError):          # tabpfn 2.x: the default model is v2
        return TabPFNRegressor(device=device, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)


def tabpfn_nll(reg, Xs, ys, Xq, yq, quant=False):
    if np.ptp(ys) == 0:
        raise RuntimeError('constant support target: TabPFN returns a point mass and its NLL is undefined')
    reg.fit(Xs, ys)
    out = reg.predict(Xq, output_type='full', **({'quantiles': QUANT} if quant else {}))
    crit, logits = out['criterion'], torch.as_tensor(out['logits'])
    mean = np.asarray(out['mean'], dtype=float)
    with torch.no_grad():
        crit = crit.to(logits.device)
        # In v9.1 the raw-unit bar borders can be float64 while logits are float32.
        # Its forward assigns borders into y in-place: all density inputs must agree.
        logits = logits.to(dtype=crit.borders.dtype)
        m = crit.mean(logits).double().cpu().numpy().reshape(-1)
        if not np.allclose(m, mean, atol=1e-3 * (1 + np.abs(mean).max())):
            raise RuntimeError('TabPFN criterion is not in raw target units; NLL would be mis-scaled')
        nll = crit(logits, torch.as_tensor(yq, dtype=logits.dtype, device=logits.device))
    nll = np.asarray(nll.double().cpu().numpy(), dtype=float).reshape(-1)
    if not np.all(np.isfinite(nll)):
        raise RuntimeError('non-finite TabPFN NLL')
    if quant:   # CRPS = 2 * integral of the pinball loss over levels (midpoint rule, QN levels); central 90% interval
        Q = np.stack([np.asarray(q, dtype=float).reshape(-1) for q in out['quantiles']], 1)
        u, p = np.asarray(yq, float)[:, None] - Q[:, :QN], np.array(QUANT[:QN])
        crps = 2 * (u * (p - (u < 0))).mean(1)
        cov = ((yq >= Q[:, QN]) & (yq <= Q[:, QN + 1])).astype(float)
        return nll, mean, crps, cov
    return nll, mean


def task_arrays(pool, t):
    sx = pool['sx'][t].numpy().astype(float); sm = pool['sm'][t].numpy() > 0
    return (np.where(sm, sx, np.nan), pool['sy'][t].numpy().astype(float),
            pool['qx'][t].numpy().astype(float), pool['qy'][t].numpy().astype(float))


def score_panel(reg, path, out, limit=None, allow_undefined=False, quant=False):
    tag = Path(path).stem
    # Limited smoke outputs must never occupy the full-panel result namespace.
    stem = f'cells_{tag}' if limit is None else f'partial_cells_{tag}_limit{limit}'
    dest = out / f'{stem}.npz'
    panel = torch.load(path, weights_only=False); pool = panel['pool']
    metrics = TABPFN_METRICS + (('crps', 'cov') if quant else ())
    expected = expected_scores(panel, metrics, limit)
    if dest.exists():
        try:
            load_scores(dest, panel, metrics, limit)
            return None
        except (ValueError, OSError, EOFError, KeyError) as e:
            print(f'recomputing invalid scores {dest}: {e}', flush=True)
    n = pool['n'] if limit is None else min(limit, pool['n'])
    res, t0, calls = {}, time.time(), 0
    if 'banks' in panel:                            # synthetic: every task x every mask in each bank
        for k, bank in panel['banks'].items():
            bank = np.asarray(bank); M = len(bank)
            nll = np.zeros((n, M)); se = np.zeros_like(nll)
            for t in range(n):
                Xs, ys, qx, qy = task_arrays(pool, t)
                Xq = np.where(bank[:, None, :] > 0, qx[None], np.nan).reshape(M * len(qy), -1)
                l, mu = tabpfn_nll(reg, Xs, ys, Xq, np.tile(qy, M)); calls += 1
                nll[t] = l.reshape(M, -1).mean(1); se[t] = ((mu.reshape(M, -1) - qy) ** 2).mean(1)
            res[f'k{k}_nll'] = nll; res[f'k{k}_se'] = se
            print(tag, 'k', k, round(float(nll.mean()), 4), f'{time.time() - t0:.0f}s', flush=True)
    else:                                           # real: natural mask x extra dropped sensors, per query
        E = list(panel['conds'])
        nll = np.zeros((len(E), n)); se = np.zeros_like(nll); und = np.zeros(n, bool)
        crps = np.zeros_like(nll); cov = np.zeros_like(nll)
        pq = None
        if quant and limit is None:            # protocol v11: per-query outputs and the support cross-fit, same pass
            import realdata
            cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
            pq = {f'{p_}e{e}_{m}': np.full((n, realdata.NS), np.nan) for p_ in ('', 'cf_') for e in E for m in ('nll', 'mean')}
        for t in range(n):
            Xs, ys, qx, qy = task_arrays(pool, t)
            Xq = np.concatenate([np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan) for e in E])
            if allow_undefined and np.ptp(ys) == 0:             # protocol v6 amendment 3: TabPFN NLL undefined here
                if pq is not None:
                    calls += _crossfit(reg, pool, t, Xs, ys, cfm, E, pq)
                und[t] = True; nll[:, t] = np.nan; se[:, t] = np.nan; crps[:, t] = np.nan; cov[:, t] = np.nan; continue
            r = tabpfn_nll(reg, Xs, ys, Xq, np.tile(qy, len(E)), quant); calls += 1
            l, mu = r[:2]
            if pq is not None:                 # after the main call, so the v10 scores come from the same call sequence
                for i, e in enumerate(E):
                    pq[f'e{e}_nll'][t] = l.reshape(len(E), -1)[i]; pq[f'e{e}_mean'][t] = mu.reshape(len(E), -1)[i]
                calls += _crossfit(reg, pool, t, Xs, ys, cfm, E, pq)
            nll[:, t] = l.reshape(len(E), -1).mean(1); se[:, t] = ((mu.reshape(len(E), -1) - qy) ** 2).mean(1)
            if quant:
                crps[:, t] = r[2].reshape(len(E), -1).mean(1); cov[:, t] = r[3].reshape(len(E), -1).mean(1)
        if pq is not None:
            tmp = out / f'pq_{tag}.tmp.npz'
            np.savez_compressed(tmp, **pq); tmp.replace(out / f'pq_{tag}.npz')
        for i, e in enumerate(E):
            res[f'e{e}_nll'] = nll[i]; res[f'e{e}_se'] = se[i]
            if quant:
                res[f'e{e}_crps'] = crps[i]; res[f'e{e}_cov'] = cov[i]
            print(tag, 'extra', e, round(float(np.nanmean(nll[i])), 4), f'undefined {int(und.sum())}', flush=True)
        if allow_undefined:                         # every defined score must still be finite; undefined ones are flagged
            validate_arrays({k: v[~und] for k, v in res.items()}, {k: (int((~und).sum()),) for k in expected})
            res['undefined'] = und
    if 'undefined' not in res:
        validate_arrays(res, expected)
    tmp = out / f'{stem}.tmp.npz'
    np.savez_compressed(tmp, **res); tmp.replace(dest)
    return dict(tag=tag, tasks=n, calls=calls, seconds=time.time() - t0)


def _crossfit(reg, pool, t, Xs, ys, cfm, E, res):
    """Protocol v11: two-fold cross-fit of episode t's support (fit on one half, predict the other under the shared
    cross-fit masks); fills res['cf_e{e}_nll'|'cf_e{e}_mean'][t]. Returns the number of TabPFN calls."""
    import realdata
    NS, CF = realdata.NS, realdata.CF
    sx, calls = pool['sx'][t].numpy().astype(float), 0
    for tr, te in ((slice(CF, NS), slice(0, CF)), (slice(0, CF), slice(CF, NS))):
        if np.ptp(ys[tr]) == 0:
            continue
        Xb = np.concatenate([np.where(cfm[e][t, te] > 0, sx[te], np.nan) for e in E])
        l, mu = tabpfn_nll(reg, Xs[tr], ys[tr], Xb, np.tile(ys[te], len(E))); calls += 1
        for i, e in enumerate(E):
            res[f'cf_e{e}_nll'][t, te] = l.reshape(len(E), -1)[i]; res[f'cf_e{e}_mean'][t, te] = mu.reshape(len(E), -1)[i]
    return calls


def score_panel_pq(reg, path, out):
    """Protocol v11: per-query NLL and mean on the test queries (every condition) and on the two-fold cross-fit of the
    support (fit on one half, predict the other under realdata.crossfit_masks). NaN where TabPFN is undefined (constant
    training target). Written to pq_<tag>.npz; arrays (n, 48)."""
    import realdata
    tag = Path(path).stem; dest = out / f'pq_{tag}.npz'
    if dest.exists():
        return None
    panel = torch.load(path, weights_only=False); pool = panel['pool']; n = pool['n']; E = list(panel['conds'])
    NS, CF = realdata.NS, realdata.CF
    cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
    res = {f'{p}e{e}_{m}': np.full((n, NS), np.nan) for p in ('', 'cf_') for e in E for m in ('nll', 'mean')}
    t0, calls = time.time(), 0
    for t in range(n):
        Xs, ys, qx, qy = task_arrays(pool, t)
        if np.ptp(ys) > 0:
            Xq = np.concatenate([np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan) for e in E])
            l, mu = tabpfn_nll(reg, Xs, ys, Xq, np.tile(qy, len(E))); calls += 1
            for i, e in enumerate(E):
                res[f'e{e}_nll'][t] = l.reshape(len(E), -1)[i]; res[f'e{e}_mean'][t] = mu.reshape(len(E), -1)[i]
        calls += _crossfit(reg, pool, t, Xs, ys, cfm, E, res)
        if (t + 1) % 200 == 0:
            print(tag, 'per-query', t + 1, n, f'{time.time() - t0:.0f}s', flush=True)
    tmp = out / f'pq_{tag}.tmp.npz'
    np.savez_compressed(tmp, **res); tmp.replace(dest)
    return dict(tag=tag + '_pq', tasks=n, calls=calls, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', nargs='*', default=[]); ap.add_argument('--panels-dir')
    ap.add_argument('--out', required=True); ap.add_argument('--version', default='v2')
    ap.add_argument('--limit', type=int, default=None, help='smoke test: only the first LIMIT tasks of each panel')
    ap.add_argument('--allow-constant-support', action='store_true', help='protocol v6 amendment 3: episodes with a constant '
                    'support target are recorded as undefined (NaN + an "undefined" mask) instead of aborting')
    ap.add_argument('--per-query', action='store_true', help='protocol v11: per-query outputs and the support cross-fit (pq_<tag>.npz)')
    ap.add_argument('--crps', action='store_true', help='protocol v9: also record CRPS and 90%% coverage (real panels)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import tabpfn
    meta = dict(model='tabpfn_' + a.version.replace('.', '_'), version=a.version, package=getattr(tabpfn, '__version__', '?'), n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE, device=a.device, note='pretrained TabPFN regressor, in context, no fine-tuning')
    if torch.cuda.is_available():
        meta['gpu'] = torch.cuda.get_device_name(0)
    old = json.loads((out / 'train.json').read_text()) if (out / 'train.json').exists() else {}
    meta['timing'] = old.get('timing', {})
    reg = make_regressor(a.device, a.version)
    paths = list(a.panel)
    if a.panels_dir:
        paths += sorted(str(p) for p in Path(a.panels_dir).glob('v2_s*.pt')) + sorted(str(p) for p in Path(a.panels_dir).glob('real_*.pt'))
    for p in paths:
        r = score_panel_pq(reg, p, out) if a.per_query else score_panel(reg, p, out, a.limit, a.allow_constant_support, a.crps)
        if r is not None:
            meta['timing'][r['tag']] = r
            (out / 'train.json').write_text(json.dumps(meta, indent=1))
    (out / 'train.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
