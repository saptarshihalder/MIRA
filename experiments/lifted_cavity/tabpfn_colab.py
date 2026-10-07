"""Score the pretrained TabPFN v2 regressor on the evaluation panels (run where Hugging Face is reachable, e.g. Colab).

    pip install tabpfn
    python tabpfn_colab.py --panels-dir $MIRA_LC_CACHE --out runs/tabpfn_v2          # every panel in the folder
    python tabpfn_colab.py --panel cache/A.pt cache/B.pt --out runs/tabpfn_v2        # these panels, in this order

colab_v4.py calls this for protocol v4. For each task, TabPFN is fitted in context on the support rows (missing entries
as NaN, which TabPFN handles natively) and predicts the query rows with unobserved sensors set to NaN. Query rows never
attend to each other in TabPFN, so every mask of a task (synthetic panels) or every condition of an episode (real
panels) is predicted in one call. NLL is TabPFN's own predictive density (bar distribution in raw target units, checked
against its predictive mean), so it is directly comparable with the NLL of every other method. Results use the cells
format of every other run. The model version is pinned with --version (default v2, the ungated TabPFN v2 weights).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from score_validation import TABPFN_METRICS, expected_scores, load_scores, validate_arrays


N_ESTIMATORS, RANDOM_STATE = 8, 0                   # fixed by protocol v4


def make_regressor(device, version):
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion(version), device=device, n_estimators=N_ESTIMATORS,
                                                          random_state=RANDOM_STATE)
    except (ImportError, AttributeError):          # tabpfn 2.x: the default model is v2
        return TabPFNRegressor(device=device, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)


def tabpfn_nll(reg, Xs, ys, Xq, yq):
    if np.ptp(ys) == 0:
        raise RuntimeError('constant support target: TabPFN returns a point mass and its NLL is undefined')
    reg.fit(Xs, ys)
    out = reg.predict(Xq, output_type='full')
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
    return nll, mean


def task_arrays(pool, t):
    sx = pool['sx'][t].numpy().astype(float); sm = pool['sm'][t].numpy() > 0
    return (np.where(sm, sx, np.nan), pool['sy'][t].numpy().astype(float),
            pool['qx'][t].numpy().astype(float), pool['qy'][t].numpy().astype(float))


def score_panel(reg, path, out, limit=None):
    tag = Path(path).stem
    # Limited smoke outputs must never occupy the full-panel result namespace.
    stem = f'cells_{tag}' if limit is None else f'partial_cells_{tag}_limit{limit}'
    dest = out / f'{stem}.npz'
    panel = torch.load(path, weights_only=False); pool = panel['pool']
    expected = expected_scores(panel, TABPFN_METRICS, limit)
    if dest.exists():
        try:
            load_scores(dest, panel, TABPFN_METRICS, limit)
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
        nll = np.zeros((len(E), n)); se = np.zeros_like(nll)
        for t in range(n):
            Xs, ys, qx, qy = task_arrays(pool, t)
            Xq = np.concatenate([np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan) for e in E])
            l, mu = tabpfn_nll(reg, Xs, ys, Xq, np.tile(qy, len(E))); calls += 1
            nll[:, t] = l.reshape(len(E), -1).mean(1); se[:, t] = ((mu.reshape(len(E), -1) - qy) ** 2).mean(1)
        for i, e in enumerate(E):
            res[f'e{e}_nll'] = nll[i]; res[f'e{e}_se'] = se[i]
            print(tag, 'extra', e, round(float(nll[i].mean()), 4), flush=True)
    validate_arrays(res, expected)
    tmp = out / f'{stem}.tmp.npz'
    np.savez_compressed(tmp, **res); tmp.replace(dest)
    return dict(tag=tag, tasks=n, calls=calls, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', nargs='*', default=[]); ap.add_argument('--panels-dir')
    ap.add_argument('--out', required=True); ap.add_argument('--version', default='v2')
    ap.add_argument('--limit', type=int, default=None, help='smoke test: only the first LIMIT tasks of each panel')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import tabpfn
    meta = dict(model='tabpfn_v2', version=a.version, package=getattr(tabpfn, '__version__', '?'), n_estimators=N_ESTIMATORS,
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
        r = score_panel(reg, p, out, a.limit)
        if r is not None:
            meta['timing'][r['tag']] = r
            (out / 'train.json').write_text(json.dumps(meta, indent=1))
    (out / 'train.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
