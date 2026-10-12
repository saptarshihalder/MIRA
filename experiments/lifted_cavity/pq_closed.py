"""Protocol v12: per-query outputs of the closed-form FA-Gaussian (UPT's anchor) on a panel, on the test queries and on the
two-fold cross-fit of the support, in the pq_<tag>.npz format of eval_real.py --per-query.

python pq_closed.py --panel <panel.pt> --out <runs_real/<net>/fa1_closed>
FA-Gaussian = one-factor model fitted to the (half) support by EM with missing entries (family.em_gaussian, 40 iterations)
and fa.fit_fa; predictive = fa.lifted_poe, exactly as realdata.closed_forms. Episodes where the fit fails fall back to
the support mean and variance of the target.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import family as fam, fa as fa_mod, realdata
from evaluate2 import gauss_metrics


def predict(sx, sy, sm, qx, qm, qy):
    """Per-query NLL and mean of FA-Gaussian fitted on (sx, sy, sm) for queries qx under masks qm."""
    nll, mean = np.zeros(len(qy)), np.zeros(len(qy))
    try:
        mu, sig = fam.em_gaussian(sx, sy, sm, 40); f1 = fa_mod.fit_fa(mu, sig, 1)
    except np.linalg.LinAlgError:
        f1 = None
    pats = {}
    for q in range(len(qy)):
        pats.setdefault(qm[q].tobytes(), []).append(q)
    for _, idx in pats.items():
        idx = np.array(idx); S = np.flatnonzero(qm[idx[0]])
        try:
            pm, pv = fa_mod.lifted_poe(f1, qx[idx], S, len(sy)) if f1 is not None and len(S) else (None, None)
        except np.linalg.LinAlgError:
            pm = None
        if pm is None or not np.all(np.isfinite(pm)):
            pm, pv = np.full(len(idx), sy.mean()), np.full(len(idx), max(sy.var(ddof=1), 1e-6))
        g = gauss_metrics(np.asarray(pm, float), np.broadcast_to(np.asarray(pv, float), (len(idx),)).copy(), qy[idx])
        nll[idx], mean[idx] = g['nll'], pm
    return nll, mean


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--panel', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    panel = torch.load(a.panel, weights_only=False); pool = panel['pool']; n = pool['n']; E = list(panel['conds'])
    NS, CF = realdata.NS, realdata.CF
    cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
    res = {f'{p}e{e}_{m}': np.full((n, NS), np.nan) for p in ('', 'cf_') for e in E for m in ('nll', 'mean')}
    for t in range(n):
        sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm'))
        qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
        for e in E:
            res[f'e{e}_nll'][t], res[f'e{e}_mean'][t] = predict(sx, sy, sm, qx, np.asarray(panel['conds'][e][t]), qy)
            for tr, te in ((slice(CF, NS), slice(0, CF)), (slice(0, CF), slice(CF, NS))):
                l, m = predict(sx[tr], sy[tr], sm[tr], sx[te], cfm[e][t, te], sy[te])
                res[f'cf_e{e}_nll'][t, te], res[f'cf_e{e}_mean'][t, te] = l, m
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'pq_{Path(a.panel).stem}.npz', **res)
    (out / 'train.json').write_text('{"model": "fa1_closed", "note": "closed-form FA-Gaussian, per-query (protocol v12)"}')
    print(Path(a.panel).stem, {f'e{e}': round(float(res[f"e{e}_nll"].mean()), 4) for e in E})


if __name__ == '__main__':
    main()
