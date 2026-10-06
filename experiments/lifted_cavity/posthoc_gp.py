"""Post-hoc (not in protocol v2): per-task GP on the real test panels and on F1 (k=2). Writes runs_posthoc/gp/cells_*.npz."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import family as fam
from evaluate2 import gauss_metrics
import warnings
warnings.filterwarnings('ignore')
out = Path('runs_posthoc/gp'); out.mkdir(parents=True, exist_ok=True)
(out / 'train.json').write_text(json.dumps(dict(model='closed_gp_cc', note='post-hoc, ARD RBF + white, complete-case, 1 restart')))
for tag in sys.argv[1:]:
    panel = torch.load(f'cache/{tag}.pt', weights_only=False); pool = panel['pool']; t0 = time.time()
    res = {}
    if 'conds' in panel:
        for e, qm in panel['conds'].items():
            acc = {m: np.zeros(pool['n']) for m in ('nll', 'se', 'cov', 'crps')}
            for t in range(pool['n']):
                sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm')); qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
                per = {m: np.zeros(len(qy)) for m in acc}
                pats = {}
                for q in range(len(qy)):
                    pats.setdefault(qm[t, q].tobytes(), []).append(q)
                for key, idx in pats.items():
                    idx = np.array(idx); S = np.flatnonzero(qm[t, idx[0]])
                    for m, v in gauss_metrics(*fam.gp_cc(sx, sy, sm.astype(bool), qx[idx], S), qy[idx]).items():
                        per[m][idx] = v
                for m in acc:
                    acc[m][t] = per[m].mean()
            for m, v in acc.items():
                res[f'e{e}_{m}'] = v
            print(tag, 'extra', e, round(float(acc['nll'].mean()), 4), f'{time.time() - t0:.0f}s', flush=True)
    else:
        for k in (2,):
            bank = panel['banks'][k]
            acc = {m: np.zeros((pool['n'], len(bank))) for m in ('nll', 'se', 'cov', 'crps')}
            for t in range(pool['n']):
                sx, sy, sm = (pool[k_][t].double().numpy() for k_ in ('sx', 'sy', 'sm')); qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
                for i, msk in enumerate(bank):
                    for m, v in gauss_metrics(*fam.gp_cc(sx, sy, sm.astype(bool), qx, np.flatnonzero(msk)), qy).items():
                        acc[m][t, i] = v.mean()
            for m, v in acc.items():
                res[f'k{k}_{m}'] = v
            print(tag, 'k', k, round(float(acc['nll'].mean()), 4), f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(out / f'cells_{tag}.npz', **res)
