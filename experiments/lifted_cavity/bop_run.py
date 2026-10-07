"""Run the Bayes-optimal (HMC) ceiling on a synthetic panel: runs/bop/cells_<tag>.npz plus diagnostics.

python bop_run.py cache/panel_....pt [--tasks N] [--chunk 64]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import bop
import family as fam

ap = argparse.ArgumentParser()
ap.add_argument('panel'); ap.add_argument('--tasks', type=int, default=None); ap.add_argument('--chunk', type=int, default=64)
ap.add_argument('--chains', type=int, default=4); ap.add_argument('--runs', default='runs'); ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
panel = torch.load(a.panel, weights_only=False); pool = panel['pool']; tag = Path(a.panel).stem
nl = float(panel['meta'].get('nonlin', .4)); P = pool['qx'].shape[-1]
T = pool['n'] if a.tasks is None else min(a.tasks, pool['n'])
c = pool['c'][:T].double(); mean, scale = c[:, :P], c[:, P:2 * P]
sm = pool['sm'][:T].double()
sx = (pool['sx'][:T].double() * scale[:, None] + mean[:, None]) * sm
qx = pool['qx'][:T].double() * scale[:, None] + mean[:, None]
sy, qy = pool['sy'][:T].double(), pool['qy'][:T].double()
banks = panel['banks'] if 'banks' in panel else {k: fam.mask_bank(k, P) for k in (0, 1, 2, 3) if k < P}
cells = {k: np.zeros((T, len(b))) for k, b in banks.items()}
diag = dict(rhat_median=[], rhat_frac_gt_1_1=[], accept=[], chain_nll_sd=[])
t0 = time.time()
for s0 in range(0, T, a.chunk):
    sl = slice(s0, min(T, s0 + a.chunk))
    S, sign, acc = bop.sample(sx[sl], sy[sl], sm[sl], nl, chains=a.chains, seed=s0)
    r = bop.rhat_params(S, sign, P)
    diag['rhat_median'].append(float(r.median())); diag['rhat_frac_gt_1_1'].append(float((r > 1.1).double().mean())); diag['accept'].append(acc)
    for i, t in enumerate(range(sl.start, sl.stop)):
        for k, bank in banks.items():
            nll = bop.predictive_nll(S[i], sign[i], nl, qx[t], qy[t], bank)
            cells[k][t] = nll.mean(1).numpy()
        if k == 2 and i < 8:       # chain-to-chain variability of the k=2 predictive (first tasks of each chunk)
            per = [float(bop.predictive_nll(S[i][ch:ch + 1], sign[i], nl, qx[t], qy[t], banks[2]).mean()) for ch in range(a.chains)]
            diag['chain_nll_sd'].append(float(np.std(per)))
    print(f'tasks {sl.stop}/{T} {time.time() - t0:.0f}s rhat_med {diag["rhat_median"][-1]:.3f} acc {acc:.2f}', flush=True)
out = Path(a.runs) / 'bop'; out.mkdir(parents=True, exist_ok=True)
(out / 'train.json').write_text(json.dumps(dict(model='bayes_optimal_hmc', chains=a.chains, draws_per_chain=50, thin=4, L=12)))
np.savez_compressed(out / f'cells_{tag}.npz', **{f'k{k}': v for k, v in cells.items()})
(out / f'diag_{tag}.json').write_text(json.dumps(dict(diag, tasks=T, seconds=time.time() - t0), indent=1))
print('bop', tag, {f'k{k}': round(float(v.mean()), 4) for k, v in cells.items()})
