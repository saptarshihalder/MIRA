"""Protocol v14 smoke test: time each step of tabpfn_feat.py on the first episodes of one network (prints as it goes).

    python -u smoke14.py --root /content/drive/MyDrive/mira_v9 --net dwd_t2m --panel-seed 9091 --episodes 2
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch

T0 = time.time()


def log(*a):
    print(f'{time.time() - T0:7.1f}s', *a, flush=True)


ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--net', required=True)
ap.add_argument('--panel-seed', type=int, required=True); ap.add_argument('--episodes', type=int, default=2)
a = ap.parse_args()
import realdata
import tabpfn_feat as tf
from tabpfn_colab import make_regressor, tabpfn_nll, task_arrays
log('imports')
tag = f'real_{a.net}_s{a.panel_seed}'
panel = torch.load(Path(a.root) / 'panels' / f'{tag}.pt', weights_only=False); pool = panel['pool']
E = list(panel['conds'])
log('panel', pool['n'], 'episodes; conditions', E)
U = tf.upt_moments(Path(a.root) / 'runs_real' / a.net, 'upt_L_sfm', tag, '_tta8', E)
log('UPT moments; e0 test sd median', float(np.median(np.sqrt(U['e0'][1]))))
cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
log('cross-fit masks')
reg = make_regressor('cuda', 'v3.5')
log('regressor', torch.cuda.get_device_name(0))
for t in range(a.episodes):
    Xs, ys, qx, qy = task_arrays(pool, t)
    if np.ptp(ys) == 0:
        log('episode', t, 'undefined'); continue
    Xq = np.concatenate([np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan) for e in E])
    crit, lg = tf.full_predict(reg, Xs, ys, Xq)
    log('episode', t, 'main call; logits', tuple(lg.shape), lg.device, lg.dtype)
    nll, mu, crps, cov = tf.scores(crit, lg[:48], qy)
    log('  scores (102 quantiles)', round(float(nll.mean()), 3))
    m, v = (x[t] for x in U['e0'])
    for ab in ((1.0, 0.0), (1.0, .35), (.8, .2)):
        f = tf.fused(crit, lg[:48], m, v, *ab)
        log('  fused', ab, round(float(crit(f, torch.as_tensor(qy, dtype=f.dtype, device=f.device)).double().mean()), 4))
    cm, cv = (x[t] for x in U['cf_e0'])
    S = np.concatenate([Xs, cm[:, None], 0.5 * np.log(cv)[:, None]], 1)
    Q = np.concatenate([Xq[:48], m[:, None], 0.5 * np.log(v)[:, None]], 1)
    r = tabpfn_nll(reg, S, ys, Q, qy, quant=True)
    log('  column call (uptcol2)', round(float(r[0].mean()), 3))
    r = tabpfn_nll(reg, S, ys - cm, Q, qy - m, quant=True)
    log('  residual call (uptres)', round(float(r[0].mean()), 3))
log('done')
