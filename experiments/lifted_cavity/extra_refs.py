"""Score closed-form extra references on a synthetic panel and store them like a run: runs/<method>/cells_<tag>.npz.

python extra_refs.py nlfa cache/panel_....pt [--tasks N]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import family as fam
import nlfa

ap = argparse.ArgumentParser()
ap.add_argument('method', choices=['nlfa', 'nlfa_lin', 'blr_cc'])
ap.add_argument('panel'); ap.add_argument('--runs', default='runs')
a = ap.parse_args()
panel = torch.load(a.panel, weights_only=False); pool = panel['pool']; tag = Path(a.panel).stem
P = pool['qx'].shape[-1]
degrees = (3,) if a.method == 'nlfa' else (1,)
out = Path(a.runs) / a.method; out.mkdir(parents=True, exist_ok=True)
(out / 'train.json').write_text(json.dumps(dict(model='closed_' + a.method, degrees=degrees, K=1, lam=100. if a.method == 'nlfa' else .1)))
t0 = time.time(); cells = {}
for k in (0, 1, 2, 3):
    if k >= P:
        continue
    bank = fam.mask_bank(k, P)
    c = np.zeros((pool['n'], len(bank)))
    for t in range(pool['n']):
        if a.method == 'blr_cc':
            sx_, sy_, sm_ = pool['sx'][t].double().numpy(), pool['sy'][t].double().numpy(), pool['sm'][t].numpy().astype(bool)
            for i, mask in enumerate(bank):
                S = np.flatnonzero(mask)
                c[t, i] = fam.gnll(*fam.blr_cc(sx_, sy_, sm_, pool['qx'][t].double().numpy(), S), pool['qy'][t].double().numpy()).mean()
            continue
        m = nlfa.fit(pool['sx'][t].numpy(), pool['sy'][t].numpy().astype(float), pool['sm'][t].numpy(), degrees=degrees, lam=100. if a.method == 'nlfa' else .1)
        for i, mask in enumerate(bank):
            c[t, i] = nlfa.predict_nll(m, pool['qx'][t].numpy(), pool['qy'][t].numpy().astype(float), np.flatnonzero(mask)).mean()
    cells[f'k{k}'] = c
np.savez_compressed(out / f'cells_{tag}.npz', **cells)
print(a.method, tag, {k: round(float(v.mean()), 4) for k, v in cells.items()}, f'{time.time() - t0:.0f}s')
