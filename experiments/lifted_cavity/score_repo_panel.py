"""Score trained models on the repository's own (already scored) cavity development panel, for direct comparison.

python score_repo_panel.py <repo_root> anchored_cavity_v1 runs/lift1_s1,runs/anchor_mlp_s1,...
Uses the panel exactly as stored (context, standardized queries, all ten two-sensor deletions).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import anchors as an, family as fam, models
from evaluate import score
from train import build

repo = Path(sys.argv[1]); name = sys.argv[2]; runs = [Path(p) for p in sys.argv[3].split(',')]
torch.set_num_threads(1)
dev = np.load(repo / f'artifacts/local/{name}/development.npz')
n = len(dev['y'])
mean, scale = np.zeros((n, 5)), np.zeros((n, 5))
for t in range(n):
    m = dev['sm'][t]; cnt = m.sum(0).clip(1); mu = (dev['sx'][t] * m).sum(0) / cnt
    xc = (dev['sx'][t] - mu) * m; mean[t] = mu; scale[t] = np.sqrt((xc * xc).sum(0) / cnt).clip(.1)
sx = (dev['sx'] - mean[:, None]) / scale[:, None] * dev['sm']
pool = dict(c=torch.as_tensor(dev['context']), sx=torch.as_tensor(sx, dtype=torch.float32), sy=torch.as_tensor(dev['sy']),
            sm=torch.as_tensor(dev['sm']), qx=torch.as_tensor(dev['x']), qy=torch.as_tensor(dev['y']), n=n)
mu, sig = an.em_batched(sx, dev['sy'], dev['sm'])
pool['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in (0, 1, 2)}
assert np.allclose(fam.context(dev['sx'], dev['sy'].astype(float), dev['sm'].astype(float))[0], dev['context'], atol=2e-5)
out = {}
for run in runs:
    meta = json.loads((run / 'train.json').read_text())
    model = build(meta['model'], meta['repo']); model.load_state_dict(torch.load(run / 'model.pt', weights_only=True))
    cells = score(model, pool)
    out[run.name] = float(cells[2].mean())
print(json.dumps({'panel': name, 'two_sensor_deletion_nll': out}, indent=1))
dest = Path(sys.argv[4]) if len(sys.argv) > 4 else Path('repo_panel_scores.json')
allres = json.loads(dest.read_text()) if dest.exists() else {}
allres.setdefault(name, {}).update(out)
dest.write_text(json.dumps(allres, indent=1))
