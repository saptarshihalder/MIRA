"""Protocol v9 variant selection on development data (docs/LIFTED_CAVITY_PROTOCOL_V9.md), run before any v9 score.

    python select9.py --v7 /content/drive/MyDrive/mira_v7 --out <mira_v9>/dev/selection.json
Development networks: the three protocol-v7 networks (PeMS03, PeMS08, China air), already used, extra = 0. Candidates:
    A  upt_L                      (protocol v7 UPT-L; single model)
    B  upt_L, tta8                (test-time ensemble over 8 feature-embedding seeds)
    C  upt_L_crps                 (fine-tuned with NLL + 1 x standardized CRPS)
    D  upt_L_crps, tta8
For each, per-episode NLL and CRPS / (support-target std) are averaged over seeds 1-3, then over all episodes. The rule:
among the candidates whose mean NLL is within .02 nats of the best, take the lowest mean standardized CRPS (ties: the
earlier letter).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

NETS = ('pems03', 'pems08', 'air_cn')
SEED_PANEL = 7071
SEEDS = (1, 2, 3)
CAND = {'A': ('upt_L', ''), 'B': ('upt_L', '_tta8'), 'C': ('upt_L_crps', ''), 'D': ('upt_L_crps', '_tta8')}


def sigma(panel):
    return torch.load(panel, weights_only=False)['pool']['sy'].double().std(1).clamp_min(.1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v7', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.v7)
    sig = {n: sigma(root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt') for n in NETS}
    res = {}
    for c, (m, sfx) in CAND.items():
        nll, crps = [], []
        for n in NETS:
            cs = [np.load(root / 'runs_real' / n / f'{m}_s{s}_ft' / f'cells_real_{n}_s{SEED_PANEL}{sfx}.npz') for s in SEEDS]
            nll.append(np.mean([x['e0_nll'] for x in cs], 0)); crps.append(np.mean([x['e0_crps'] for x in cs], 0) / sig[n])
        nll, crps = np.concatenate(nll), np.concatenate(crps)
        assert np.isfinite(nll).all() and np.isfinite(crps).all(), c
        res[c] = dict(model=m, cells_suffix=sfx, nll=float(nll.mean()), crps_std=float(crps.mean()))
    best = min(v['nll'] for v in res.values())
    ok = [c for c in CAND if res[c]['nll'] <= best + .02]
    pick = min(ok, key=lambda c: (res[c]['crps_std'], c))
    out = dict(candidates=res, eligible=ok, selected=pick, model=res[pick]['model'], cells_suffix=res[pick]['cells_suffix'],
               crps_objective=res[pick]['model'].endswith('_crps'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
