"""Evaluate the endpoints frozen in docs/LIFTED_CAVITY_CONFIRMATION.md from saved per-cell scores."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_REP, _CCH = _ROOT / 'artifacts' / 'reports' / 'lifted_cavity_v1', _ROOT / 'artifacts' / 'runs' / 'lifted_cavity' / 'cache'
_args = sys.argv[1:] + [None] * 4
runs = Path(_args[0] or _REP / 'runs'); cache = Path(_args[1] or _CCH); no2 = Path(_args[2] or _REP / 'airq_no2')
out = Path(_args[3] or _REP / 'confirmation' / 'conf_results.json')
C1, C2, C3 = 'panel_s20261101_n256_nl0.4_sr48_sm0.2', 'panel_s20261102_n128_nl0.4_sr48_sm0.2_p8', 'panel_s20261103_n128_nl0.8_sr48_sm0.2'
K = 2


def cells(name, seeds, tag):
    return [np.load(runs / f'{name}_s{s}' / f'cells_{tag}.npz')[f'k{K}'] for s in seeds]


def ref(tag, name):
    return torch.load(cache / f'{tag}.pt', weights_only=False)['refs'][K][name]


def paired(base_task, new_task):
    d = base_task - new_task; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(gain=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se), n=int(len(d)))


res = {}
seeds = (1, 2, 3)
lift_c1 = cells('lift1', seeds, C1); fa_c1 = ref(C1, 'fa1')
per_seed = [float(fa_c1.mean() - c.mean()) for c in lift_c1]
e1 = paired(fa_c1.mean(1), np.mean([c.mean(1) for c in lift_c1], 0))
res['E1'] = dict(per_seed_gain=per_seed, **e1, passed=bool(min(per_seed) >= .01 and e1['lo'] > 0))
mlp_c1 = cells('anchor_mlp', seeds, C1)
e2 = paired(np.mean([c.mean(1) for c in mlp_c1], 0), np.mean([c.mean(1) for c in lift_c1], 0))
res['E2'] = dict(**e2, passed=bool(e2['gain'] >= .01 and e2['lo'] > 0))
if (runs / 'lift1_s1' / f'cells_{C2}.npz').exists():
    lift_c2 = cells('lift1', seeds, C2)
    e3 = paired(ref(C2, 'fa1').mean(1), np.mean([c.mean(1) for c in lift_c2], 0))
    res['E3'] = dict(**e3, passed=bool(e3['gain'] >= .01 and e3['lo'] > 0))
lift_c3 = cells('lift1', seeds, C3); mlp_c3 = cells('anchor_mlp', seeds, C3)
e4 = paired(np.mean([c.mean(1) for c in mlp_c3], 0), np.mean([c.mean(1) for c in lift_c3], 0))
res['E4'] = dict(**e4, passed=bool(e4['gain'] >= .01 and e4['lo'] > 0))
r1 = json.loads((no2 / 'summary_airq_test.json').read_text())['lift1_s1_ft']['k2']
res['R1'] = dict(r1, passed=bool(r1['lo'] > 0))
# descriptive context
res['context'] = {
    'C1_k2': {'oracle': float(ref(C1, 'oracle').mean()), 'fa1': float(fa_c1.mean()), 'em_gauss': float(ref(C1, 'em_gauss').mean()),
              'ridge_repo': float(ref(C1, 'ridge_repo').mean()), 'lift1_mean': float(np.mean([c.mean() for c in lift_c1])),
              'anchor_mlp_mean': float(np.mean([c.mean() for c in mlp_c1]))},
    'C3_k2': {'oracle': float(ref(C3, 'oracle').mean()), 'fa1': float(ref(C3, 'fa1').mean()),
              'lift1_mean': float(np.mean([c.mean() for c in lift_c3])), 'anchor_mlp_mean': float(np.mean([c.mean() for c in mlp_c3]))}}
if 'E3' in res:
    res['context']['C2_k2'] = {'oracle': float(ref(C2, 'oracle').mean()), 'fa1': float(ref(C2, 'fa1').mean()),
                               'em_gauss': float(ref(C2, 'em_gauss').mean()), 'lift1_mean': float(np.mean([c.mean() for c in lift_c2]))}
res['checkpoint_sha256'] = {d.name: hashlib.sha256((d / 'model.pt').read_bytes()).hexdigest()
                            for d in sorted(runs.iterdir()) if (d / 'model.pt').exists() and d.name.split('_s')[0] in ('lift1', 'anchor_mlp')}
out.write_text(json.dumps(res, indent=1))
for k in ('E1', 'E2', 'E3', 'E4', 'R1'):
    if k in res:
        v = res[k]
        print(f"{k}: gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" + (f"  per-seed {[round(x, 4) for x in v['per_seed_gain']]}" if 'per_seed_gain' in v else '') + f"  -> {'PASS' if v['passed'] else 'FAIL'}")
print(json.dumps(res['context'], indent=1))
