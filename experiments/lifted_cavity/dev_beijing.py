"""Development check on Beijing (2015-07..2015-12 weekly episodes; never the frozen test period)."""
import json, sys, numpy as np, torch
import realdata
from finetune_real import load_raw
from train import build
torch.set_num_threads(1)
raw = load_raw('beijing', realdata.DEFAULT)
eps = realdata.episodes('beijing', raw, 'test', 5, period=('2015-07-01', '2016-01-01'))
pool = realdata.to_pool(eps)
print('dev episodes', pool['n'])
for extra in (0, 3, 6):
    qm = realdata.condition_masks(pool, extra, 7 + extra)
    ref = realdata.closed_forms(pool, qm)
    row = {k: float(v['nll'].mean()) for k, v in ref.items()}
    for r in sys.argv[1:]:
        meta = json.loads(open(f'{r}/train.json').read()); m = build(meta['model'], meta.get('repo'), meta)
        m.load_state_dict(torch.load(f'{r}/model.pt', weights_only=True))
        v = realdata.score_model(m, pool, qm)['nll']; row[r.split('/')[-1]] = float(v.mean())
        d = ref['blr_cc']['nll'] - v; row[r.split('/')[-1] + '_vs_ridge'] = f'{d.mean():+.3f}±{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}'
    print('extra', extra, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
