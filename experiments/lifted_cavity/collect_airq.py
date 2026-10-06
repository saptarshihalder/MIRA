"""Air Quality test-week table: closed forms, zero-shot and source-fine-tuned models; paired week-level CIs."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from evaluate import score
from train import build

panel = torch.load(sys.argv[1], weights_only=False)
zero_shot = [Path(p) for p in sys.argv[2].split(',') if p]
ft_dir = Path(sys.argv[3]); ref = sys.argv[4] if len(sys.argv) > 4 else 'em_gauss'
torch.set_num_threads(1)
test = panel['split']['test']
rows = {n: {k: test['refs'][k][n] for k in test['refs']} for n in test['refs'][0]}
for run in zero_shot:
    meta = json.loads((run / 'train.json').read_text())
    model = build(meta['model'], meta['repo']); model.load_state_dict(torch.load(run / 'model.pt', weights_only=True))
    rows[run.name + ' (zero-shot)'] = score(model, test['pool'])
for run in sorted(ft_dir.iterdir()):
    f = run / 'cells_airq_test.npz'
    if f.exists():
        z = np.load(f); rows[run.name] = {k: z[f'k{k}'] for k in test['refs']}
print(f'Air Quality test weeks (n={test["pool"]["n"]}), target log {panel["meta"].get("target", "CO(GT)")}, mean NLL by number of missing sensors')
print(f"{'method':34s}" + ''.join(f'{f"k={k}":>9s}' for k in test['refs']))
for name, r in rows.items():
    print(f'{name:34s}' + ''.join(f'{r[k].mean():9.4f}' for k in r))
print(f'\npaired week-level 95% CI, NLL gain over {ref}')
out = {}
for name, r in rows.items():
    if name == ref:
        continue
    s = ''; out[name] = {}
    for k in r:
        d = (rows[ref][k] - r[k]).mean(1); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        s += f'  {m:+.3f} [{m - 1.96 * se:+.3f},{m + 1.96 * se:+.3f}]'
        out[name][f'k{k}'] = dict(nll=float(r[k].mean()), gain=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se),
                                   weeks_better=int((d > 0).sum()))
    print(f'{name:34s}' + s)
print('\nweeks (of %d) where method beats %s at k=2:' % (test['pool']['n'], ref), {n: v['k2']['weeks_better'] for n, v in out.items()})
Path(ft_dir, 'summary_airq_test.json').write_text(json.dumps(out, indent=1))
