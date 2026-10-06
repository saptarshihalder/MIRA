"""Collect per-cell scores for all runs on one synthetic panel; paired task-level intervals vs a reference."""
import json, sys
from pathlib import Path
import numpy as np
import torch

panel_path = Path(sys.argv[1]); runs = Path(sys.argv[2]); ref_name = sys.argv[3] if len(sys.argv) > 3 else 'fa1'
panel = torch.load(panel_path, weights_only=False)
refs = panel['refs']; tag = panel_path.stem
rows = {}
for name, cells in refs[0].items():
    rows[name] = {k: refs[k][name] for k in refs}
for run in sorted(runs.iterdir()):
    f = run / f'cells_{tag}.npz'
    if f.exists():
        z = np.load(f)
        rows[run.name] = {k: z[f'k{k}'] for k in refs}


def ci(d):
    d = d.mean(1)  # task-level paired differences
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


print(f'panel {tag}; mean NLL by number of missing sensors; gap = share of ({ref_name} -> oracle) gap closed' if 'oracle' in rows else f'panel {tag}')
hdr = f"{'method':28s}" + ''.join(f'{f"k={k}":>10s}' for k in refs) + ''.join(f'{f"gap k={k}":>10s}' for k in refs if 'oracle' in rows)
print(hdr)
out = {}
for name, r in rows.items():
    line = f'{name:28s}' + ''.join(f'{r[k].mean():10.4f}' for k in refs)
    if 'oracle' in rows:
        line += ''.join(f'{(rows[ref_name][k].mean() - r[k].mean()) / (rows[ref_name][k].mean() - rows["oracle"][k].mean()):10.2f}' for k in refs)
    print(line)
    out[name] = {f'k{k}': float(r[k].mean()) for k in refs}
print(f'\npaired task-level 95% CI of NLL gain over {ref_name} (positive = better than {ref_name})')
for name, r in rows.items():
    if name == ref_name:
        continue
    print(f'{name:28s}' + ''.join('  {:+.4f} [{:+.4f},{:+.4f}]'.format(*ci(rows[ref_name][k] - r[k])) for k in refs))
Path(runs, f'summary_{tag}.json').write_text(json.dumps(out, indent=1))
