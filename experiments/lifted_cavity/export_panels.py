"""Export the closed-form reference scores, cluster keys and metadata of evaluation panels (small files), so that
make_paper.py can regenerate every number without the large panel caches.

python export_panels.py --out ../../artifacts/reports/lifted_cavity_v2/panels cache/v2_s20261201_n256_p5_nl0.4_sr48.pt ...
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('panels', nargs='+')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in a.panels:
    p = torch.load(f, weights_only=False); tag = Path(f).stem
    arrs = {f'{c}|{name}|{m}': np.asarray(v) for c, d in p['refs'].items() for name, mets in d.items() for m, v in mets.items()}
    np.savez_compressed(out / f'{tag}_refs.npz', **arrs)
    info = dict(meta={k: v for k, v in p['meta'].items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))},
                n=int(p['pool']['n']), keys=list(p['pool'].get('keys', [])),
                banks={str(k): int(len(b)) for k, b in p.get('banks', {}).items()})
    (out / f'{tag}.json').write_text(json.dumps(info, indent=1, default=str))
    print(tag, len(arrs), 'arrays')
