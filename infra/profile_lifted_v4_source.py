"""Source-only inference timing at 128 repeated tasks; no efficacy estimates.

Repeated source fixtures measure batch/launch overhead, not sample independence.
Uses unchanged frozen scoring programs and source-smoke checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

CODE = Path('/opt/mira/experiments/lifted_cavity')

def expand(value, old_n=4):
    if isinstance(value, dict):
        return {k: (128 if k == 'n' and v == old_n else expand(v, old_n)) for k, v in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == old_n:
        return value.repeat((32,) + (1,) * (value.ndim - 1))
    if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == old_n:
        return np.concatenate([value] * 32, axis=0)
    if isinstance(value, list) and len(value) == old_n:
        return value * 32
    return value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise ValueError('Real CUDA device required')
    dest = a.root / 'source_profile'; dest.mkdir(exist_ok=True)
    panels = {}
    for kind, original in [('synthetic', 'smoke_source_synthetic'), ('real', 'smoke_source_beijing')]:
        p = torch.load(a.root / 'smoke' / (original + '.pt'), weights_only=False)
        if p['meta'].get('smoke') is not True or p['pool']['n'] != 4:
            raise ValueError('Only the four-task source fixtures are allowed')
        p['pool'] = expand(p['pool'])
        if 'conds' in p:
            p['conds'] = expand(p['conds'])
        p['meta'].update(source_only=True, repeated_for_timing=True, unique_source_tasks=4)
        target = dest / ('source_profile_' + kind + '.pt')
        torch.save(p, target); panels[kind] = target
    report = dict(source_only=True, tasks=128, unique_tasks=4, repeated_for_timing=True,
                  models={}, fixture_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in panels.items()})
    start = time.monotonic()
    for model in ('pfn_L', 'lct_L'):
        times = {}
        for kind in ('synthetic', 'real'):
            run = a.root / 'smoke' / (model if kind == 'synthetic' else model + '_ft')
            before = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
            argv = (['evaluate2.py', 'model', '--run', str(run)] if kind == 'synthetic'
                    else ['eval_real.py', 'score', '--runs', str(run)])
            argv += ['--panel', str(panels[kind]), '--device', 'cuda']
            t0 = time.monotonic()
            with (dest / (model + '_' + kind + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-u', *argv], cwd=CODE, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=max(1, 170-(time.monotonic()-start)))
            times[kind + '_seconds'] = time.monotonic() - t0
            if hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest() != before:
                raise ValueError('Timing must not alter trained checkpoints')
        report['models'][model] = times
    report['seconds'] = time.monotonic() - start
    a.out.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report))

if __name__ == '__main__':
    main()
