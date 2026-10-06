"""Diagnose factor-sign sensitivity; never choose signs for deployment or alter checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import audit_import as audit
import data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    pool = data.build_pool(128, 20261102, P=8)
    run = audit.REPORT / 'runs/lift1_s1'
    original = audit.replay(run, pool)
    pool['anchors'][1]['D'] = -pool['anchors'][1]['D']
    flipped = audit.replay(run, pool)
    with np.load(run / 'cells_panel_s20261102_n128_nl0.4_sr48_sm0.2_p8.npz', allow_pickle=False) as f:
        saved = f['k2'].copy()
    e0 = np.abs(original - saved).max(1)
    e1 = np.abs(flipped - saved).max(1)
    result = dict(diagnostic_only=True, original_error=float(e0.max()),
                  best_sign_error=float(np.minimum(e0, e1).max()),
                  tasks_original_error_above_1e_3=int((e0 > 1e-3).sum()),
                  tasks_fixed_by_sign=int(((e0 > 1e-3) & (e1 < 1e-3)).sum()),
                  max_model_sign_sensitivity=float(np.abs(original - flipped).max()),
                  mean_original=float(original.mean()), mean_flipped=float(flipped.mean()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
