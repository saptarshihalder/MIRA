"""Read-only replay of imported confirmation checkpoints; never fits or replaces scores.

Run from the repository root after placing the official CSV under the ignored
artifacts/runs/lifted_cavity/audit_official directory. Output is a separate audit.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import airq
import data
import fa
import family
import models
from train import build


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'artifacts/reports/lifted_cavity_v1'
OUT = ROOT / 'artifacts/reports/lifted_cavity_import_audit'


def paired(base, new):
    d = np.asarray(base) - np.asarray(new)
    assert d.ndim == 1 and len(d) > 1 and np.isfinite(d).all()
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return dict(gain=float(mean), lo=float(mean - 1.96 * se),
                hi=float(mean + 1.96 * se), n=len(d))


@torch.no_grad()
def replay(run, pool):
    meta = json.loads((run / 'train.json').read_text())
    model = build(meta['model'])
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    masks = family.mask_bank(2, pool['qx'].shape[-1])
    parts = []
    for ids, batch in data.eval_batches(pool, masks, tasks_per_batch=8):
        mu, lv = model(batch)
        assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
        parts.append(models.gauss_nll(mu, lv, batch['qy']).reshape(len(ids), len(masks), -1).mean(-1).numpy())
    return np.concatenate(parts)


def closed_reference(pool, kind):
    masks = family.mask_bank(2, pool['qx'].shape[-1])
    rows = []
    for t in range(pool['n']):
        sx, sy, sm, qx, qy = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm', 'qx', 'qy'))
        mu, sig = family.em_gaussian(sx, sy, sm, 40)
        anchor = fa.fit_fa(mu, sig, 1) if kind == 'fa1' else None
        scores = []
        for mask in masks:
            observed = np.flatnonzero(mask)
            pred = (fa.lifted_poe(anchor, qx, observed, len(sy)) if anchor is not None
                    else family.em_conditional(mu, sig, qx, observed, len(sy)))
            scores.append(family.gnll(*pred, qy).mean())
        rows.append(scores)
    return np.array(rows)


def main():
    if OUT.exists():
        raise FileExistsError('Audit exists; retain it and use an explicitly versioned successor.')
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = json.loads((REPORT / 'confirmation/conf_results.json').read_text())
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                  torch=torch.__version__, numpy=np.__version__, cloud_calls=0, provider_charge_usd=0,
                  trained_updates=0, checkpoint_hashes={}, replay_max_abs={}, endpoints={})
    for name, expected in original['checkpoint_sha256'].items():
        actual = hashlib.sha256((REPORT / 'runs' / name / 'model.pt').read_bytes()).hexdigest()
        assert actual == expected, name
        result['checkpoint_hashes'][name] = actual
    for label, seed, n, width, nl in [('C1', 20261101, 256, 5, .4),
                                      ('C2', 20261102, 128, 8, .4),
                                      ('C3', 20261103, 128, 5, .8)]:
        pool = data.build_pool(n, seed, nonlin=nl, P=width)
        tag = f'panel_s{seed}_n{n}_nl{nl}_sr48_sm0.2' + (f'_p{width}' if width != 5 else '')
        predictions = {}
        for method in (['lift1'] if width == 8 else ['lift1', 'anchor_mlp']):
            arrays = []
            for train_seed in (1, 2, 3):
                run = REPORT / 'runs' / f'{method}_s{train_seed}'
                fresh = replay(run, pool)
                with np.load(run / f'cells_{tag}.npz', allow_pickle=False) as saved:
                    assert fresh.shape == saved['k2'].shape
                    result['replay_max_abs'][f'{label}/{run.name}'] = float(np.abs(fresh - saved['k2']).max())
                arrays.append(fresh)
            predictions[method] = np.array(arrays)
        lift = predictions['lift1'].mean((0, 2))
        if label in ('C1', 'C2'):
            reference = closed_reference(pool, 'fa1').mean(1)
            e = paired(reference, lift)
            e['per_seed_gain'] = [float(reference.mean() - a.mean()) for a in predictions['lift1']]
            e['passed'] = bool(e['lo'] > 0 and (min(e['per_seed_gain']) if label == 'C1' else e['gain']) >= .01)
            result['endpoints']['E1' if label == 'C1' else 'E3'] = e
        if label in ('C1', 'C3'):
            e = paired(predictions['anchor_mlp'].mean((0, 2)), lift)
            e['passed'] = bool(e['gain'] >= .01 and e['lo'] > 0)
            result['endpoints']['E2' if label == 'C1' else 'E4'] = e
        print(label, 'replayed', flush=True)
    csv = ROOT / 'artifacts/runs/lifted_cavity/audit_official/AirQualityUCI.csv'
    if csv.exists():
        result['csv_sha256'] = hashlib.sha256(csv.read_bytes()).hexdigest()
        assert result['csv_sha256'] == '13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996'
        frame = airq.load(csv, 'NO2(GT)')
        episodes = airq.episodes(frame, 7)
        sel = ~np.asarray(episodes['source'])
        sub = {k: (v[sel] if k != 'start' else [s for s, f in zip(v, sel) if f]) for k, v in episodes.items()}
        pool = airq.to_pool(sub)
        run = REPORT / 'airq_no2/lift1_s1_ft'
        fresh = replay(run, pool)
        with np.load(run / 'cells_airq_test.npz', allow_pickle=False) as saved:
            result['replay_max_abs']['R1/lift1_s1_ft'] = float(np.abs(fresh - saved['k2']).max())
        e = paired(closed_reference(pool, 'em_gauss').mean(1), fresh.mean(1))
        e['passed'] = bool(e['lo'] > 0)
        result['endpoints']['R1'] = e
        result['real_checkpoint_sha256'] = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
        result['real_test_week_starts'] = sub['start']
    else:
        result['R1_status'] = 'not replayed: official CSV unavailable'
    result['endpoint_max_abs'] = {k: max(abs(e[f] - original[k][f]) for f in ('gain', 'lo', 'hi'))
                                  for k, e in result['endpoints'].items()}
    result['all_five_replayed'] = len(result['endpoints']) == 5
    result['numeric_replay_pass'] = max(result['replay_max_abs'].values()) < 1e-3
    result['endpoint_replay_pass'] = max(result['endpoint_max_abs'].values()) < 1e-3
    result['seconds'] = time.perf_counter() - started
    OUT.mkdir(parents=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['all_five_replayed', 'numeric_replay_pass', 'endpoint_replay_pass', 'seconds']}))


if __name__ == '__main__':
    main()
