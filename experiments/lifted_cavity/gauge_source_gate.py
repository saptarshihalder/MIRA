"""Execute only the frozen source engineering gate, refusing existing output."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import data
import family
import models
from gauge import SignAveragedCavity, reverse_factor
from train import build

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/lifted_gauge_v2_source.json'


@torch.no_grad()
def score(model, pool, check_sign=False):
    out, error = {}, 0.
    for missing in (0, 1):
        masks = family.mask_bank(missing, pool['qx'].shape[-1])
        rows = []
        for ids, batch in data.eval_batches(pool, masks, tasks_per_batch=8):
            mu, lv = model(batch)
            assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
            if check_sign:
                other = model(reverse_factor(batch))
                error = max(error, *(float((a - b).abs().max()) for a, b in zip((mu, lv), other)))
            rows.append(models.gauss_nll(mu, lv, batch['qy']).reshape(len(ids), len(masks), -1).mean(-1).numpy())
        out[f'k{missing}'] = np.concatenate(rows)
    return out, error


class Anchor(torch.nn.Module):
    def forward(self, batch):
        return models.closed_form(batch['anchors'][1], batch['qx'], batch['qm'], batch['tid'])


def main():
    cfg = json.loads(CONFIG.read_text())
    out = ROOT / cfg['output']
    if out.exists():
        raise FileExistsError(out)
    # The config must be committed and all frozen source/checkpoint identities match.
    committed = subprocess.check_output(['git', 'show', 'HEAD:configs/lifted_gauge_v2_source.json'], cwd=ROOT)
    assert committed == CONFIG.read_bytes(), 'Config is not frozen in HEAD.'
    for name, expected in cfg['hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = dict(protocol_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                  config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  torch=torch.__version__, numpy=np.__version__, comparisons=[],
                  cloud_calls=0, provider_charge_usd=0, trained_updates=0)
    out.mkdir(parents=True)
    for panel in cfg['panels']:
        width, seed = panel['width'], panel['seed']
        pool = data.build_pool(cfg['tasks'], seed, P=width)
        anchor, _ = score(Anchor(), pool)
        np.savez_compressed(out / f'w{width}_anchor.npz', **anchor)
        for train_seed in cfg['training_seeds']:
            base = build('lift1')
            ckpt = ROOT / f'artifacts/reports/lifted_cavity_v1/runs/lift1_s{train_seed}/model.pt'
            base.load_state_dict(torch.load(ckpt, weights_only=True, map_location='cpu'))
            base.eval()
            original, _ = score(base, pool)
            repaired, sign_error = score(SignAveragedCavity(base), pool, check_sign=True)
            # All masks are equally weighted; query rows already averaged by score().
            old = np.concatenate(list(original.values()), axis=1).mean(1)
            new = np.concatenate(list(repaired.values()), axis=1).mean(1)
            delta = new - old
            gain = float(delta.mean())
            se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
            row = dict(width=width, training_seed=train_seed, original_nll=float(old.mean()),
                       repaired_nll=float(new.mean()), deterioration=gain,
                       lo=gain-1.96*se, hi=gain+1.96*se, sign_error=sign_error)
            row['passed'] = bool(gain <= cfg['max_mean_deterioration'] and row['hi'] <= cfg['max_upper_deterioration'] and sign_error == 0)
            result['comparisons'].append(row)
            np.savez_compressed(out / f'w{width}_s{train_seed}.npz',
                                **{f'original_{k}': v for k, v in original.items()},
                                **{f'repaired_{k}': v for k, v in repaired.items()})
        print(f'width {width}: complete', flush=True)
    result['passed'] = all(r['passed'] for r in result['comparisons'])
    result['seconds'] = time.perf_counter()-started
    (out / 'summary.json').write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
