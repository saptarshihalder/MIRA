"""Exact resume for long runs. A checkpoint holds the model, optimizer, scheduler and every random state (CPU, NumPy and
CUDA), so an interrupted-and-resumed run reproduces the uninterrupted one step for step on a single-threaded CPU; on a
GPU, nondeterministic kernels can make it differ in the last bits. The previous checkpoint is kept as <name>.prev.pt and
used if the newest one is missing or unreadable (e.g. a cloud drive lost the last write)."""
import os
from pathlib import Path
import torch


def _prev(path):
    return path.with_name(path.stem + '.prev' + path.suffix)


def exists(path):
    path = Path(path)
    return path.exists() or _prev(path).exists()


def save(path, step, model, opt, sched, rng, extra):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    torch_rng=torch.get_rng_state(), np_rng=rng.bit_generator.state, extra=extra,
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), tmp)
    if path.exists():
        os.replace(path, _prev(path))
    os.replace(tmp, path)


def load(path, model, opt, sched, rng):
    path = Path(path); err = None
    for p in (path, _prev(path)):
        if not p.exists():
            continue
        try:
            ck = torch.load(p, weights_only=False, map_location='cpu')
            break
        except Exception as e:                      # truncated or corrupt: fall back to the previous checkpoint
            err = e; print(f'checkpoint {p} unreadable ({e}); trying the previous one', flush=True)
    else:
        raise err or FileNotFoundError(path)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    torch.set_rng_state(ck['torch_rng']); rng.bit_generator.state = ck['np_rng']
    if ck.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['step'], ck['extra']


def clear(path):
    path = Path(path); path.unlink(missing_ok=True); _prev(path).unlink(missing_ok=True)
