"""Exact resume for long runs. A checkpoint holds the model, optimizer, scheduler and every random state (CPU, NumPy and
CUDA), so an interrupted-and-resumed run reproduces the uninterrupted one step for step on a single-threaded CPU; on a
GPU, nondeterministic kernels can make it differ in the last bits."""
import os
from pathlib import Path
import torch


def save(path, step, model, opt, sched, rng, extra):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    torch_rng=torch.get_rng_state(), np_rng=rng.bit_generator.state, extra=extra,
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), tmp)
    os.replace(tmp, path)


def load(path, model, opt, sched, rng):
    ck = torch.load(path, weights_only=False, map_location='cpu')
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    torch.set_rng_state(ck['torch_rng']); rng.bit_generator.state = ck['np_rng']
    if ck.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['step'], ck['extra']
