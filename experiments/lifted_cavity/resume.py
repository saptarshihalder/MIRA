"""Exact resume for long CPU runs. A checkpoint holds the model, optimizer, scheduler and every random state, so a run
that is interrupted and resumed reproduces the uninterrupted run step for step (single-threaded CPU)."""
import os
from pathlib import Path
import torch


def save(path, step, model, opt, sched, rng, extra):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    torch_rng=torch.get_rng_state(), np_rng=rng.bit_generator.state, extra=extra), tmp)
    os.replace(tmp, path)


def load(path, model, opt, sched, rng):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    torch.set_rng_state(ck['torch_rng']); rng.bit_generator.state = ck['np_rng']
    return ck['step'], ck['extra']
