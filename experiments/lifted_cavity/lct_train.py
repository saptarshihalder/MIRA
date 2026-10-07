"""Train the lifted cavity transformer on fresh tasks (same prior, masks and recipe as the transformer baseline).

python lct_train.py --steps 30000 --seed 1 --out runs/lct_s1
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import devutil
import anchors as an
import family as fam
import lct, models, resume


def fresh_batch(rng, B=16, Q=16, nonlin=.4, P=5):
    T = fam.make_tasks(rng, B, nonlin=nonlin, P=P)
    q = rng.integers(0, T['qx'].shape[1], (B, Q)); ar = np.arange(B)[:, None]
    qm = fam.source_query_masks(rng, B, Q, P)
    mu, sig = an.em_batched(T['sx'], T['sy'], T['sm'])
    anc = {k: v.float() for k, v in an.fa_batched(mu, sig, 1).items()}
    t = lambda a: torch.as_tensor(np.asarray(a, dtype='float32'))
    return dict(sx=t(T['sx']), sy=t(T['sy']), sm=t(T['sm']), anchors={1: anc},
                qx=t(T['qx'][ar, q]).reshape(B * Q, P), qm=t(qm).reshape(B * Q, P), qy=t(T['qy'][ar, q]).reshape(-1),
                tid=torch.arange(B).repeat_interleave(Q))


def _np(x):
    return {k: _np(v) for k, v in x.items()} if isinstance(x, dict) else x.numpy()


def _th(x):
    return {k: _th(v) for k, v in x.items()} if isinstance(x, dict) else torch.from_numpy(x)


def _produce(state, B, Q, q):
    """Background producer: the same rng stream as the sequential loop, each batch sent with the rng state after it."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(); rng.bit_generator.state = state
    while True:
        b = fresh_batch(rng, B, Q)
        q.put((_np(b), rng.bit_generator.state))


class Prefetch:
    """Overlaps the CPU-side anchor fits (EM + FA, ~45 ms per batch) with the GPU step. Batches, their order and the
    checkpointed rng state are exactly those of the sequential loop."""
    def __init__(self, rng, B, Q, depth):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.rng, self.q = rng, ctx.Queue(maxsize=depth)
        self.w = ctx.Process(target=_produce, args=(rng.bit_generator.state, B, Q, self.q), daemon=True)
        self.w.start()

    def __call__(self):
        b, state = self.q.get(timeout=600)
        self.rng.bit_generator.state = state          # the main rng mirrors the producer, so checkpoints stay exact
        return _th(b)

    def close(self):
        self.w.terminate(); self.w.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch-tasks', type=int, default=16); ap.add_argument('--queries', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--save-every', type=int, default=2500)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--ff', type=int, default=128)
    ap.add_argument('--ckpt-every', type=int, default=250)
    ap.add_argument('--prefetch', type=int, default=0, help='batches built ahead in a background process (0: off)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(20_000 + a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ck = out / 'ckpt.pt'
    dev = devutil.pick(a.device)
    arch = dict(d=a.d, layers=a.layers, heads=a.heads, ff=a.ff, K=1)
    model = lct.LCT(**arch).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1., (s + 1) / a.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    meta = dict(vars(a), model='lct', parameters=nparam, arch=arch)
    trace, run, first, elapsed = [], None, 0, 0.
    if resume.exists(ck):
        first, extra = resume.load(ck, model, opt, sched, rng); trace, run, elapsed = extra['trace'], extra['run'], extra['seconds']
    elif (out / 'model.pt').exists():
        raise SystemExit(f'{out} holds a model but no checkpoint; refusing to overwrite')
    start = time.time() - elapsed
    model.train()
    nxt = Prefetch(rng, a.batch_tasks, a.queries, a.prefetch) if a.prefetch and first < a.steps else \
        (lambda: fresh_batch(rng, a.batch_tasks, a.queries))
    for step in range(first, a.steps):
        batch = devutil.to_dev(nxt(), dev)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step()
        run = loss.item() if run is None else .99 * run + .01 * loss.item()
        if (step + 1) % 250 == 0:
            trace.append(dict(step=step + 1, ema_loss=run, seconds=time.time() - start))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / 'model.pt')
            (out / 'train.json').write_text(json.dumps(dict(meta, steps_done=step + 1, seconds=time.time() - start, trace=trace), indent=1))
            print(f'step {step + 1} ema {run:.4f} {time.time() - start:.0f}s', flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            resume.save(ck, step + 1, model, opt, sched, rng, dict(trace=trace, run=run, seconds=time.time() - start))
        if os.environ.get('MIRA_STOP_AT') == str(step + 1):        # test hook: simulate an interruption
            raise SystemExit('stopped for resume test')
    if isinstance(nxt, Prefetch):
        nxt.close()
    resume.clear(ck)


if __name__ == '__main__':
    main()
