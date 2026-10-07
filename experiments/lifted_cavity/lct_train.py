"""Train the lifted cavity transformer on fresh tasks (same prior, masks and recipe as the transformer baseline).

python lct_train.py --steps 30000 --seed 1 --out runs/lct_s1
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch-tasks', type=int, default=16); ap.add_argument('--queries', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--save-every', type=int, default=2500)
    ap.add_argument('--ckpt-every', type=int, default=250)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(20_000 + a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ck = out / 'ckpt.pt'
    model = lct.LCT()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1., (s + 1) / a.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    meta = dict(vars(a), model='lct', parameters=nparam, arch=dict(d=64, layers=4, heads=4, ff=128, K=1))
    trace, run, first, elapsed = [], None, 0, 0.
    if ck.exists():
        first, extra = resume.load(ck, model, opt, sched, rng); trace, run, elapsed = extra['trace'], extra['run'], extra['seconds']
    elif (out / 'model.pt').exists():
        raise SystemExit(f'{out} holds a model but no checkpoint; refusing to overwrite')
    start = time.time() - elapsed
    model.train()
    for step in range(first, a.steps):
        batch = fresh_batch(rng, a.batch_tasks, a.queries)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step()
        run = loss.item() if run is None else .99 * run + .01 * loss.item()
        if (step + 1) % 250 == 0:
            trace.append(dict(step=step + 1, ema_loss=run, seconds=time.time() - start))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save(model.state_dict(), out / 'model.pt')
            (out / 'train.json').write_text(json.dumps(dict(meta, steps_done=step + 1, seconds=time.time() - start, trace=trace), indent=1))
            print(f'step {step + 1} ema {run:.4f} {time.time() - start:.0f}s', flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            resume.save(ck, step + 1, model, opt, sched, rng, dict(trace=trace, run=run, seconds=time.time() - start))
        if os.environ.get('MIRA_STOP_AT') == str(step + 1):        # test hook: simulate an interruption
            raise SystemExit('stopped for resume test')
    ck.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
