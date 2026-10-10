"""Train one model on the sensor-task family.  All models share data stream, optimizer and schedule.

python train.py --model lift1 --steps 20000 --pool 60000 --seed 1 --out runs/lift1_s1
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import data, models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path(os.environ.get('MIRA_LC_CACHE', ROOT / 'artifacts' / 'runs' / 'lifted_cavity' / 'cache'))  # git-ignored


def build(name, repo_dir=None, meta=None):
    if name == 'pfn':
        import pfn
        arch = (meta or {}).get('arch', {})
        return pfn.CellPFN(**arch)
    if name == 'upt':
        import upt
        arch = dict((meta or {}).get('arch', {}))
        return upt.UPT(**arch)
    if name == 'lct':
        import lct
        arch = dict((meta or {}).get('arch', {}))
        return lct.LCT(**arch)
    if name.startswith('repo_'):
        cands = [Path(p) for p in (repo_dir, HERE.parent, HERE.parent / 'repo_experiments') if p]
        repo_dir = next(p for p in cands if (p / 'anchored_cavity.py').exists())
        sys.path.insert(0, str(repo_dir))
        import anchored_cavity as ac
        return models.RepoWrapper(ac.Model(name[5:]))
    if name == 'anchor_mlp':
        return models.AnchorMLP(K=1)
    K = int(name[4])
    return models.LiftedCavity(K=K, cavity=not name.endswith('_static'))


def get_pool(size, seed, nonlin, cache):
    path = cache / f'pool_n{size}_s{seed}_nl{nonlin}.pt'
    if path.exists():
        return torch.load(path, weights_only=False)
    pool = data.build_pool(size, seed, nonlin=nonlin)
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(pool, path)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--pool', type=int, default=60000)
    ap.add_argument('--pool-seed', type=int, default=5150)
    ap.add_argument('--nonlin', type=float, default=.4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1.8e-3)
    ap.add_argument('--schedule', default='cosine', choices=['cosine', 'constant'])
    ap.add_argument('--batch-tasks', type=int, default=128)
    ap.add_argument('--queries', type=int, default=8)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--repo', default=str(HERE.parent))
    ap.add_argument('--cache', default=str(CACHE))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    pool = get_pool(args.pool, args.pool_seed, args.nonlin, Path(args.cache))
    model = build(args.model, args.repo)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.schedule == 'cosine':
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .1 + .9 * .5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.)
    trace, start = [], time.time()
    run = 0.
    for step in range(args.steps):
        batch = data.train_batch(pool, rng, args.batch_tasks if args.pool >= args.batch_tasks else args.pool, args.queries)
        mu, lv = model(batch)
        loss = models.gauss_nll(mu, lv, batch['qy']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step}')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step(); sched.step()
        lv_ = loss.item(); run = .98 * run + .02 * lv_ if step else lv_
        if (step + 1) % 250 == 0:
            trace.append(dict(step=step + 1, ema_loss=run, seconds=time.time() - start))
    torch.save(model.state_dict(), out / 'model.pt')
    meta = dict(vars(args), parameters=nparam, seconds=time.time() - start, trace=trace)
    (out / 'train.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('model', 'steps', 'pool', 'seed', 'parameters', 'seconds')}), 'final ema', run)


if __name__ == '__main__':
    main()
