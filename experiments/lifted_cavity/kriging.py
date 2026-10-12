"""Deep kriging baseline (protocol v8): IGNNK-style inductive kriging (Wu et al. 2021) on each target's local graph.

python kriging.py --dataset metr_la --data <folder> --panel <cache>/real_metr_la_s6061.pt --out <runs_real>/metr_la/ignnk_s1 --seed 1

Graph: the target and its 11 input sensors (the same inputs as every other method); edge weights are the absolute
source-period correlations between the 12 sensors (the PeMS releases have no coordinates), row-normalized into a
diffusion transition matrix. Model: 3 diffusion graph-convolution layers (2 hops, hidden 64, ReLU) as in IGNNK, on node
features [standardized value x mask, mask]; the target node is always masked; Gaussian head (mean, log-variance) on the
target node, trained by NLL. Values are standardized per sensor with source-period statistics. Training: 20,000 Adam
steps (lr 1e-3, batch 256) on random (row, target) pairs from the source period where the target is observed, with the
network's natural missingness (PEMS-BAY / PeMS03 / PeMS08: 20% simulated dropout, as for their support rows) and
0-6 additional dropped inputs, uniformly. At test time the query rows of each panel episode are fed with the condition's
query mask. Plain kriging does not use the episode's 48 labeled support rows. Calibrated kriging (ignnk_cal, the
stronger variant and the protocol-v8 comparator) also predicts the 48 support rows and fits y ~ a + b * mean by least
squares on them; its predictive is N(a + b * mean, RSS / 46) (in-context recalibration under drift).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import devutil
from evaluate2 import gauss_metrics
from finetune_real import load_raw

SUPPORT_DROP = {'pems_bay': .2, 'pems03': .2, 'pems08': .2, 'dwd_t2m': .2, 'aurn_no2': .2, 'usgs_q': .2,
                'usgs_ny': .2, 'usgs_wa': .2, 'dwd_p': .2, 'aurn_pm25': .2}


def split(name):
    import realdata_v6, realdata_v7, realdata_v9
    if name in realdata_v6.SPLIT:
        return realdata_v6.SPLIT[name][:2]
    if name in realdata_v9.NETS:
        return realdata_v9.NETS[name][1:3]
    import realdata_v10
    if name in realdata_v10.NETS:
        return realdata_v10.NETS[name][1:3]
    return realdata_v7.NETS[name][2:4]


def neighbours(name, raw, tgt):
    import realdata_v6
    return list(raw['nbr'][tgt]) if 'nbr' in raw else realdata_v6.neighbours(raw['xy'], tgt)


class DGCN(nn.Module):
    def __init__(self, h=64, hops=2, layers=3):
        super().__init__()
        dims = [2] + [h] * layers
        self.hops = hops
        self.lin = nn.ModuleList([nn.Linear(dims[i] * (hops + 1), dims[i + 1]) for i in range(layers)])
        self.out = nn.Linear(h, 2)

    def forward(self, z, T):                        # z (B, 12, 2), T (B, 12, 12) row-stochastic; node 0 = target
        for i, lin in enumerate(self.lin):
            hs, h = [z], z
            for _ in range(self.hops):
                h = T @ h; hs.append(h)
            z = lin(torch.cat(hs, -1))
            if i < len(self.lin) - 1:
                z = torch.relu(z)
        o = self.out(z[:, 0])
        return o[:, 0], 5 * torch.tanh(o[:, 1] / 5)


def prepare(name, raw):
    V, ts = raw['V'], raw['ts']
    s0, t0 = (np.datetime64(x) for x in split(name))
    src = (ts >= s0) & (ts < t0)
    mu, sd = np.nanmean(V[src], 0), np.nanstd(V[src], 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.)
    mu = np.where(np.isfinite(mu), mu, 0.)
    Z = (V - mu) / sd
    N = V.shape[1]
    nodes = np.array([[t] + neighbours(name, raw, t) for t in range(N)])               # (N, 12)
    C = np.array(np.abs(np.nan_to_num(np.ma.corrcoef(np.ma.masked_invalid(V[src]).T).filled(0.))), copy=True)
    A = C[nodes[:, :, None], nodes[:, None, :]]                                           # (N, 12, 12)
    T = A / A.sum(-1, keepdims=True).clip(1e-6)
    return dict(Z=Z, mu=mu, sd=sd, nodes=nodes, T=T.astype('float32'), src=np.flatnonzero(src))


def train(name, prep, seed, steps, dev):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Z, nodes, T, src = prep['Z'], prep['nodes'], torch.as_tensor(prep['T']), prep['src']
    obs = ~np.isnan(Z)
    model = DGCN().to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    drop = SUPPORT_DROP.get(name, 0.)
    run = None
    for step in range(steps):
        t_idx, g = np.empty(0, int), np.empty(0, int)
        while len(t_idx) < 256:                                                           # rows where the target is observed
            tt = rng.choice(src, 512); gg = rng.integers(0, Z.shape[1], 512); ok = obs[tt, gg]
            t_idx, g = np.r_[t_idx, tt[ok]][:256], np.r_[g, gg[ok]][:256]
        nd = nodes[g]                                                                     # (256, 12)
        x = Z[t_idx[:, None], nd]; m = ~np.isnan(x)
        m[:, 1:] &= rng.random((256, nd.shape[1] - 1)) >= drop
        k = rng.integers(0, 7, 256)                                       # drop k random observed inputs, keep >= 1
        nobs = m[:, 1:].sum(1)
        keff = np.minimum(k, np.maximum(nobs - 1, 0))
        key = np.where(m[:, 1:], rng.random((256, nd.shape[1] - 1)), np.inf)   # uniform random order of observed inputs
        rank = key.argsort(1).argsort(1)
        m[:, 1:] &= ~(rank < keff[:, None])
        y = torch.as_tensor(x[:, 0], dtype=torch.float32, device=dev)
        m[:, 0] = False
        feat = torch.as_tensor(np.stack((np.nan_to_num(x) * m, m), -1), dtype=torch.float32, device=dev)
        mu, lv = model(feat, T[g].to(dev))
        loss = (.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        run = loss.item() if run is None else .99 * run + .01 * loss.item()
    return model, run


def _predict(model, prep, tgt, x, m, dev):
    """Kriging predictive (raw units) of the target from standardized-episode inputs x (rows, 11) under mask m."""
    nd = prep['nodes'][tgt]
    xs = (x - prep['mu'][nd[1:]]) / prep['sd'][nd[1:]]
    feat = np.zeros((len(xs), 12, 2), 'float32')
    feat[:, 1:, 0] = np.where(m, xs, 0.); feat[:, 1:, 1] = m
    Tm = torch.as_tensor(prep['T'][tgt])[None].expand(len(xs), -1, -1).to(dev)
    mu, lv = model(torch.as_tensor(feat, device=dev), Tm)
    s = prep['sd'][nd[0]]
    return mu.cpu().double().numpy() * s + prep['mu'][nd[0]], np.exp(lv.cpu().double().numpy()) * s * s


@torch.no_grad()
def score(name, raw, prep, model, panel, dev):
    pool = panel['pool']; P = pool['qx'].shape[-1]
    ids = {i: n for n, i in enumerate(raw['ids'])}
    tg = np.array([ids[k.split('|')[1]] for k in pool['keys']])
    mean, scale = pool['c'][:, :P].numpy(), pool['c'][:, P:2 * P].numpy()               # realdata.to_pool context layout
    cells, cal = {}, {}
    for e, qm in panel['conds'].items():
        acc = {k: np.zeros(pool['n']) for k in ('nll', 'se', 'cov', 'crps')}
        acc_c = {k: np.zeros(pool['n']) for k in acc}
        for t in range(pool['n']):
            qy = pool['qy'][t].double().numpy(); sy = pool['sy'][t].double().numpy()
            mu, var = _predict(model, prep, tg[t], pool['qx'][t].numpy() * scale[t] + mean[t], np.asarray(qm[t]) > 0, dev)
            smu, _ = _predict(model, prep, tg[t], pool['sx'][t].numpy() * scale[t] + mean[t], pool['sm'][t].numpy() > 0, dev)
            A = np.c_[np.ones_like(smu), smu]
            coef = np.linalg.lstsq(A, sy, rcond=None)[0]
            rss = max(float(((sy - A @ coef) ** 2).sum()) / max(len(sy) - 2, 1), 1e-6)
            for d_, (m_, v_) in ((acc, (mu, var)), (acc_c, (coef[0] + coef[1] * mu, np.full_like(mu, rss)))):
                for k_, v in gauss_metrics(m_, v_, qy).items():
                    d_[k_][t] = v.mean()
        cells[e], cal[e] = acc, acc_c
    return cells, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--data', required=True); ap.add_argument('--panel', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = devutil.pick(a.device); t0 = time.time()
    raw = load_raw(a.dataset, dict(v6=a.data, airq=None, beijing=None, gas=None))
    prep = prepare(a.dataset, raw)
    model, ema = train(a.dataset, prep, a.seed, a.steps, dev)
    panel = torch.load(a.panel, weights_only=False)
    cells, cal = score(a.dataset, raw, prep, model.eval(), panel, dev)
    tag = Path(a.panel).stem
    for kind, res, out in (('ignnk', cells, Path(a.out)), ('ignnk_cal', cal, Path(str(a.out).replace('ignnk_', 'ignnk_cal_')))):
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f'cells_{tag}.npz', **{f'e{e}_{m}': v for e, d in res.items() for m, v in d.items()})
        torch.save(model.state_dict(), out / 'model.pt')
        (out / 'train.json').write_text(json.dumps(dict(model=kind, dataset=a.dataset, seed=a.seed, steps=a.steps, final_ema=ema,
                                                        seconds=time.time() - t0, files=raw['files']), indent=1))
        print(a.dataset, kind, a.seed, {f'e{e}': round(float(d['nll'].mean()), 4) for e, d in res.items()}, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
