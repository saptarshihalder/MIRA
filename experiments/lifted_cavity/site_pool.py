"""Transformer-as-a-site: fuse the lifted posterior with a free-form transformer by a tempered product of experts.

Both the lifted cavity transformer (LCT) and the cell-token transformer (PFN) return a Gaussian predictive for every
query. The lifted posterior is prior x product of observed sensor sites; the transformer's predictive enters as one
more site on the target, and every factor is tempered so that the exponents sum to one (generalized product of experts,
Cao & Fleet 2014; tempered/power-EP sites, Minka 2004):

    p_pool(y) ∝ p_LCT(y)^(1-b) p_PFN(y)^b,   b = 1/2 fixed before any score.

For Gaussians the pool is Gaussian (precision-weighted). Hölder's inequality gives Z = ∫ p_a^(1-b) p_b^b <= 1, so for
EVERY query and EVERY outcome y
    NLL_pool(y) = (1-b) NLL_a(y) + b NLL_b(y) + log Z  <=  (1-b) NLL_a(y) + b NLL_b(y),
with slack -log Z = b(1-b) * (Rényi-type divergence between the experts) >= 0, zero iff they agree. The linear pool
(mixture) satisfies the same bound by convexity of -log and is reported as a secondary pool.

python site_pool.py predict --panel ../../cache/real_beijing_no2_s3031.pt --runs <run dirs>   (writes preds_<panel>.npz)
python site_pool.py report  --out ../../artifacts/reports/lifted_cavity_paper/site_pool/report.json
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import realdata
from train import build

LOG2PI = math.log(2 * math.pi)
BETA = .5                                                       # fixed before any pooled score was computed


def gauss_nll(mu, var, y):
    return .5 * (LOG2PI + np.log(var) + (y - mu) ** 2 / var)


def log_pool(mu_a, var_a, mu_b, var_b, b=BETA):
    """Normalized geometric pool p_a^(1-b) p_b^b of two Gaussians (a Gaussian)."""
    prec = (1 - b) / var_a + b / var_b
    return ((1 - b) * mu_a / var_a + b * mu_b / var_b) / prec, 1 / prec


def log_pool_slack(mu_a, var_a, mu_b, var_b, b=BETA):
    """-log Z >= 0: how much the pool beats the weighted average NLL, for every y (closed form for Gaussians)."""
    mu, var = log_pool(mu_a, var_a, mu_b, var_b, b)
    lz = (.5 * np.log(var) - (1 - b) * .5 * np.log(var_a) - b * .5 * np.log(var_b)
          - .5 * ((1 - b) * mu_a ** 2 / var_a + b * mu_b ** 2 / var_b - mu ** 2 / var)
          - (1 - b) * .5 * LOG2PI - b * .5 * LOG2PI + .5 * LOG2PI)
    return -lz


def lin_pool_nll(mu_a, var_a, mu_b, var_b, y, b=BETA):
    la, lb = -gauss_nll(mu_a, var_a, y) + math.log(1 - b), -gauss_nll(mu_b, var_b, y) + math.log(b)
    m = np.maximum(la, lb)
    return -(m + np.log(np.exp(la - m) + np.exp(lb - m)))


@torch.no_grad()
def predict(model, pool, qm, tasks_per_batch=8):
    """Per-query Gaussian predictive (mu, var), exactly as realdata.score_model computes it before scoring."""
    model.eval()
    NQ = realdata.NQ
    qmt = torch.as_tensor(qm, dtype=torch.float32)
    mus, lvs = [], []
    for s0 in range(0, pool['n'], tasks_per_batch):
        t = torch.arange(s0, min(pool['n'], s0 + tasks_per_batch))
        B = len(t)
        batch = {k: pool[k][t] for k in ('c', 'sx', 'sy', 'sm')}
        batch['anchors'] = {K: {k: v[t] for k, v in pool['anchors'][K].items()} for K in (0, 1, 2)}
        batch['qx'] = pool['qx'][t].reshape(B * NQ, -1); batch['qy'] = pool['qy'][t].reshape(-1)
        batch['qm'] = qmt[t].reshape(B * NQ, -1); batch['tid'] = torch.arange(B).repeat_interleave(NQ)
        mu, lv = model(batch)
        mus.append(mu.double().numpy().reshape(B, NQ)); lvs.append(lv.double().numpy().reshape(B, NQ))
    return np.concatenate(mus), np.exp(np.concatenate(lvs))


def cmd_predict(a):
    panel = torch.load(a.panel, weights_only=False)
    tag = Path(a.panel).stem
    torch.set_num_threads(a.threads)
    out = Path(a.preds) / f'preds_{tag}.npz'
    have = dict(np.load(out)) if out.exists() else {}
    have['qy'] = panel['pool']['qy'].double().numpy()
    have['weeks'] = np.array([k.split('|')[0] for k in panel['pool']['keys']])
    for run in [Path(r) for r in a.runs.split(',') if r]:
        meta = json.loads((run / 'train.json').read_text())
        model = build(meta['model'], meta.get('repo'), meta)
        model.load_state_dict(torch.load(run / 'model.pt', weights_only=True, map_location='cpu'))
        ref = np.load(run / f'cells_{tag}.npz')
        for e, qm in panel['conds'].items():
            mu, var = predict(model, panel['pool'], qm)
            nll = gauss_nll(mu, var, have['qy']).mean(1)
            err = float(np.abs(nll - ref[f'e{e}_nll']).max())      # must replay the committed per-episode scores
            assert err < 1e-4, (run, e, err)
            have[f'{run.name}|e{e}|mu'], have[f'{run.name}|e{e}|var'] = mu, var
            print(run.name, f'e{e}', f'nll {nll.mean():.4f}', f'replay max err {err:.1e}', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **have)


def week_ci(base, new, weeks, noninf=None, margin=.01):
    """Paired gain base - new (positive: new is better), cluster-robust over weeks, as in confirm3.py."""
    d = base - new
    d = np.array([d[weeks == w].mean() for w in sorted(set(weeks))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)),
                noninferior=bool(lo > (noninf if noninf is not None else -.02)), superior=bool(m >= margin and lo > 0))


def episode_nll(P, run, e):
    return gauss_nll(P[f'{run}|e{e}|mu'], P[f'{run}|e{e}|var'], P['qy']).mean(1)


def pooled(P, a, b, e, kind='log'):
    ma, va, mb, vb = (P[f'{r}|e{e}|{s}'] for r in (a, b) for s in ('mu', 'var'))
    if kind == 'log':
        mu, var = log_pool(ma, va, mb, vb)
        return gauss_nll(mu, var, P['qy']).mean(1), log_pool_slack(ma, va, mb, vb).mean(1)
    return lin_pool_nll(ma, va, mb, vb, P['qy']).mean(1), None


TARGETS = {   # target -> (preds file, pairs to pool). Site expert first, transformer second.
    'beijing_no2': ('preds_real_beijing_no2_s3031.npz', [('lct_s1_ft', 'pfn_s1_ft'), ('lct_s2_ft', 'pfn_s2_ft')],
                    [('pfn_s1_ft', 'pfn_s2_ft'), ('lct_s1_ft', 'lct_s2_ft')]),
    'beijing_pm25': ('preds_real_beijing_s2027.npz', [('lct_s1_ft', 'pfn_s1_ft')], [('pfn_s1_ft', 'pfn_s2_ft')]),
    'beijing_co': ('preds_real_beijing_co_s3031.npz', [('lct_s1_ft', 'pfn_s1_ft')], []),
}


def cmd_report(a):
    res = {}
    for tgt, (fn, pairs, controls) in TARGETS.items():
        f = Path(a.preds) / fn
        if not f.exists():
            continue
        P = dict(np.load(f)); weeks = P['weeks']
        conds = sorted({int(k.split('|')[1][1:]) for k in P if k.count('|') == 2})
        for e in conds:
            row = {}
            single = {r: episode_nll(P, r, e) for r in {x for pr in pairs + controls for x in pr}}
            row['single'] = {r: float(v.mean()) for r, v in sorted(single.items())}
            for sa, sb in pairs + controls:
                nm = f'pool({sa},{sb})'
                lp, slack = pooled(P, sa, sb, e, 'log')
                mp, _ = pooled(P, sa, sb, e, 'lin')
                avg = .5 * (single[sa] + single[sb])
                assert (lp <= avg + 1e-9).all()                       # the Hölder bound, checked per episode
                row[nm] = dict(log_pool=float(lp.mean()), lin_pool=float(mp.mean()), avg_of_experts=float(avg.mean()),
                               holder_slack=float(slack.mean()),
                               vs_transformer=week_ci(single[sb], lp, weeks),
                               vs_best_single=week_ci(min(single[sa], single[sb], key=np.mean), lp, weeks))
            if controls:                                               # complementarity: site+transformer vs same-family pools
                for (sa, sb) in pairs:
                    for (ca, cb) in controls:
                        row[f'pool({sa},{sb}) vs pool({ca},{cb})'] = week_ci(pooled(P, ca, cb, e)[0], pooled(P, sa, sb, e)[0], weeks)
            res[f'{tgt}|e{e}'] = row
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, row in res.items():
        print(f'== {k}', {r: round(v, 4) for r, v in row['single'].items()})
        for nm, v in row.items():
            if nm.startswith('pool(') and ' vs ' not in nm:
                t = v['vs_transformer']
                print(f'  {nm:30s} log {v["log_pool"]:.4f} lin {v["lin_pool"]:.4f} avg {v["avg_of_experts"]:.4f} slack {v["holder_slack"]:.4f}'
                      f' | vs transformer {t["gain"]:+.4f} [{t["lo"]:+.4f},{t["hi"]:+.4f}]')
            elif ' vs ' in nm:
                print(f'  {nm:58s} {v["gain"]:+.4f} [{v["lo"]:+.4f},{v["hi"]:+.4f}]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['predict', 'report'])
    ap.add_argument('--panel'); ap.add_argument('--runs', default='')
    ap.add_argument('--preds', default='../../cache/site_pool'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default='../../artifacts/reports/lifted_cavity_paper/site_pool/report.json')
    a = ap.parse_args()
    cmd_predict(a) if a.what == 'predict' else cmd_report(a)
