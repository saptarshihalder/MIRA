"""Vectorized sensor-task family used by experiments/cavity_site.py, plus exact oracles and closed-form baselines.

The distribution is identical to cavity_site.generate (verified against its per-task code in check_family.py):
  y ~ N(0,1);  x_j = a_j y + b_j + .4 sin(f_j y) + d_j u + s_j eps_j,  u ~ N(0,1) shared across sensors.
Support: 48 labeled rows with independent 20% missingness.  Queries: 48 rows, fully measured, masks applied at use.
"""
import itertools
import numpy as np

LOG2PI = np.log(2 * np.pi)
NS, NQ, P = 48, 48, 5


def sample_params(rng, n, nonlin=.4, P=P):
    a = rng.choice([-1., 1.], (n, P)) * rng.uniform(.6, 1.8, (n, P))
    return dict(a=a, b=rng.normal(0, .4, (n, P)), f=rng.uniform(.5, 2.5, (n, P)),
                d=rng.normal(0, .7, (n, P)), s=rng.uniform(.15, .8, (n, P)), nonlin=np.full(n, nonlin))


def simulate(rng, prm, rows):
    n, P = prm['a'].shape
    y = rng.normal(size=(n, rows)); u = rng.normal(size=(n, rows))
    g = y[..., None] * prm['a'][:, None] + prm['b'][:, None] + prm['nonlin'][:, None, None] * np.sin(y[..., None] * prm['f'][:, None])
    x = g + u[..., None] * prm['d'][:, None] + rng.normal(size=(n, rows, P)) * prm['s'][:, None]
    return x, y


def context(x, y, m):
    """Batched copy of cavity_site.context: x,m (B,48,5), y (B,48) -> c (B,55), mean, scale."""
    n = m.sum(1).clip(1)
    mean = (x * m).sum(1) / n
    xc = (x - mean[:, None]) * m
    scale = np.sqrt((xc * xc).sum(1) / n).clip(.1)
    xs = xc / scale[:, None]
    ym = y.mean(1); ys = y.std(1).clip(.1)
    cross = (xs * (y - ym[:, None])[..., None]).sum(1) / n
    pairs = np.einsum('bni,bnj->bij', xs, xs) / np.einsum('bni,bnj->bij', m, m).clip(1)
    B, P = len(x), x.shape[-1]
    c = np.concatenate([mean, scale, np.repeat(ym[:, None], P, 1), np.repeat(ys[:, None], P, 1), cross,
                        m.mean(1), pairs.reshape(B, -1)], 1)
    return c.astype('float32'), mean, scale


def make_tasks(rng, n, nonlin=.4, support_missing=.2, support_rows=NS, P=P):
    """Return a dict of numpy arrays for n fresh tasks (support + standardized queries + true params)."""
    prm = sample_params(rng, n, nonlin, P)
    x, y = simulate(rng, prm, support_rows + NQ)
    sm = (rng.random((n, support_rows, P)) > support_missing).astype(float)
    c, mean, scale = context(x[:, :support_rows], y[:, :support_rows], sm)
    sx = (x[:, :support_rows] - mean[:, None]) / scale[:, None] * sm   # standardized support, missing -> 0
    qx = (x[:, support_rows:] - mean[:, None]) / scale[:, None]
    return dict(prm=prm, c=c, mean=mean, scale=scale, sx=sx.astype('float32'), sy=y[:, :support_rows].astype('float32'),
                sm=sm.astype('float32'), qx=qx.astype('float32'), qy=y[:, support_rows:].astype('float32'),
                raw_qx=x[:, support_rows:], raw_sx=x[:, :support_rows])


def mask_bank(k_missing, P=P):
    out = []
    for drop in itertools.combinations(range(P), k_missing):
        m = np.ones(P); m[list(drop)] = 0; out.append(m)
    return np.array(out, dtype='float32')


def source_query_masks(rng, n, q, P=P):
    """Repo source regime: each query drops at most one sensor (uniform over none + 5 singles)."""
    m = np.ones((n, q, P), dtype='float32')
    drop = rng.integers(0, P + 1, (n, q))
    ii, jj = np.nonzero(drop < P)
    m[ii, jj, drop[ii, jj]] = 0
    return m


# ---------------------------------------------------------------- exact, privileged references
GRID = np.linspace(-7, 7, 2801); DY = GRID[1] - GRID[0]


def oracle(prm, t, S, raw_qx, qy):
    """Exact posterior of y given x_S for task t (known parameters). Returns -log p(y_true), mean, var."""
    a, b, f, d, s = (prm[k][t, S] for k in ('a', 'b', 'f', 'd', 's'))
    nl = prm['nonlin'][t]
    cov = np.outer(d, d) + np.diag(s ** 2)
    prec = np.linalg.inv(cov); logdet = np.linalg.slogdet(cov)[1]; k = len(S)

    def loglik(yv, xs):
        r = xs - (yv[..., None] * a + b + nl * np.sin(yv[..., None] * f))
        return -.5 * (k * LOG2PI + logdet + np.einsum('...i,ij,...j->...', r, prec, r))

    xs = raw_qx[:, S]
    ll = loglik(GRID[None], xs[:, None]) - .5 * (LOG2PI + GRID ** 2)[None]
    mx = ll.max(1, keepdims=True); w = np.exp(ll - mx)
    logz = np.log(w.sum(1) * DY) + mx[:, 0]
    post = w / w.sum(1, keepdims=True)
    pm = (post * GRID).sum(1); pv = (post * (GRID - pm[:, None]) ** 2).sum(1)
    return -(loglik(qy, xs) - .5 * (LOG2PI + qy ** 2) - logz), pm, pv


def pop_linear(prm, t, S):
    """Best linear predictor with known population moments (infinite-support joint Gaussian)."""
    a, b, f, d, s = (prm[k][t] for k in ('a', 'b', 'f', 'd', 's')); nl = prm['nonlin'][t]
    eys = f * np.exp(-f ** 2 / 2)
    cxy = a + nl * eys
    F1, F2 = np.meshgrid(f, f, indexing='ij')
    ess = .5 * (np.exp(-(F1 - F2) ** 2 / 2) - np.exp(-(F1 + F2) ** 2 / 2))
    cxx = np.outer(a, a) + nl * (np.outer(a, eys) + np.outer(eys, a)) + nl ** 2 * ess + np.outer(d, d) + np.diag(s ** 2)
    cxx, cxy = cxx[np.ix_(S, S)], cxy[S]
    beta = np.linalg.solve(cxx, cxy)
    return beta, b[S], 1 - cxy @ beta


# ---------------------------------------------------------------- support-only closed-form baselines
def gnll(mu, var, y):
    return .5 * (LOG2PI + np.log(var) + (y - mu) ** 2 / var)


def ridge_repo(sx, sy, qx, S):
    """Exactly the repo's support ridge: mean-imputed support, penalty 1, LOO residual variance."""
    design = np.column_stack((np.ones(len(sy)), sx[:, S]))
    pen = np.eye(design.shape[1]); pen[0, 0] = 0
    inv = np.linalg.inv(design.T @ design + pen); w = inv @ design.T @ sy
    lev = np.sum((design @ inv) * design, axis=1)
    loo = (sy - design @ w) / (1 - lev).clip(.05)
    var = max(float((loo ** 2).mean()), .01)
    return np.column_stack((np.ones(len(qx)), qx[:, S])) @ w, np.full(len(qx), var)


def ridge_cc(sx, sy, sm, qx, S):
    """Complete-case ridge: only support rows where every sensor in S is observed."""
    cc = sm[:, S].all(1)
    if cc.sum() < len(S) + 3:
        return ridge_repo(sx, sy, qx, S)
    return ridge_repo(sx[cc], sy[cc], qx, S)


def em_gaussian(sx, sy, sm, iters=40):
    """ML joint Gaussian of (x_1..x_5, y) from incomplete support rows (y always observed)."""
    n, P = sx.shape
    Z = np.column_stack((sx, sy)).astype(float)
    O = np.column_stack((sm, np.ones(n))).astype(bool)
    mu = np.array([Z[O[:, j], j].mean() if O[:, j].any() else 0. for j in range(P + 1)])
    sig = np.diag([Z[O[:, j], j].var() + 1e-3 if O[:, j].sum() > 1 else 1. for j in range(P + 1)])
    pats = {}
    for i in range(n):
        pats.setdefault(O[i].tobytes(), []).append(i)
    pats = [(O[idx[0]], np.array(idx)) for idx in pats.values()]
    for _ in range(iters):
        Zc = Z.copy(); C = np.zeros((P + 1, P + 1))
        for o, idx in pats:
            mi = ~o
            if mi.any():
                K = sig[np.ix_(mi, o)] @ np.linalg.inv(sig[np.ix_(o, o)])
                Zc[np.ix_(idx, mi)] = mu[mi] + (Z[np.ix_(idx, o)] - mu[o]) @ K.T
                C[np.ix_(mi, mi)] += len(idx) * (sig[np.ix_(mi, mi)] - K @ sig[np.ix_(o, mi)])
        mu = Zc.mean(0)
        dev = Zc - mu
        sig = (dev.T @ dev + C) / n + 1e-4 * np.eye(P + 1)
    return mu, sig


def em_conditional(mu, sig, qx, S, n):
    S = list(S); P = len(mu) - 1
    beta = np.linalg.solve(sig[np.ix_(S, S)], sig[S, P])
    var = (sig[P, P] - sig[P, S] @ beta) * n / max(n - len(S) - 1, 1)
    return mu[P] + (qx[:, S] - mu[S]) @ beta, np.full(len(qx), max(var, 1e-3))


def blr_cc(sx, sy, sm, qx, S):
    """Bayesian linear regression on complete-case support rows, evidence-maximised (MacKay) precisions.
    Predictive variance includes parameter uncertainty."""
    from sklearn.linear_model import BayesianRidge
    S = list(S)
    cc = sm[:, S].all(1)
    if cc.sum() < len(S) + 3:
        cc = np.ones(len(sy), dtype=bool)
    m = BayesianRidge(fit_intercept=True, compute_score=False, max_iter=300)
    m.fit(sx[cc][:, S], sy[cc])
    mu, sd = m.predict(qx[:, S], return_std=True)
    return mu, np.maximum(sd, 1e-3) ** 2
