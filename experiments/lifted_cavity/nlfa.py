"""Plug-in nonlinear supervised factor model with exact 1-D grid inference (support-only, closed form per task).

Per task, from the labeled support rows only:
  * each sensor j:  x_j = g_j(y) + e_j, with g_j a cubic in the standardised target whose curvature terms are
    ridge-penalised (lambda = 100, tuned on development tasks; the intercept and slope are unpenalised) and
    clamped to the support's target range (linear extrapolation);
  * residual covariance by EM over the incomplete residuals, then a K-factor structure  D D^T + diag(psi);
  * prediction: p(y | x_S) on a fine grid with prior N(mean_y, var_y) and likelihood N(x_S; g_S(y), Sigma_SS),
    exact up to quadrature for every missingness pattern S.
"""
import numpy as np

LOG2PI = np.log(2 * np.pi)


def _em_cov(R, M, iters=30, ridge=1e-4):
    n, P = R.shape
    mu = np.array([R[M[:, j], j].mean() if M[:, j].any() else 0. for j in range(P)])
    sig = np.diag([R[M[:, j], j].var() + 1e-3 if M[:, j].sum() > 1 else 1. for j in range(P)])
    pats = {}
    for i in range(n):
        pats.setdefault(M[i].tobytes(), []).append(i)
    pats = [(M[idx[0]], np.array(idx)) for idx in pats.values()]
    for _ in range(iters):
        Z = R.copy(); C = np.zeros((P, P))
        for o, idx in pats:
            mi = ~o
            if not mi.any():
                continue
            if o.any():
                K = sig[np.ix_(mi, o)] @ np.linalg.inv(sig[np.ix_(o, o)])
                Z[np.ix_(idx, mi)] = mu[mi] + (R[np.ix_(idx, o)] - mu[o]) @ K.T
                C[np.ix_(mi, mi)] += len(idx) * (sig[np.ix_(mi, mi)] - K @ sig[np.ix_(o, mi)])
            else:
                Z[np.ix_(idx, mi)] = mu[mi]
                C[np.ix_(mi, mi)] += len(idx) * sig[np.ix_(mi, mi)]
        mu = Z.mean(0)
        d = Z - mu
        sig = (d.T @ d + C) / n + ridge * np.eye(P)
    return mu, (sig + sig.T) / 2


def _factor(S, K, iters=50, floor=1e-3):
    psi = np.clip(np.diag(S) * .5, floor, None)
    if K == 0:
        return np.zeros((len(S), 0)), np.clip(np.diag(S), floor, None)
    for _ in range(iters):
        vals, vecs = np.linalg.eigh(S - np.diag(psi))
        top = np.argsort(vals)[::-1][:K]
        D = vecs[:, top] * np.sqrt(np.clip(vals[top], 0, None))
        psi = np.clip(np.diag(S) - (D * D).sum(1), floor, None)
    return D, psi


def fit(sx, sy, sm, degrees=(3,), lam=100., K=1):
    sm = sm.astype(bool)
    n, P = sx.shape
    my, sdy = float(sy.mean()), float(max(sy.std(), .1))
    yt = (sy - my) / sdy
    W, R = [], np.zeros_like(sx, dtype=float)
    dof = []
    lo, hi = float(yt.min()), float(yt.max())
    for j in range(P):
        o = sm[:, j]
        x = sx[o, j].astype(float)
        best = None
        for q in degrees:
            if o.sum() < q + 3:
                continue
            Phi = basis(yt[o], q, lo, hi)
            pen = lam * np.eye(q + 1); pen[0, 0] = 0; pen[1, 1] = 0
            Ainv = np.linalg.inv(Phi.T @ Phi + pen)
            w = Ainv @ Phi.T @ x
            h = np.sum((Phi @ Ainv) * Phi, 1)
            loo = np.mean(((x - Phi @ w) / np.clip(1 - h, .05, None)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, q, w, Phi)
        if best is None:                                            # too few rows: intercept only
            w = np.array([x.mean() if len(x) else 0.]); q = 0; Phi = np.ones((o.sum(), 1))
        else:
            _, q, w, Phi = best
        W.append(np.pad(w, (0, 4 - len(w))))
        R[o, j] = x - Phi @ w
        dof.append(q + 1)
    mu_r, S = _em_cov(R, sm)
    D, psi = _factor(S, K)
    infl = n / max(n - np.mean(dof) - 1, 1)
    return dict(my=my, sdy=sdy, W=np.array(W), mu_r=mu_r, D=D * np.sqrt(infl), psi=psi * infl, lo=lo, hi=hi)


def basis(yt, q, lo, hi):
    """1, y, then powers of y clamped to the support range: linear extrapolation beyond the observed targets."""
    yc = np.clip(yt, lo, hi)
    cols = [np.ones_like(yt), yt] + [yc ** p for p in range(2, q + 1)]
    return np.stack(cols[:q + 1], -1)


GRID = np.linspace(-5, 5, 401)


def predict_nll(model, qx, qy, S):
    """Exact (grid) predictive NLL of qy given qx[:, S] for one task and one mask."""
    S = np.asarray(S)
    W, D, psi, mr = model['W'][S], model['D'][S], model['psi'][S], model['mu_r'][S]
    Sig = D @ D.T + np.diag(psi)
    prec = np.linalg.inv(Sig); logdet = np.linalg.slogdet(Sig)[1]
    k = len(S)

    def loglik(yt, xs):                                            # yt (...,), xs (..., k)
        B = basis(np.asarray(yt, dtype=float), 3, model['lo'], model['hi'])
        G = np.einsum('...p,kp->...k', B, W)
        r = xs - G - mr
        return -.5 * (k * LOG2PI + logdet + np.einsum('...i,ij,...j->...', r, prec, r)) - .5 * (LOG2PI + yt ** 2)

    xs = qx[:, S].astype(float)
    ll = loglik(GRID[None, :], xs[:, None, :])
    mx = ll.max(1, keepdims=True)
    logz = np.log(np.trapezoid(np.exp(ll - mx), GRID, axis=1)) + mx[:, 0]
    yt = (qy - model['my']) / model['sdy']
    return -(loglik(yt, xs) - logz - np.log(model['sdy']))
