"""Task pools (with precomputed support-only anchors) and batch construction."""
import numpy as np
import torch
import family as fam
import anchors as an

KS = (0, 1, 2)


def build_pool(n, seed, nonlin=.4, support_missing=.2, support_rows=48, chunk=2000, keep_truth=False, P=5):
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, n, chunk):
        T = fam.make_tasks(rng, min(chunk, n - start), nonlin=nonlin, support_missing=support_missing, support_rows=support_rows, P=P)
        mu, sig = an.em_batched(T['sx'], T['sy'], T['sm'])
        T['anchors'] = {K: {k: v.float() for k, v in an.fa_batched(mu, sig, K).items()} for K in KS}
        T['em_mu'], T['em_sig'] = mu.float(), sig.float()
        if not keep_truth:
            T.pop('raw_qx'); T.pop('raw_sx'); T.pop('prm')
        parts.append(T)
    pool = {}
    for key in ('c', 'sx', 'sy', 'sm', 'qx', 'qy'):
        pool[key] = torch.as_tensor(np.concatenate([p[key] for p in parts]))
    pool['anchors'] = {K: {k: torch.cat([p['anchors'][K][k] for p in parts]) for k in parts[0]['anchors'][K]} for K in KS}
    pool['em_mu'] = torch.cat([p['em_mu'] for p in parts]); pool['em_sig'] = torch.cat([p['em_sig'] for p in parts])
    if keep_truth:
        pool['raw_qx'] = np.concatenate([p['raw_qx'] for p in parts])
        pool['prm'] = {k: np.concatenate([p['prm'][k] for p in parts]) for k in parts[0]['prm']}
    pool['n'] = n
    return pool


def subset(pool, idx):
    idx = torch.as_tensor(idx)
    out = {k: pool[k][idx] for k in ('c', 'sx', 'sy', 'sm', 'qx', 'qy')}
    out['anchors'] = {K: {k: v[idx] for k, v in pool['anchors'][K].items()} for K in KS}
    return out


def train_batch(pool, rng, B=128, Q=8):
    """Fresh-ish source minibatch: B tasks, Q queries each, repo source masks (at most one sensor missing)."""
    t = rng.choice(pool['n'], B, replace=False)
    batch = subset(pool, t)
    q = torch.as_tensor(rng.integers(0, pool['qx'].shape[1], (B, Q)))
    ar = torch.arange(B)[:, None]
    batch['qx'] = batch['qx'][ar, q].reshape(B * Q, -1)
    batch['qy'] = batch['qy'][ar, q].reshape(-1)
    batch['qm'] = torch.as_tensor(fam.source_query_masks(rng, B, Q, pool['qx'].shape[-1])).reshape(B * Q, -1)
    batch['tid'] = torch.arange(B).repeat_interleave(Q)
    return batch


def eval_batches(pool, masks, tasks_per_batch=16):
    """Every task x every mask in `masks` x every query. Yields (task_idx, mask_idx, batch)."""
    masks = torch.as_tensor(masks)
    nq = pool['qx'].shape[1]
    for start in range(0, pool['n'], tasks_per_batch):
        t = np.arange(start, min(start + tasks_per_batch, pool['n']))
        batch = subset(pool, t)
        B = len(t); M = len(masks)
        batch['qx'] = batch['qx'][:, None].expand(B, M, nq, -1).reshape(B * M * nq, -1)
        batch['qy'] = batch['qy'][:, None].expand(B, M, nq).reshape(-1)
        batch['qm'] = masks[None, :, None].expand(B, M, nq, -1).reshape(B * M * nq, -1).float()
        batch['tid'] = torch.arange(B).repeat_interleave(M * nq)
        yield t, batch
