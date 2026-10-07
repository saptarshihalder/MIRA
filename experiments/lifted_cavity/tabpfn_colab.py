"""Score the pretrained TabPFN v2 regressor on the evaluation panels (run where Hugging Face is reachable, e.g. Colab).

    pip install tabpfn torch numpy scipy scikit-learn pandas
    # 1. regenerate the synthetic panels deterministically (no data needed), e.g. F1, F3, G1, G3:
    python evaluate2.py panel --seed 20261201 --tasks 256 --sensors 5
    python evaluate2.py panel --seed 20261203 --tasks 128 --sensors 16
    python evaluate2.py panel --seed 20261301 --tasks 256 --sensors 5
    python evaluate2.py panel --seed 20261303 --tasks 128 --sensors 16
    # (real panels need the UCI files: eval_real.py build --dataset beijing --seed 2027 --out cache/real_beijing_s2027.pt)
    # 2. score every panel in the cache folder:
    python tabpfn_colab.py --panels-dir $MIRA_LC_CACHE --out runs/tabpfn_v2

For each task and mask, TabPFN is fitted in context on the support rows (missing entries as NaN, which TabPFN handles
natively) and predicts the queries with unobserved sensors set to NaN. NLL is TabPFN's own predictive density (bar
distribution in raw target units, checked against its predictive mean), so it is directly comparable with the NLL of
every other method. Results use the same cells format as every other run; copy runs/tabpfn_v2 into
artifacts/reports/lifted_cavity_paper/runs/ and make_paper.py adds a 'TabPFN v2 (pretrained)' row.
The model version is pinned with --version (default v2); newer TabPFN releases may require accepting a licence.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def make_regressor(device, version):
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion(version), device=device)
    except (ImportError, AttributeError):          # tabpfn 2.x: the default model is v2
        return TabPFNRegressor(device=device)


def tabpfn_nll(reg, Xs, ys, Xq, yq):
    reg.fit(Xs, ys)
    out = reg.predict(Xq, output_type='full')
    crit, logits = out['criterion'], torch.as_tensor(out['logits'])
    mean = np.asarray(out['mean'], dtype=float)
    with torch.no_grad():
        crit = crit.to(logits.device)
        m = crit.mean(logits).double().cpu().numpy().reshape(-1)
        if not np.allclose(m, mean, atol=1e-3 * (1 + np.abs(mean).max())):
            raise RuntimeError('TabPFN criterion is not in raw target units; NLL would be mis-scaled')
        nll = crit(logits, torch.as_tensor(yq, dtype=logits.dtype, device=logits.device))
    return np.asarray(nll.double().cpu().numpy(), dtype=float).reshape(-1), mean


def task_arrays(pool, t):
    sx = pool['sx'][t].numpy().astype(float); sm = pool['sm'][t].numpy() > 0
    return (np.where(sm, sx, np.nan), pool['sy'][t].numpy().astype(float),
            pool['qx'][t].numpy().astype(float), pool['qy'][t].numpy().astype(float))


def score_panel(reg, path, out):
    panel = torch.load(path, weights_only=False); pool = panel['pool']; tag = Path(path).stem
    if (out / f'cells_{tag}.npz').exists():
        return
    res = {}
    if 'banks' in panel:                            # synthetic: every task x every mask in each bank
        for k, bank in panel['banks'].items():
            nll = np.zeros((pool['n'], len(bank))); se = np.zeros_like(nll)
            for t in range(pool['n']):
                Xs, ys, qx, qy = task_arrays(pool, t)
                for i, msk in enumerate(bank):
                    l, mu = tabpfn_nll(reg, Xs, ys, np.where(np.asarray(msk)[None] > 0, qx, np.nan), qy)
                    nll[t, i] = l.mean(); se[t, i] = ((mu - qy) ** 2).mean()
            res[f'k{k}_nll'] = nll; res[f'k{k}_se'] = se
            print(tag, 'k', k, round(float(nll.mean()), 4), flush=True)
    else:                                           # real: natural mask x extra dropped sensors, per query
        for e, qm in panel['conds'].items():
            nll = np.zeros(pool['n']); se = np.zeros(pool['n'])
            for t in range(pool['n']):
                Xs, ys, qx, qy = task_arrays(pool, t)
                l, mu = tabpfn_nll(reg, Xs, ys, np.where(qm[t] > 0, qx, np.nan), qy)
                nll[t] = l.mean(); se[t] = ((mu - qy) ** 2).mean()
            res[f'e{e}_nll'] = nll; res[f'e{e}_se'] = se
            print(tag, 'extra', e, round(float(nll.mean()), 4), flush=True)
    np.savez_compressed(out / f'cells_{tag}.npz', **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', nargs='*', default=[]); ap.add_argument('--panels-dir')
    ap.add_argument('--out', required=True); ap.add_argument('--version', default='v2')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'train.json').write_text(json.dumps(dict(model='tabpfn_v2', version=a.version,
                                                    note='pretrained TabPFN regressor, in context, no fine-tuning')))
    reg = make_regressor(a.device, a.version)
    paths = list(a.panel)
    if a.panels_dir:
        paths += sorted(str(p) for p in Path(a.panels_dir).glob('v2_s*.pt')) + sorted(str(p) for p in Path(a.panels_dir).glob('real_*.pt'))
    for p in paths:
        score_panel(reg, p, out)


if __name__ == '__main__':
    main()
