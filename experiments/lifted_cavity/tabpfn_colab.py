"""Score the pretrained TabPFN v2 regressor on the protocol-v2 panels (run where Hugging Face is reachable, e.g. Colab).

    pip install "tabpfn>=2.0" torch numpy scipy scikit-learn pandas
    python evaluate2.py panel --seed 20261201 --tasks 256 --sensors 5      # regenerates F1 deterministically
    python tabpfn_colab.py --panel <cache>/v2_s20261201_n256_p5_nl0.4_sr48.pt --out runs/tabpfn_v2
    python tabpfn_colab.py --real <cache>/real_beijing_s2027.pt --out runs/tabpfn_v2

For each task and mask, TabPFN is fitted in context on the 48 support rows (missing entries as NaN, which TabPFN
handles natively) and predicts the queries with unobserved sensors set to NaN. NLL uses TabPFN's own predictive
(bar distribution), so it is directly comparable with the other methods' NLL. Results are written in the same cells
format as every other run, so make_paper.py and confirm2.py pick them up as the model name 'tabpfn_v2'.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def tabpfn_nll(reg, Xs, ys, Xq, yq):
    reg.fit(Xs, ys)
    out = reg.predict(Xq, output_type='full')
    crit, logits = out['criterion'], out['logits']
    with torch.no_grad():
        nll = crit(torch.as_tensor(logits), torch.as_tensor(yq, dtype=torch.float32).to(torch.as_tensor(logits).device))
    return np.asarray(nll.detach().cpu().numpy(), dtype=float).reshape(-1), np.asarray(out['mean'], dtype=float)


def main():
    from tabpfn import TabPFNRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel'); ap.add_argument('--real'); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'train.json').write_text(json.dumps(dict(model='tabpfn_v2', note='pretrained TabPFN v2 regressor, in-context, no fine-tuning')))
    reg = TabPFNRegressor(device=a.device)
    if a.panel:
        panel = torch.load(a.panel, weights_only=False); pool = panel['pool']; tag = Path(a.panel).stem
        res = {}
        for k, bank in panel['banks'].items():
            nll = np.zeros((pool['n'], len(bank))); se = np.zeros_like(nll)
            for t in range(pool['n']):
                sx = pool['sx'][t].numpy().astype(float); sm = pool['sm'][t].numpy() > 0
                Xs = np.where(sm, sx, np.nan); ys = pool['sy'][t].numpy().astype(float)
                qx, qy = pool['qx'][t].numpy().astype(float), pool['qy'][t].numpy().astype(float)
                for i, m in enumerate(bank):
                    Xq = np.where(m[None] > 0, qx, np.nan)
                    l, mu = tabpfn_nll(reg, Xs, ys, Xq, qy)
                    nll[t, i] = l.mean(); se[t, i] = ((mu - qy) ** 2).mean()
            res[f'k{k}_nll'] = nll; res[f'k{k}_se'] = se
            print(tag, 'k', k, nll.mean(), flush=True)
        np.savez_compressed(out / f'cells_{tag}.npz', **res)
    if a.real:
        panel = torch.load(a.real, weights_only=False); pool = panel['pool']; tag = Path(a.real).stem
        res = {}
        for e, qm in panel['conds'].items():
            nll = np.zeros(pool['n']); se = np.zeros(pool['n'])
            for t in range(pool['n']):
                sx = pool['sx'][t].numpy().astype(float); sm = pool['sm'][t].numpy() > 0
                Xs = np.where(sm, sx, np.nan); ys = pool['sy'][t].numpy().astype(float)
                qx, qy = pool['qx'][t].numpy().astype(float), pool['qy'][t].numpy().astype(float)
                Xq = np.where(qm[t] > 0, qx, np.nan)
                l, mu = tabpfn_nll(reg, Xs, ys, Xq, qy)
                nll[t] = l.mean(); se[t] = ((mu - qy) ** 2).mean()
            res[f'e{e}_nll'] = nll; res[f'e{e}_se'] = se
            print(tag, 'extra', e, nll.mean(), flush=True)
        np.savez_compressed(out / f'cells_{tag}.npz', **res)


if __name__ == '__main__':
    main()
