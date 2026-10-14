"""Protocol v10 variant selection on development data (docs/LIFTED_CAVITY_PROTOCOL_V10.md), run before any v10 score.

    python select10.py --v9 /content/drive/MyDrive/mira_v9 --out <mira_v10>/dev/selection10.json
Development networks: the three protocol-v9 networks (DWD temperature, AURN NO2, USGS Pennsylvania), now used, extra 0,
test-time ensemble of 8. Candidates (both with the scale-free pipeline SF: per-episode target standardization and
log-variance bound 10, and the UPT* objective NLL + CRPS):
    E  upt_L_sf    (two-component pool: lifted site + one free Gaussian)
    F  upt_L_sfm   (lifted site + a three-component free mixture)
Per-episode NLL is averaged over seeds 1-3, then over all episodes. Rule: F if its mean NLL is lower than E's by at
least .005 nats, else E (the simpler model). Other models are reported for information only.
"""
import argparse, json
from pathlib import Path
import numpy as np

NETS = ('dwd_t2m', 'aurn_no2', 'usgs_q')
SEED = 9091
SEEDS = (1, 2, 3)
CAND = {'E': 'upt_L_sf', 'F': 'upt_L_sfm'}
INFO = {'UPT* (v9)': ('upt_L_crps_s{}_ft', '_tta8'), 'PFN-L-SF': ('pfn_L_sf_s{}_ft', '_tta8'), 'PFN-L (v9)': ('pfn_L_s{}_ft', '_tta8'),
        'TabPFN v2': ('tabpfn_v2', '')}


def nll(root, run, sfx, net):
    if '{}' in run:
        return np.mean([np.load(root / 'runs_real' / net / run.format(s) / f'cells_real_{net}_s{SEED}{sfx}.npz')['e0_nll'] for s in SEEDS], 0)
    return np.load(root / 'runs_real' / net / run / f'cells_real_{net}_s{SEED}{sfx}.npz')['e0_nll']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v9', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.v9)
    res = {}
    for c, m in CAND.items():
        v = np.concatenate([nll(root, m + '_s{}_ft', '_tta8', n) for n in NETS])
        assert np.isfinite(v).all(), c
        res[c] = dict(model=m, nll=float(v.mean()), per_network={n: float(nll(root, m + '_s{}_ft', '_tta8', n).mean()) for n in NETS})
    pick = 'F' if res['F']['nll'] <= res['E']['nll'] - .005 else 'E'
    info = {}
    for k, (run, sfx) in INFO.items():
        try:
            info[k] = {n: float(np.nanmean(nll(root, run, sfx, n))) for n in NETS}
        except FileNotFoundError:
            info[k] = None
    out = dict(candidates=res, selected=pick, model=CAND[pick], cells_suffix='_tta8', information=info)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
