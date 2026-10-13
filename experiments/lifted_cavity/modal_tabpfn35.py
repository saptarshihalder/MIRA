"""Protocol v13: TabPFN-3.5 on the four v10 panels, on Modal, exactly as TabPFN v2 in v10/v11 (modal_tabpfn.py) except
for the model version and the GPU (L4 by default, for the budget; MIRA_GPU at deploy time overrides it).

    python -m modal deploy modal_tabpfn35.py
    python modal_spawn.py smoke usgs_wa mira-v13-tabpfn35        synthetic inputs only: license, raw units, time per call
    python modal_spawn.py tabpfn usgs_ny,usgs_wa,dwd_p,aurn_pm25 mira-v13-tabpfn35
Panels are the v10 panels on the volume 'mira-v10' (panels/real_<net>_s10101.pt). The license token comes from the Modal
secret 'custom-secret' (key tabpfn_auth) and reaches tabpfn as TABPFN_TOKEN; it is never printed. Outputs:
runs_real/<net>/tabpfn_v3_5/ on the volume.
"""
import os, subprocess, sys, time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CODE = '/repo/experiments/lifted_cavity'           # realdata.py resolves paths two levels up
GPU = os.environ.get('MIRA_GPU', 'L4')
VERSION = 'v3.5'
app = modal.App('mira-v13-tabpfn35')
vol = modal.Volume.from_name('mira-v10')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('tabpfn==9.1.0', 'numpy', 'pandas', 'scipy', 'scikit-learn')
         .add_local_dir(str(HERE), CODE, ignore=lambda p: p.suffix not in ('.py',)))
secret = modal.Secret.from_name('custom-secret')


def _auth():
    tok = (os.environ.get('tabpfn_auth') or os.environ.get('TABPFN_TOKEN') or '').strip()
    if not tok:
        raise RuntimeError("Modal secret 'custom-secret' has no key tabpfn_auth")
    os.environ['TABPFN_TOKEN'] = tok


def _panel(net):
    return f'/vol/panels/real_{net}_s10101.pt'


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=2, memory=8192, timeout=1800)
def smoke(net):
    """Synthetic inputs shaped like an episode of `net` (no real value is read): the model loads, its predictive is in
    raw target units (tabpfn_nll checks this), and the time of a main call (48 support rows) and a cross-fit call (24)."""
    _auth(); sys.path.insert(0, CODE)
    import numpy as np, torch
    import tabpfn_colab as tc
    pool = torch.load(_panel(net), weights_only=False)['pool']
    n, NS, D = pool['sx'].shape; NQ = pool['qx'].shape[1]
    t0 = time.time(); reg = tc.make_regressor('cuda', VERSION); load = time.time() - t0
    rng = np.random.default_rng(0); out = dict(net=net, episodes=int(n), NS=int(NS), NQ=int(NQ), D=int(D), gpu=torch.cuda.get_device_name(0))
    b = rng.normal(size=D)
    for name, ns, nq in (('main', NS, 3 * NQ), ('crossfit', NS // 2, 3 * (NS - NS // 2))):
        ts = []
        for r in range(4):
            X = rng.normal(size=(ns + nq, D)); y = X @ b + 3 + .3 * rng.normal(size=ns + nq)
            X[rng.random(X.shape) < .3] = np.nan
            t1 = time.time(); tc.tabpfn_nll(reg, X[:ns], y[:ns], X[ns:], y[ns:], quant=(name == 'main')); ts.append(time.time() - t1)
        out[name + '_s'] = [round(t, 3) for t in ts]
    out['first_load_s'] = round(load, 1)
    out['episode_s'] = round(float(np.median(out['main_s'][1:]) + 2 * np.median(out['crossfit_s'][1:])), 3)
    out['network_h'] = round(out['episode_s'] * n / 3600, 2)
    return out


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=2, memory=8192, timeout=10 * 3600)
def tabpfn(net):
    _auth(); vol.reload()
    out = f'/vol/runs_real/{net}/tabpfn_v3_5'
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, 'tabpfn_colab.py', '--panel', _panel(net), '--out', out, '--version', VERSION,
                    '--allow-constant-support', '--crps', '--device', 'cuda'], cwd=CODE, check=True)
    vol.commit()
    return net, sorted(os.listdir(out))
