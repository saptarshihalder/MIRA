"""Protocol v15: TabPFN-3.5's 4-fold cross-fit on the four v10 panels, on Modal, in one L4 container (the workspace runs
one GPU container at a time): one tabpfn_cfk.py process per (network, half of the episodes). Self-contained: only this
file is importable inside the container.

    python -m modal deploy modal_tabpfn35_cfk.py
    python -c "import modal; print(modal.Function.from_name('mira-v15-cfk', 'cfk_many').spawn(['usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25']).object_id)"
Outputs: /vol/runs_real/<net>/tabpfn_v3_5/pqk4_real_<net>_s10101_o<i>s2.npz (+ .json). These are cross-fit outputs on
the support rows only; no test query is scored here.
"""
import os, subprocess, sys
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CODE = '/repo/experiments/lifted_cavity'
GPU = os.environ.get('MIRA_GPU', 'L4')
app = modal.App('mira-v15-cfk')
vol = modal.Volume.from_name('mira-v10')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('tabpfn==9.1.0', 'numpy', 'pandas', 'scipy', 'scikit-learn')
         .add_local_dir(str(HERE), CODE, ignore=lambda p: p.suffix not in ('.py',)))
secret = modal.Secret.from_name('custom-secret')
PARTS = 2


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=2, memory=8192, timeout=1200)
def smoke(net):
    """Two episodes into /tmp (partial_ file, not on the volume): the script runs and the outputs have the right shape."""
    import numpy as np
    tok = (os.environ.get('tabpfn_auth') or '').strip()
    r = subprocess.run([sys.executable, 'tabpfn_cfk.py', '--panel', f'/vol/panels/real_{net}_s10101.pt', '--out', '/tmp/smoke', '--version', 'v3.5',
                        '--k', '4', '--limit', '2', '--device', 'cuda'], cwd=CODE, capture_output=True, text=True,
                       env={**os.environ, 'TABPFN_TOKEN': tok})
    f = [p for p in os.listdir('/tmp/smoke') if p.startswith('partial_')] if os.path.isdir('/tmp/smoke') else []
    info = {}
    if f:
        z = np.load(f'/tmp/smoke/{f[0]}')
        info = {k: dict(shape=z[k].shape, finite_rows0=int(np.isfinite(z[k][0]).sum()), mean0=float(np.nanmean(z[k][0]))) for k in z.files}
    return dict(rc=r.returncode, tail=(r.stdout + r.stderr)[-1500:], files=f, info=info)


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=8, memory=32768, timeout=12 * 3600)
def cfk_many(nets):
    tok = (os.environ.get('tabpfn_auth') or os.environ.get('TABPFN_TOKEN') or '').strip()
    if not tok:
        raise RuntimeError("Modal secret 'custom-secret' has no key tabpfn_auth")
    vol.reload()
    procs = {}
    for net in nets:
        out = f'/vol/runs_real/{net}/tabpfn_v3_5'
        os.makedirs(out, exist_ok=True)
        for o in range(PARTS):
            log = open(f'/tmp/{net}_o{o}.log', 'w')
            procs[(net, o)] = (subprocess.Popen([sys.executable, 'tabpfn_cfk.py', '--panel', f'/vol/panels/real_{net}_s10101.pt', '--out', out,
                                                 '--version', 'v3.5', '--k', '4', '--stride', str(PARTS), '--offset', str(o), '--device', 'cuda'],
                                                cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, 'TABPFN_TOKEN': tok, 'OMP_NUM_THREADS': '1'}), log)
    res = {}
    for (net, o), (p, log) in procs.items():
        rc = p.wait(); log.close()
        res[f'{net}_o{o}'] = dict(rc=rc, log=open(f'/tmp/{net}_o{o}.log').read()[-1500:],
                                  files=sorted(f for f in os.listdir(f'/vol/runs_real/{net}/tabpfn_v3_5') if f.startswith('pqk4')))
    vol.commit()
    return res
