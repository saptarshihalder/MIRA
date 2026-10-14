"""Protocols v16 and v17 (seed 17171): TabPFN-3.5 on the fresh panels, on Modal, in one L4 container: per network, one tabpfn_colab.py
process (test predictions, CRPS, per-query outputs and the 2-fold cross-fit, exactly as v13) and one tabpfn_cfk.py
process (4-fold cross-fit, as v15). Self-contained: only this file is importable inside the container.

    python -m modal volume put mira-v10 <local panel> panels/real_<net>_s16161.pt      (the four panels)
    python -m modal deploy modal_tabpfn35_v16.py
    python -c "import modal; print(modal.Function.from_name('mira-v16-tab', 'tab16').spawn(['usgs_ny_f', 'usgs_wa_f', 'dwd_p_f', 'aurn_pm25_f']).object_id)"
Outputs: /vol/runs_real/<net>/tabpfn_v3_5/ (cells_*, pq_*, pqk4_*, train.json).
"""
import os, subprocess, sys
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CODE = '/repo/experiments/lifted_cavity'
app = modal.App('mira-v16-tab')
vol = modal.Volume.from_name('mira-v10')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('tabpfn==9.1.0', 'numpy', 'pandas', 'scipy', 'scikit-learn')
         .add_local_dir(str(HERE), CODE, ignore=lambda p: p.suffix not in ('.py',)))
secret = modal.Secret.from_name('custom-secret')


WARM = ("import numpy as np; from tabpfn_colab import make_regressor; r = make_regressor('cuda', 'v3.5'); "
        "r.fit(np.random.randn(30, 3), np.random.randn(30)); r.predict(np.random.randn(5, 3)); print('WARM_OK')")


def _warm(env):
    """Download TabPFN-3.5's weights once, into the volume's cache, with retries (parallel first downloads can fail)."""
    import time
    for i in range(6):
        r = subprocess.run([sys.executable, '-c', WARM], cwd=CODE, capture_output=True, text=True, env=env)
        if 'WARM_OK' in r.stdout:
            vol.commit()
            return
        time.sleep(20 * (i + 1))
    raise RuntimeError('TabPFN-3.5 weights could not be downloaded: ' + (r.stdout + r.stderr)[-1200:])


@app.function(image=image, volumes={'/vol': vol}, cpu=4, memory=16384, timeout=3 * 3600)
def build(name, seed, data_dir):
    """Build one panel on the volume from a snapshot folder on the volume (eval_real.py build); returns its digest."""
    vol.reload()
    out = f'/vol/panels/real_{name}_s{seed}.pt'
    r = subprocess.run([sys.executable, 'eval_real.py', 'build', '--dataset', name, '--seed', str(seed), '--v6', data_dir, '--out', out],
                       cwd=CODE, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-1500:])
    vol.commit()
    sys.path.insert(0, CODE)
    import panel_digest
    return panel_digest.digest(out)


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu='L4', cpu=8, memory=32768, timeout=8 * 3600)
def tab16(nets, seed=16161):
    tok = (os.environ.get('tabpfn_auth') or os.environ.get('TABPFN_TOKEN') or '').strip()
    if not tok:
        raise RuntimeError("Modal secret 'custom-secret' has no key tabpfn_auth")
    vol.reload()
    os.makedirs('/vol/tabpfn_cache', exist_ok=True)
    env = {**os.environ, 'TABPFN_TOKEN': tok, 'OMP_NUM_THREADS': '1', 'TABPFN_MODEL_CACHE_DIR': '/vol/tabpfn_cache'}
    _warm(env)
    procs = {}
    for net in nets:
        panel, out = f'/vol/panels/real_{net}_s{seed}.pt', f'/vol/runs_real/{net}/tabpfn_v3_5'
        os.makedirs(out, exist_ok=True)
        for kind, cmd in (('main', ['tabpfn_colab.py', '--panel', panel, '--out', out, '--version', 'v3.5', '--allow-constant-support',
                                    '--crps', '--device', 'cuda']),
                          ('cfk', ['tabpfn_cfk.py', '--panel', panel, '--out', out, '--version', 'v3.5', '--k', '4', '--device', 'cuda'])):
            log = open(f'/tmp/{net}_{kind}.log', 'w')
            procs[(net, kind)] = (subprocess.Popen([sys.executable] + cmd, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env=env), log)
    res = {}
    for (net, kind), (p, log) in procs.items():
        rc = p.wait(); log.close()
        res[f'{net}|{kind}'] = dict(rc=rc, log=open(f'/tmp/{net}_{kind}.log').read()[-1500:],
                                    files=sorted(os.listdir(f'/vol/runs_real/{net}/tabpfn_v3_5')))
    vol.commit()
    return res
