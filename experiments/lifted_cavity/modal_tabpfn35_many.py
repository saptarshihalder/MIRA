"""Protocol v13, execution: several networks at once in one Modal container (the workspace runs one GPU container at a
time, and one TabPFN-3.5 process uses about a quarter of an L4). Each network is its own tabpfn_colab.py process, with
exactly the arguments of modal_tabpfn35.tabpfn, so the outputs are the same as one network per container. Self-contained:
only this file is importable as a module inside the container.

    python -m modal deploy modal_tabpfn35_many.py
    python -c "import modal; print(modal.Function.from_name('mira-v13-tabpfn35-many', 'tabpfn_many').spawn(['usgs_wa', 'dwd_p', 'aurn_pm25']).object_id)"
"""
import os, subprocess, sys
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CODE = '/repo/experiments/lifted_cavity'           # realdata.py resolves paths two levels up
GPU = os.environ.get('MIRA_GPU', 'L4')
VERSION = 'v3.5'
app = modal.App('mira-v13-tabpfn35-many')
vol = modal.Volume.from_name('mira-v10')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('tabpfn==9.1.0', 'numpy', 'pandas', 'scipy', 'scikit-learn')
         .add_local_dir(str(HERE), CODE, ignore=lambda p: p.suffix not in ('.py',)))
secret = modal.Secret.from_name('custom-secret')


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=6, memory=24576, timeout=12 * 3600)
def tabpfn_many(nets):
    tok = (os.environ.get('tabpfn_auth') or os.environ.get('TABPFN_TOKEN') or '').strip()
    if not tok:
        raise RuntimeError("Modal secret 'custom-secret' has no key tabpfn_auth")
    vol.reload()
    procs = {}
    for net in nets:
        out = f'/vol/runs_real/{net}/tabpfn_v3_5'
        os.makedirs(out, exist_ok=True)
        log = open(f'/tmp/{net}.log', 'w')
        procs[net] = (subprocess.Popen([sys.executable, 'tabpfn_colab.py', '--panel', f'/vol/panels/real_{net}_s10101.pt', '--out', out,
                                        '--version', VERSION, '--allow-constant-support', '--crps', '--device', 'cuda'], cwd=CODE,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'TABPFN_TOKEN': tok, 'OMP_NUM_THREADS': '2'}), log)
    res = {}
    for net, (p, log) in procs.items():
        rc = p.wait(); log.close()
        out = f'/vol/runs_real/{net}/tabpfn_v3_5'
        res[net] = dict(rc=rc, files=sorted(os.listdir(out)), log=open(f'/tmp/{net}.log').read()[-2000:])
    vol.commit()
    return res
