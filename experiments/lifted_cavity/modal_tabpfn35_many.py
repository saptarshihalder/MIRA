"""Protocol v13, execution: several networks at once in one Modal container (the workspace runs one GPU container at a
time, and one TabPFN-3.5 process uses about a quarter of an L4). Each network is its own tabpfn_colab.py process, with
exactly the arguments of modal_tabpfn35.tabpfn, so the outputs are the same as one network per container.

    python -m modal deploy modal_tabpfn35_many.py
    python -c "import modal; print(modal.Function.from_name('mira-v13-tabpfn35-many', 'tabpfn_many').spawn(['usgs_wa', 'dwd_p', 'aurn_pm25']).object_id)"
"""
import os, subprocess, sys
import modal
from modal_tabpfn35 import CODE, GPU, VERSION, _auth, _panel, image, secret, vol

app = modal.App('mira-v13-tabpfn35-many')


@app.function(image=image, volumes={'/vol': vol}, secrets=[secret], gpu=GPU, cpu=6, memory=24576, timeout=12 * 3600)
def tabpfn_many(nets):
    _auth(); vol.reload()
    procs = {}
    for net in nets:
        out = f'/vol/runs_real/{net}/tabpfn_v3_5'
        os.makedirs(out, exist_ok=True)
        log = open(f'/tmp/{net}.log', 'w')
        procs[net] = (subprocess.Popen([sys.executable, 'tabpfn_colab.py', '--panel', _panel(net), '--out', out, '--version', VERSION,
                                        '--allow-constant-support', '--crps', '--device', 'cuda'], cwd=CODE, stdout=log,
                                       stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '2'}), log)
    res = {}
    for net, (p, log) in procs.items():
        rc = p.wait(); log.close()
        out = f'/vol/runs_real/{net}/tabpfn_v3_5'
        res[net] = dict(rc=rc, files=sorted(os.listdir(out)), log=open(f'/tmp/{net}.log').read()[-2000:])
    vol.commit()
    return res
