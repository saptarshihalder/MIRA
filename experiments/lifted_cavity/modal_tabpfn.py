"""Protocol v10 amendment 5: TabPFN v2 on the fresh v10 panels (with the v11 per-query outputs and support cross-fit),
run on Modal instead of Colab.

    python -m modal run --detach modal_tabpfn.py --nets usgs_wa,dwd_p,aurn_pm25 --data <folder with <net>.npz>
Panels are rebuilt on Modal with eval_real.py from data snapshots whose array contents equal the Colab snapshots of
record (npz_digest.py); panel_digest.py checks the rebuilt panels against Colab's. TabPFN runs exactly as on Colab:
tabpfn 9.1.0, the ungated v2 weights, 8 estimators, --allow-constant-support --crps. Outputs go to the Modal volume
'mira-v10' (runs_real/<net>/tabpfn_v2/) and are copied back to Drive by colab_v10.py.
"""
import os, subprocess, sys
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = os.environ.get('MIRA_V10_DATA', '')
app = modal.App('mira-v10-tabpfn')
vol = modal.Volume.from_name('mira-v10', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('tabpfn==9.1.0', 'numpy', 'pandas', 'scipy', 'scikit-learn')
         .add_local_dir(str(HERE), '/code', ignore=lambda p: p.suffix not in ('.py',)))
if DATA:
    image = image.add_local_dir(DATA, '/data', ignore=lambda p: p.suffix not in ('.npz', '.json'))


def _panel(net):
    return f'/vol/panels/real_{net}_s10101.pt'


@app.function(image=image, volumes={'/vol': vol}, cpu=4, memory=8192, timeout=3 * 3600)
def build_panel(net):
    sys.path.insert(0, '/code')
    os.makedirs('/vol/panels', exist_ok=True)
    if not os.path.exists(_panel(net)):
        subprocess.run([sys.executable, 'eval_real.py', 'build', '--dataset', net, '--seed', '10101', '--v6', '/data', '--out', _panel(net)],
                       cwd='/code', check=True)
        vol.commit()
    import panel_digest
    return net, panel_digest.digest(_panel(net))


@app.function(image=image, volumes={'/vol': vol}, gpu='A100-40GB', cpu=4, memory=16384, timeout=8 * 3600)
def tabpfn(net):
    vol.reload()
    out = f'/vol/runs_real/{net}/tabpfn_v2'
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, 'tabpfn_colab.py', '--panel', _panel(net), '--out', out, '--allow-constant-support', '--crps',
                    '--device', 'cuda'], cwd='/code', check=True)
    vol.commit()
    return net, sorted(os.listdir(out))


@app.local_entrypoint()
def main(nets: str):
    nets = nets.split(',')
    for net, d in build_panel.map(nets):
        print('PANEL', net, d, flush=True)
    for net, files in tabpfn.map(nets):
        print('TABPFN', net, files, flush=True)
