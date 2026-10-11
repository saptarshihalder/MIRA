"""Start the remaining Colab GPU work as detached background chains (logs on Drive); returns immediately.

    python colab_launch.py [v9|v7v8]   # idempotent: a chain whose marker says it is running is not started twice
v9 (default): chain A9 = colab_v9.py A (v9 networks), chain B9 = colab_v9.py B (development, selection, variant).
v7v8: chain A = v7 main then v8 baselines; chain B = v7 UPT-L. Every stage skips finished items, so relaunching after a
disconnect resumes. Status: tail /content/drive/MyDrive/mira_chain_<name>.log
"""
import subprocess, sys
from pathlib import Path

D = '/content/drive/MyDrive'
V4, V6, V7, V8, V9 = f'{D}/mira_v4', f'{D}/mira_v6', f'{D}/mira_v7', f'{D}/mira_v8', f'{D}/mira_v9'
PY = sys.executable
V9_ARGS = ['--root', V9, '--v4', V4, '--v6', V6, '--v7', V7, '--v8', V8]
SETS = {
    'v7v8': {'A': [[PY, 'colab_v7.py', '--root', V7, '--v6', V6, 'all'],
                   [PY, 'colab_v8.py', '--root', V8, '--v6', V6, '--v7', V7, 'all']],
             'B': [[PY, 'colab_v7.py', '--root', V7, '--v6', V6, 'sourceL', 'finetuneL', 'scoreL', 'confirm', 'pack']]},
    'v9': {'A9': [[PY, 'colab_v9.py', *V9_ARGS, 'A']],          # v9 networks; waits for B9's selection before scoring
           'B9': [[PY, 'colab_v9.py', *V9_ARGS, 'B']]},         # development fine-tunes, selection, selected variant
}
CHAINS = SETS[sys.argv[1] if len(sys.argv) > 1 else 'v9']

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    for name, cmds in CHAINS.items():
        script = ' && '.join(' '.join(f"'{c}'" for c in cmd) for cmd in cmds)
        log = f'{D}/mira_chain_{name}.log'
        busy = subprocess.run(['pgrep', '-f', f'mira_chain_{name}_marker'], capture_output=True).returncode == 0
        if busy:
            print(name, 'already running'); continue
        subprocess.Popen(['bash', '-c', f': mira_chain_{name}_marker; cd {here} && ({script}) >> {log} 2>&1; echo CHAIN_{name}_EXIT $? >> {log}'],
                         start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True)                       # no terminal handles, so the notebook cell returns at once
        print(name, 'started ->', log)
