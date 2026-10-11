"""Start the remaining Colab GPU work as detached background chains (logs on Drive); returns immediately.

    python colab_launch.py          # idempotent: a chain whose marker says it is running is not started twice
Chain A: v7 main (data .. confirm, pack), then v8 baselines (all). Chain B: v7 UPT-L (sourceL, finetuneL, scoreL), then
v7 confirm again (now with S1/S2). Every stage skips finished items, so relaunching after a disconnect resumes.
Status: tail /content/drive/MyDrive/mira_chain_{A,B}.log
"""
import subprocess, sys
from pathlib import Path

D = '/content/drive/MyDrive'
V6, V7, V8 = f'{D}/mira_v6', f'{D}/mira_v7', f'{D}/mira_v8'
PY = sys.executable
CHAINS = {
    'A': [[PY, 'colab_v7.py', '--root', V7, '--v6', V6, 'all'],
          [PY, 'colab_v8.py', '--root', V8, '--v6', V6, '--v7', V7, 'all']],
    'B': [[PY, 'colab_v7.py', '--root', V7, '--v6', V6, 'sourceL', 'finetuneL', 'scoreL', 'confirm', 'pack']],
}

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    for name, cmds in CHAINS.items():
        script = ' && '.join(' '.join(f"'{c}'" for c in cmd) for cmd in cmds)
        log = f'{D}/mira_chain_{name}.log'
        busy = subprocess.run(['pgrep', '-f', f'mira_chain_{name}_marker'], capture_output=True).returncode == 0
        if busy:
            print(name, 'already running'); continue
        subprocess.Popen(['bash', '-c', f': mira_chain_{name}_marker; cd {here} && ({script}) >> {log} 2>&1; echo CHAIN_{name}_EXIT $? >> {log}'],
                         start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(name, 'started ->', log)
