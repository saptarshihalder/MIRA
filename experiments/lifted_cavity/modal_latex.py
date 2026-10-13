"""Compile paper/upt on Modal (CPU, TeX Live) and bring back the PDF and the log.

    python -m modal run modal_latex.py --out <local folder>
"""
from pathlib import Path
import modal

PAPER = Path(__file__).resolve().parents[2] / 'paper' / 'upt'
app = modal.App('mira-latex')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('texlive-latex-base', 'texlive-latex-recommended', 'texlive-latex-extra', 'texlive-fonts-recommended',
                      'texlive-science', 'texlive-pictures', 'texlive-bibtex-extra', 'latexmk', 'lmodern')
         .add_local_dir(str(PAPER), '/paper'))


@app.function(image=image, cpu=2, memory=4096, timeout=900)
def build():
    import shutil, subprocess
    shutil.copytree('/paper', '/tmp/p', dirs_exist_ok=True)
    r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd='/tmp/p',
                       capture_output=True, text=True)
    log = Path('/tmp/p/main.log').read_text(errors='replace') if Path('/tmp/p/main.log').exists() else ''
    pdf = Path('/tmp/p/main.pdf').read_bytes() if Path('/tmp/p/main.pdf').exists() else b''
    return r.returncode, r.stdout[-3000:] + r.stderr[-2000:], log, pdf


@app.local_entrypoint()
def main(out: str = '.'):
    code, tail, log, pdf = build.remote()
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    (o / 'main.log').write_text(log, encoding='utf-8')
    if pdf:
        (o / 'main.pdf').write_bytes(pdf)
    print('latexmk exit', code, 'pdf bytes', len(pdf))
    print(tail[-2500:])
