"""Generate every number, table and figure of the paper from saved per-task scores (no model is run here).

python make_paper.py --out /home/claude/mira/paper/lifted_cavity
Writes generated/numbers.tex (macros), generated/*_table.tex, figures/*.pdf. Missing inputs print as '--'.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--cache', default='cache'); ap.add_argument('--real', default='runs_real')
ap.add_argument('--out', required=True)
a = ap.parse_args()
R, C, RR, OUT = Path(a.runs), Path(a.cache), Path(a.real), Path(a.out)
(OUT / 'generated').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
PANEL = dict(F1='v2_s20261201_n256_p5_nl0.4_sr48', F2='v2_s20261202_n128_p8_nl0.4_sr48', F3='v2_s20261203_n128_p16_nl0.4_sr48',
             F4='v2_s20261204_n128_p5_nl0.8_sr48', F5='v2_s20261205_n128_p5_nl0.0_sr48', F6='v2_s20261206_n128_p5_nl0.4_sr16',
             F7='v2_s20261207_n128_p5_nl0.4_sr96')
PDESC = dict(F2='8 sensors', F3='16 sensors', F4='nonlinearity 0.8', F5='linear sensors', F6='16 support rows', F7='96 support rows')
PHEAD = dict(F2='$P{=}8$', F3='$P{=}16$', F4='nonlin.\\ 0.8', F5='linear', F6='$n{=}16$', F7='$n{=}96$')
SEEDS = dict(lift1=(1, 2, 3), lift2=(1,), lift1_static=(1, 2, 3), lift0=(1, 2, 3), anchor_mlp=(1, 2, 3), repo_cavity_fresh=(1,), pfn=(1,), pfnall=(1,))
macros = {}
_panels = {}


def panel(tag):
    if tag not in _panels:
        f = C / f'{tag}.pt'
        _panels[tag] = torch.load(f, weights_only=False) if f.exists() else None
    return _panels[tag]


def cells(name, tag, k, metric='nll'):
    """Per-task values (tasks,), averaged over masks (and seeds for learned models); None if unavailable."""
    p = panel(tag)
    if p is None:
        return None
    if name in p['refs'][k]:
        v = p['refs'][k][name][metric]
        return None if np.isnan(v).all() else v.mean(1)
    if name == 'bop':
        f = R / 'bop' / f'cells_{tag}.npz'
        return np.load(f)[f'k{k}'].mean(1) if (f.exists() and metric == 'nll') else None
    vals = []
    for s in SEEDS.get(name, (1,)):
        f = R / f'{name}_s{s}' / f'cells_{tag}.npz'
        if f.exists():
            z = np.load(f)
            if f'k{k}_{metric}' in z:
                vals.append(z[f'k{k}_{metric}'].mean(1))
    return np.mean(vals, 0) if vals else None


def ci(base, new):
    d = base - new; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def f3(x, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    s = f'{x:+.3f}' if plus else f'{x:.3f}'
    return s.replace('-', '$-$') if not plus else s.replace('-', '$-$').replace('+', '$+$')


DIG = dict(zip('0123456789', ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']))


def mac(name, val):
    """LaTeX macro names cannot contain digits or underscores: digits become words.
    Signed values also get an unsigned '...abs' variant for prose."""
    key = ''.join(DIG.get(ch, ch) for ch in name if ch.isalnum())
    macros[key] = val
    if isinstance(val, str) and (val.startswith('$+$') or val.startswith('$-$')) and not key.endswith('ci'):
        macros[key + 'abs'] = val[3:]


# ------------------------------------------------------------------ Table 1: main synthetic panel F1
ROWS1 = [('Oracle (true parameters)', 'oracle', 'priv'), ('Bayes-optimal (HMC, true prior)', 'bop', 'priv'),
         ('FA-Gaussian', 'fa1', 'cf'), ('EM-Gaussian', 'em_gauss', 'cf'), ('NL-FA (cubic, tuned)', 'nlfa', 'cf'),
         ('Bayesian linear regression', 'blr_cc', 'cf'), ('Ridge (complete case)', 'ridge_cc', 'cf'), ('Ridge (mean-imputed)', 'ridge_repo', 'cf'),
         ('\\textbf{Lifted cavity (ours)}, 8k', 'lift1', 'ic'), ('Transformer (TabPFN-v2 style), 203k', 'pfn', 'ic'),
         ('Residual MLP on FA anchor, 15k', 'anchor_mlp', 'ic'), ('Scalar-cavity network, 3k', 'repo_cavity_fresh', 'ic'),
         ('\\emph{Transformer, trained on all masks}', 'pfnall', 'ic*')]
T = PANEL['F1']
vals = {name: [cells(name, T, k) for k in range(4)] for _, name, _ in ROWS1}
fa2, bop2 = vals['fa1'][2], vals['bop'][2]
best = {}
for k in range(4):
    cand = [(round(float(np.mean(v[k])), 3), n) for _, n, g in ROWS1 if g in ('cf', 'ic') for v in [vals[n]] if v[k] is not None]
    best[k] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
lines = ['\\begin{table}[t]', '\\centering\\small', '\\caption{\\label{tab:main}Main synthetic panel (F1: 256 fresh tasks, five sensors). Mean test NLL by number of missing query sensors; '
         'learned models saw at most one missing sensor in training, so $k=2,3$ are unseen patterns. Learned rows average three training seeds '
         '(seed s.d.\\ $\\leq0.002$) except the transformers (one seed). Last column: share of the achievable gap (FA-Gaussian to Bayes-optimal) closed at $k=2$. '
         'Best non-privileged entry per column in bold.}',
         '\\begin{tabular}{lccccc}', '\\toprule', 'Method & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$ & Gap closed\\\\', '\\midrule']
groups = {'priv': 'Privileged references', 'cf': 'Support-only closed forms', 'ic': 'Learned in context (trained on $k\\leq1$)'}
last = None
for label, name, g in ROWS1:
    gg = g.rstrip('*')
    if gg != last:
        if last is not None:
            lines.append('\\midrule')
        lines.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{groups[gg]}}}}}\\\\')
        last = gg
    row = []
    for k in range(4):
        v = vals[name][k]
        s = f3(None if v is None else float(np.mean(v)))
        row.append(f'\\textbf{{{s}}}' if name in best[k] else s)
    gap = '--'
    if vals[name][2] is not None and fa2 is not None and bop2 is not None and g != 'priv':
        g_ = 100 * (fa2.mean() - vals[name][2].mean()) / (fa2.mean() - bop2.mean())
        gap = (f'{g_:.0f}\\%').replace('-', '$-$')
    lines.append(f'{label} & ' + ' & '.join(row) + f' & {gap}\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'main_table.tex').write_text('\n'.join(lines) + '\n')
for _, name, _ in ROWS1:
    for k in range(4):
        v = vals[name][k]
        if v is not None:
            mac(f'F{name.replace("_", "")}k{"abcd"[k]}', f3(float(v.mean())))
if fa2 is not None and bop2 is not None:
    for name in ('lift1', 'pfn', 'pfnall', 'anchor_mlp', 'nlfa'):
        v = vals[name][2]
        if v is not None:
            mac(f'Gap{name.replace("_", "")}', f'{100 * (fa2.mean() - v.mean()) / (fa2.mean() - bop2.mean()):.0f}')

# ------------------------------------------------------------------ Table 2: generalisation panels, k=2
ROWS2 = [('Oracle', 'oracle'), ('FA-Gaussian', 'fa1'), ('EM-Gaussian', 'em_gauss'), ('NL-FA', 'nlfa'), ('BLR', 'blr_cc'),
         ('Lifted cavity (ours)', 'lift1'), ('Transformer', 'pfn'), ('Residual MLP', 'anchor_mlp'), ('\\emph{Transformer, all masks}', 'pfnall')]
cols = ['F2', 'F3', 'F4', 'F5', 'F6', 'F7']
lines = ['\\begin{table}[t]', '\\centering\\small', '\\setlength{\\tabcolsep}{5pt}', '\\caption{\\label{tab:shift}Generalization beyond the training distribution (fresh panels of 128 tasks). '
         'Mean test NLL with two missing query sensors. Every learned model was trained on $P{=}5$ sensors, nonlinearity 0.4 and $n{=}48$ support rows; '
         'the residual MLP is defined only for five sensors. Best non-privileged entry per column in bold.}',
         '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
         'Method & ' + ' & '.join(PHEAD[c] for c in cols) + '\\\\', '\\midrule']
tab = {n: [cells(n, PANEL[c], 2) for c in cols] for _, n in ROWS2}
bestc = {}
for i, c in enumerate(cols):
    cand = [(round(float(tab[n][i].mean()), 3), n) for _, n in ROWS2 if n not in ('oracle', 'pfnall') and tab[n][i] is not None]
    bestc[i] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
for label, n in ROWS2:
    row = []
    for i in range(len(cols)):
        v = tab[n][i]
        s = f3(None if v is None else float(v.mean()))
        row.append(f'\\textbf{{{s}}}' if n in bestc[i] else s)
    if n == 'oracle':
        lines.append(f'{label} & ' + ' & '.join(row) + '\\\\\\midrule')
    else:
        lines.append(f'{label} & ' + ' & '.join(row) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'shift_table.tex').write_text('\n'.join(lines) + '\n')
for i, c in enumerate(cols):
    for n in ('lift1', 'pfn', 'fa1', 'pfnall', 'nlfa', 'oracle', 'anchor_mlp'):
        if tab[n][i] is not None:
            mac(f'S{c}{n.replace("_", "")}', f3(float(tab[n][i].mean())))
    if tab['lift1'][i] is not None and tab['fa1'][i] is not None:
        m, lo, hi = ci(tab['fa1'][i], tab['lift1'][i]); mac(f'S{c}gainfa', f3(m, True)); mac(f'S{c}gainfaci', f'[{f3(lo, True)}, {f3(hi, True)}]')
    if tab['lift1'][i] is not None and tab['pfn'][i] is not None:
        m, lo, hi = ci(tab['pfn'][i], tab['lift1'][i]); mac(f'S{c}gainpfn', f3(m, True)); mac(f'S{c}gainpfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')

for c in ('F4', 'F5', 'F6', 'F7', 'F2', 'F3'):
    for k in (0, 2):
        l, p_, f_ = cells('lift1', PANEL[c], k), cells('pfn', PANEL[c], k), cells('fa1', PANEL[c], k)
        if l is not None and p_ is not None:
            m, lo, hi = ci(p_, l); mac(f'X{c}kk{k}liftvspfn', f3(m, True)); mac(f'X{c}kk{k}liftvspfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        if p_ is not None and f_ is not None:
            m, lo, hi = ci(f_, p_); mac(f'X{c}kk{k}pfnvsfa', f3(m, True)); mac(f'X{c}kk{k}pfnvsfaci', f'[{f3(lo, True)}, {f3(hi, True)}]')
for k in range(4):
    l, p_ = cells('lift1', T, k), cells('pfn', T, k)
    if l is not None and p_ is not None:
        m, lo, hi = ci(p_, l); mac(f'MainkK{k}liftvspfn', f3(m, True)); mac(f'MainkK{k}liftvspfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')
    l, p_ = cells('lift1', T, k), cells('pfnall', T, k)
    if l is not None and p_ is not None:
        m, lo, hi = ci(p_, l); mac(f'MainkK{k}liftvspfnall', f3(m, True)); mac(f'MainkK{k}liftvspfnallci', f'[{f3(lo, True)}, {f3(hi, True)}]')
    l, p_ = cells('lift1', T, k), cells('anchor_mlp', T, k)
    if l is not None and p_ is not None:
        m, lo, hi = ci(p_, l); mac(f'MainkK{k}liftvsmlp', f3(m, True)); mac(f'MainkK{k}liftvsmlpci', f'[{f3(lo, True)}, {f3(hi, True)}]')

# ------------------------------------------------------------------ Table 3: ablations on F1
ROWS3 = [('Lifted cavity, $K=1$ (ours)', 'lift1'), ('Lifted cavity, $K=2$', 'lift2'), ('Lifted sites, no cavity input', 'lift1_static'),
         ('Scalar sites ($K=0$), same machinery', 'lift0'), ('Scalar-cavity network (retrained)', 'repo_cavity_fresh'), ('FA-Gaussian (untrained)', 'fa1')]
lines = ['\\begin{table}[t]', '\\centering\\small', '\\caption{\\label{tab:ablation}Ablations on F1 (mean test NLL). Same training recipe for every learned row.}',
         '\\begin{tabular}{lcccc}', '\\toprule', 'Variant & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$\\\\', '\\midrule']
for label, n in ROWS3:
    lines.append(f'{label} & ' + ' & '.join(f3(None if cells(n, T, k) is None else float(cells(n, T, k).mean())) for k in range(4)) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'ablation_table.tex').write_text('\n'.join(lines) + '\n')
for n in ('lift1', 'lift1_static', 'lift0', 'repo_cavity_fresh', 'lift2'):
    v, f = cells(n, T, 2), cells('fa1', T, 2)
    if v is not None:
        m, lo, hi = ci(f, v); mac(f'Abl{n.replace("_", "")}vsfa', f3(m, True))
for n1, n2, key in (('lift1', 'lift1_static', 'Cav'), ('lift1', 'lift0', 'Lift')):
    v1, v2 = cells(n1, T, 2), cells(n2, T, 2)
    if v1 is not None and v2 is not None:
        m, lo, hi = ci(v2, v1); mac(f'Abl{key}', f3(m, True)); mac(f'Abl{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')

# ------------------------------------------------------------------ Table 4: real data
REAL = [('beijing', 'Beijing PM$_{2.5}$ (11 stations)', (0, 3, 6)), ('airq_co', 'Air Quality CO', (0, 2)), ('airq_no2', 'Air Quality NO$_2$', (0, 2))]
ROWS4 = [('EM-Gaussian', 'ref', 'em_gauss'), ('FA-Gaussian', 'ref', 'fa1'), ('NL-FA', 'ref', 'nlfa'), ('Bayesian linear regression', 'ref', 'blr_cc'),
         ('Ridge (complete case)', 'ref', 'ridge_cc'),
         ('Lifted cavity, zero-shot', 'zs', 'lift1_s1'), ('Transformer, zero-shot', 'zs', 'pfn_s1'),
         ('\\textbf{Lifted cavity, fine-tuned (ours)}', 'ft', 'lift1_s{1,2,3}_ft'), ('\\quad no synthetic pretraining', 'ft', 'lift1_untrained_ft'),
         ('\\quad no cavity input', 'ft', 'lift1_static_s1_ft'), ('\\quad scalar sites ($K=0$)', 'ft', 'lift0_s1_ft'),
         ('Transformer, fine-tuned', 'ft', 'pfn_s1_ft'), ('Residual MLP, fine-tuned', 'ft', 'anchor_mlp_s{1,2,3}_ft')]


def real_cells(ds, kind, name, e, metric='nll'):
    tag = f'real_{ds}_s2027'
    pf = C / f'{tag}.pt'
    if not pf.exists():
        return None
    if kind == 'ref':
        p = _panels.setdefault(tag, torch.load(pf, weights_only=False))
        v = p['refs'][e][name][metric]
        return None if np.isnan(v).all() else v
    names = [name.replace('{1,2,3}', str(s)) for s in (1, 2, 3)] if '{1,2,3}' in name else [name]
    vals = []
    for nm in names:
        f = (R / nm if kind == 'zs' else RR / ds / nm) / f'cells_{tag}.npz'
        if f.exists():
            vals.append(np.load(f)[f'e{e}_{metric}'])
    return np.mean(vals, 0) if vals else None


def week_ci(base, new, keys):
    d = base - new
    wk = [k.split('|')[0] for k in keys]
    if len(set(wk)) < len(wk):
        d = np.array([d[[w == u for w in wk]].mean() for u in sorted(set(wk))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


colspec, head1, head2 = 'l', ['Method'], ['']
for ds, lab, es in REAL:
    colspec += 'c' * len(es)
    head1.append(f'\\multicolumn{{{len(es)}}}{{c}}{{{lab}}}')
    head2 += [('natural' if (ds == 'beijing' and e == 0) else (f'$+{e}$' if ds == 'beijing' else f'$k{{=}}{e}$')) for e in es]
lines = ['\\begin{table}[t]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{2.5pt}',
         '\\caption{\\label{tab:real}Real sensor networks: mean test NLL on later, held-out periods. Beijing: 12 target stations $\\times$ weekly episodes in '
         '2016--2017 with natural missingness, plus extra dropped sensors. Air Quality: weekly episodes after October 2004, sensor dropout simulated. '
         'Fine-tuning uses only earlier periods and one recipe for every model; lifted-cavity and residual-MLP rows average three seeds. Best entry per column in bold.}',
         '\\begin{tabular}{' + colspec + '}', '\\toprule', ' & '.join(head1) + '\\\\', ' & '.join(head2) + '\\\\', '\\midrule']
rv = {}
for label, kind, name in ROWS4:
    rv[(kind, name)] = [real_cells(ds, kind, name, e) for ds, _, es in REAL for e in es]
ncol = sum(len(es) for _, _, es in REAL)
bestr = {}
for i in range(ncol):
    cand = [(round(float(rv[(k, n)][i].mean()), 3), (k, n)) for _, k, n in ROWS4 if rv[(k, n)][i] is not None]
    bestr[i] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
for label, kind, name in ROWS4:
    row = []
    for i in range(ncol):
        v = rv[(kind, name)][i]
        s = f3(None if v is None else float(v.mean()))
        row.append(f'\\textbf{{{s}}}' if (kind, name) in bestr[i] else s)
    if label.startswith('Lifted cavity, zero-shot'):
        lines.append('\\midrule')
    if label.startswith('\\textbf{Lifted cavity, fine-tuned'):
        lines.append('\\midrule')
    lines.append(f'{label} & ' + ' & '.join(row) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'real_table.tex').write_text('\n'.join(lines) + '\n')
for ds, _, es in REAL:
    pf = C / f'real_{ds}_s2027.pt'
    if not pf.exists():
        continue
    keys = _panels.setdefault(f'real_{ds}_s2027', torch.load(pf, weights_only=False))['pool']['keys']
    meta = _panels[f'real_{ds}_s2027']['meta']
    mac(f'R{ds.replace("_", "")}episodes', str(meta['episodes']))
    mac(f'R{ds.replace("_", "")}natmiss', f'{100 * meta["natural_query_missing_frac"]:.1f}')
    for e in es:
        l = real_cells(ds, 'ft', 'lift1_s{1,2,3}_ft', e)
        for other, key in (('blr_cc', 'blr'), ('em_gauss', 'em'), ('fa1', 'fa')):
            b = real_cells(ds, 'ref', other, e)
            if l is not None and b is not None:
                m, lo, hi = week_ci(b, l, keys); mac(f'R{ds.replace("_", "")}e{e}vs{key}', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}vs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        for other, key in (('pfn_s1_ft', 'pfn'), ('lift1_untrained_ft', 'nopre'), ('anchor_mlp_s{1,2,3}_ft', 'mlp'), ('lift1_static_s1_ft', 'static')):
            b = real_cells(ds, 'ft', other, e)
            if l is not None and b is not None:
                m, lo, hi = week_ci(b, l, keys); mac(f'R{ds.replace("_", "")}e{e}vs{key}', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}vs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')

# ------------------------------------------------------------------ calibration table (F1 k=2, Beijing natural)
CAL = [('FA-Gaussian', 'fa1'), ('EM-Gaussian', 'em_gauss'), ('BLR', 'blr_cc'), ('Lifted cavity (ours)', 'lift1'), ('Transformer', 'pfn'), ('Residual MLP', 'anchor_mlp')]
lines = ['\\begin{table}[t]', '\\centering\\small', '\\caption{\\label{tab:calib}Calibration and point accuracy. Coverage of the central 90\\% predictive interval (target 0.90), '
         'continuous ranked probability score (CRPS, lower is better) and RMSE. F1 with two missing sensors; Beijing test with natural missingness (fine-tuned learned models).}',
         '\\begin{tabular}{lcccccc}', '\\toprule', ' & \\multicolumn{3}{c}{F1, $k=2$} & \\multicolumn{3}{c}{Beijing, natural}\\\\',
         'Method & Cov.\\ 90 & CRPS & RMSE & Cov.\\ 90 & CRPS & RMSE\\\\', '\\midrule']
for label, n in CAL:
    row = []
    for metric in ('cov', 'crps', 'se'):
        v = cells(n, T, 2, metric)
        row.append(f3(None if v is None else float(np.sqrt(v.mean()) if metric == 'se' else v.mean())))
    bmap = {'fa1': ('ref', 'fa1'), 'em_gauss': ('ref', 'em_gauss'), 'blr_cc': ('ref', 'blr_cc'), 'lift1': ('ft', 'lift1_s{1,2,3}_ft'),
            'pfn': ('ft', 'pfn_s1_ft'), 'anchor_mlp': (None, None)}
    kind, nm = bmap[n]
    for metric in ('cov', 'crps', 'se'):
        v = real_cells('beijing', kind, nm, 0, metric) if kind else None
        row.append(f3(None if v is None else float(np.sqrt(v.mean()) if metric == 'se' else v.mean())))
    lines.append(f'{label} & ' + ' & '.join(row) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'calib_table.tex').write_text('\n'.join(lines) + '\n')

# ------------------------------------------------------------------ endpoints
for fname, key in (('confirm_v2.json', 'V2'), ('confirm_v1.json', 'V1')):
    f = R / fname
    if f.exists():
        res = json.loads(f.read_text())
        for i, (k, v) in enumerate(res.items()):
            if isinstance(v, dict) and 'gain' in v:
                tagk = re.sub(r'[^A-Za-z]', '', k.split('_')[0])
                mac(f'{key}{tagk}gain', f3(v['gain'], True)); mac(f'{key}{tagk}ci', f'[{f3(v["lo"], True)}, {f3(v["hi"], True)}]')
                mac(f'{key}{tagk}pass', 'pass' if v.get('passed') else 'fail')

# ------------------------------------------------------------------ figure: gains over FA-Gaussian by k, and width transfer
INK2 = '#52514e'
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
ks = np.arange(4)
SER = [('lift1', 'Lifted cavity (ours)', '#2a78d6', 'o', '-'), ('pfn', 'Transformer', '#eb6834', 's', '-'),
       ('pfnall', 'Transformer, all masks', '#eb6834', 's', ':'), ('anchor_mlp', 'Residual MLP', '#1baf7a', 'D', '-'),
       ('nlfa', 'NL-FA', '#eda100', 'v', '-')]
ax = axes[0]
fa_k = [vals['fa1'][k].mean() for k in range(4)]
for name, lab, ls in (('oracle', 'Oracle', '--'), ('bop', 'Bayes-optimal (HMC)', '-.')):
    if all(vals[name][k] is not None for k in range(4)):
        ax.plot(ks, [fa_k[k] - vals[name][k].mean() for k in range(4)], color=INK2, lw=1.1, ls=ls, label=lab)
for name, lab, col, mk, ls in SER:
    if all(vals[name][k] is not None for k in range(4)):
        ax.plot(ks, [fa_k[k] - vals[name][k].mean() for k in range(4)], color=col, lw=1.7, ls=ls, marker=mk, ms=3.5, label=lab)
ax.axhline(0, color='#c8c7c2', lw=.8)
ax.axvspan(1.5, 3.2, color='#f0efeb', zorder=0)
ax.set_xticks(ks); ax.set_xlim(-.2, 3.2)
ax.set_xlabel('missing query sensors $k$ (shaded: unseen in training)', fontsize=8); ax.set_ylabel('NLL gain over FA-Gaussian', fontsize=8)
ax.set_title('(a) Main panel F1', fontsize=9, loc='left')
ax = axes[1]
widths = [(5, 'F1'), (8, 'F2'), (16, 'F3')]
for name, lab, col, mk, ls in SER:
    xs, ys = [], []
    for P, c in widths:
        v, f = cells(name, PANEL[c], 2), cells('fa1', PANEL[c], 2)
        if v is not None and f is not None:
            xs.append(P); ys.append(f.mean() - v.mean())
    if xs:
        ax.plot(xs, ys, color=col, lw=1.7, ls=ls, marker=mk, ms=3.5)
ax.axhline(0, color='#c8c7c2', lw=.8)
ax.set_xticks([5, 8, 16]); ax.set_xlabel('number of sensors $P$ (all models trained on 5)', fontsize=8); ax.set_ylabel('NLL gain over FA-Gaussian, $k=2$', fontsize=8)
ax.set_title('(b) Width transfer', fontsize=9, loc='left')
for ax in axes:
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(.5, 1.08))
fig.tight_layout(rect=(0, 0, 1, .88))
fig.savefig(OUT / 'figures' / 'main.pdf', bbox_inches='tight'); fig.savefig(OUT / 'figures' / 'main.png', dpi=180, bbox_inches='tight')

# ------------------------------------------------------------------ appendix: every panel and k
ROWSM = [('Oracle', 'oracle'), ('Bayes-optimal (HMC)', 'bop'), ('FA-Gaussian', 'fa1'), ('EM-Gaussian', 'em_gauss'), ('NL-FA', 'nlfa'), ('BLR', 'blr_cc'),
         ('Ridge (complete case)', 'ridge_cc'), ('Lifted cavity (ours)', 'lift1'), ('Lifted cavity, $K=2$', 'lift2'), ('Lifted sites, no cavity', 'lift1_static'),
         ('Scalar sites ($K=0$)', 'lift0'), ('Transformer', 'pfn'), ('Transformer, all masks', 'pfnall'), ('Residual MLP', 'anchor_mlp'),
         ('Scalar-cavity network', 'repo_cavity_fresh')]
more = []
for c in ('F2', 'F3', 'F4', 'F5', 'F6', 'F7'):
    tg = PANEL[c]
    lines = ['\\begin{table}[h]', '\\centering\\small', f'\\caption{{\\label{{tab:more{c}}}Panel {c} ({PDESC[c]}; 128 tasks): mean test NLL by number of missing sensors.}}',
             '\\begin{tabular}{lcccc}', '\\toprule', 'Method & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$\\\\', '\\midrule']
    for label, n in ROWSM:
        vv = [cells(n, tg, k) for k in range(4)]
        if all(v is None for v in vv):
            continue
        lines.append(f'{label} & ' + ' & '.join(f3(None if v is None else float(v.mean())) for v in vv) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    more.append('\n'.join(lines))
for ds, lab in (('airq_co', 'Air Quality CO'), ('airq_no2', 'Air Quality NO$_2$')):
    lines = ['\\begin{table}[h]', '\\centering\\small', f'\\caption{{\\label{{tab:more{ds.replace("_", "")}}}{lab}: mean test NLL for every number of simulated missing sensors (23 later weeks).}}',
             '\\begin{tabular}{lcccc}', '\\toprule', 'Method & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$\\\\', '\\midrule']
    for label, kind, name in ROWS4:
        vv = [real_cells(ds, kind, name, e) for e in range(4)]
        if all(v is None for v in vv):
            continue
        lines.append(f'{label} & ' + ' & '.join(f3(None if v is None else float(v.mean())) for v in vv) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    more.append('\n'.join(lines))
(OUT / 'generated' / 'more.tex').write_text('\n\n'.join(more) + '\n')

# ------------------------------------------------------------------ v1 table and endpoint tables
v1f = Path('conf_results.json')
if v1f.exists():
    v1 = json.loads(v1f.read_text())
    desc = dict(E1='Fresh tasks, vs FA-Gaussian', E2='Fresh tasks, vs residual MLP', E3='8 sensors, vs FA-Gaussian',
                E4='Nonlinearity 0.8, vs residual MLP', R1='Air Quality NO$_2$, 23 later weeks, vs EM-Gaussian')
    lines = ['\\begin{table}[h]', '\\centering\\small', '\\caption{\\label{tab:v1}Protocol v1 endpoints (two missing sensors; positive gain favours the lifted cavity network; pass requires gain $\\geq0.01$ and a positive lower bound).}',
             '\\begin{tabular}{llcc}', '\\toprule', 'Endpoint & Comparison & Gain [95\\% CI] & Result\\\\', '\\midrule']
    for k in ('E1', 'E2', 'E3', 'E4', 'R1'):
        v = v1[k]
        lines.append(f"{k} & {desc[k]} & {f3(v['gain'], True)} [{f3(v['lo'], True)}, {f3(v['hi'], True)}] & {'pass' if v['passed'] else 'fail'}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'v1_table.tex').write_text('\n'.join(lines) + '\n')
v2f = R / 'confirm_v2.json'
if v2f.exists():
    v2 = json.loads(v2f.read_text())
    desc = {'E1_F1_k2_lift1_vs_nlfa': ('E1', 'F1, $k=2$: vs NL-FA'), 'E2_F1_k2_lift1_vs_pfn': ('E2', 'F1, $k=2$: vs transformer'),
            'E3a_F3_k2_lift1_vs_fa1': ('E3a', '16 sensors, $k=2$: vs FA-Gaussian'), 'E3b_F3_k2_lift1_vs_pfn': ('E3b', '16 sensors, $k=2$: vs transformer'),
            'E4a_beijing_nat_lift1ft_vs_blr': ('E4a', 'Beijing, natural: vs BLR'), 'E4b_beijing_nat_lift1ft_vs_pfnft': ('E4b', 'Beijing, natural: vs fine-tuned transformer')}
    lines = ['\\begin{table}[t]', '\\centering\\small', '\\caption{\\label{tab:v2}Protocol v2 endpoints, committed before any v2 panel existed. Gains in nats (positive favours the lifted cavity network); '
             'Beijing intervals are cluster-robust over weeks. Pass requires gain $\\geq0.01$ and a positive lower bound.}',
             '\\begin{tabular}{llcc}', '\\toprule', 'Endpoint & Comparison & Gain [95\\% CI] & Result\\\\', '\\midrule']
    for k, (tagk, d) in desc.items():
        if k in v2:
            v = v2[k]
            lines.append(f"{tagk} & {d} & {f3(v['gain'], True)} [{f3(v['lo'], True)}, {f3(v['hi'], True)}] & {'pass' if v['passed'] else 'fail'}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'v2_table.tex').write_text('\n'.join(lines) + '\n')

(OUT / 'generated' / 'numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
print(len(macros), 'macros;', ', '.join(sorted(macros)[:12]), '...')
