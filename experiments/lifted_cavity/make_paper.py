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
ap.add_argument('--out', required=True); ap.add_argument('--v1', default='conf_results.json'); ap.add_argument('--posthoc', default='runs_posthoc'); ap.add_argument('--panels', default=''); ap.add_argument('--explore', default='runs_explore'); ap.add_argument('--init', default='runs_init')
a = ap.parse_args()
R, C, RR, OUT, PH = Path(a.runs), Path(a.cache), Path(a.real), Path(a.out), Path(a.posthoc)
(OUT / 'generated').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
PANEL = dict(F1='v2_s20261201_n256_p5_nl0.4_sr48', F2='v2_s20261202_n128_p8_nl0.4_sr48', F3='v2_s20261203_n128_p16_nl0.4_sr48',
             F4='v2_s20261204_n128_p5_nl0.8_sr48', F5='v2_s20261205_n128_p5_nl0.0_sr48', F6='v2_s20261206_n128_p5_nl0.4_sr16',
             F7='v2_s20261207_n128_p5_nl0.4_sr96')
PDESC = dict(F2='8 sensors', F3='16 sensors', F4='nonlinearity 0.8', F5='linear sensors', F6='16 support rows', F7='96 support rows')
PHEAD = dict(F2='$P{=}8$', F3='$P{=}16$', F4='nonlin.\\ 0.8', F5='linear', F6='$n{=}16$', F7='$n{=}96$')
SEEDS = dict(lift1=(1, 2, 3), lift2=(1,), lift1_static=(1, 2, 3), lift0=(1, 2, 3), anchor_mlp=(1, 2, 3), repo_cavity_fresh=(1,), pfn=(1,), pfnall=(1,), lct=(1,))
macros = {}
_panels = {}


def load_exported(tag):
    d = Path(a.panels)
    if not a.panels or not (d / f'{tag}_refs.npz').exists():
        return None
    z = np.load(d / f'{tag}_refs.npz'); info = json.loads((d / f'{tag}.json').read_text())
    refs = {}
    for key in z.files:
        c, name, m = key.split('|')
        refs.setdefault(int(c), {}).setdefault(name, {})[m] = z[key]
    return dict(refs=refs, pool=dict(n=info['n'], keys=info['keys']), meta=info['meta'])


def panel(tag):
    if tag not in _panels:
        f = C / f'{tag}.pt'
        _panels[tag] = torch.load(f, weights_only=False) if f.exists() else load_exported(tag)
    return _panels[tag]


def cells(name, tag, k, metric='nll'):
    """Per-task values (tasks,), averaged over masks (and seeds for learned models); None if unavailable."""
    p = panel(tag)
    if p is None:
        return None
    if name in p['refs'][k]:
        v = p['refs'][k][name][metric]
        return None if np.isnan(v).all() else v.mean(1)
    if name == 'tabpfn_v2':
        f = R / 'tabpfn_v2' / f'cells_{tag}.npz'
        if not f.exists():
            return None
        z = np.load(f)
        return z[f'k{k}_{metric}'].mean(1) if f'k{k}_{metric}' in z else None
    if name == 'gp':
        f = PH / 'gp' / f'cells_{tag}.npz'
        if not f.exists():
            return None
        z = np.load(f)
        return z[f'k{k}_{metric}'].mean(1) if f'k{k}_{metric}' in z else None
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
         ('GP, complete case$^\\dagger$', 'gp', 'cf'),
         ('\\textbf{Lifted cavity (ours)}, 8k', 'lift1', 'ic'), ('Transformer (TabPFN-v2 style), 203k', 'pfn', 'ic'),
         ('Residual MLP on FA anchor, 15k', 'anchor_mlp', 'ic'), ('Scalar-cavity network, 3k', 'repo_cavity_fresh', 'ic'),
         ('\\emph{Transformer, trained on all masks}', 'pfnall', 'ic*'), ('TabPFN v2, pretrained (in context only)', 'tabpfn_v2', 'ic*'),
         ('\\emph{Lifted cavity transformer (v3)}$^\\ddagger$, 203k', 'lct', 'ic*')]
T = PANEL['F1']
vals = {name: [cells(name, T, k) for k in range(4)] for _, name, _ in ROWS1}
ROWS1 = [r for r in ROWS1 if r[1] not in ('tabpfn_v2', 'lct', 'gp') or any(v is not None for v in vals[r[1]])]
fa2, bop2 = vals['fa1'][2], vals['bop'][2]
best = {}
for k in range(4):
    cand = [(round(float(np.mean(v[k])), 3), n) for _, n, g in ROWS1 if g in ('cf', 'ic') for v in [vals[n]] if v[k] is not None]
    best[k] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
lines = ['\\begin{table}[t]', '\\centering\\small', '\\caption{\\label{tab:main}Main synthetic panel (F1: 256 fresh tasks, five sensors). Mean test NLL by number of missing query sensors; '
         'learned models saw at most one missing sensor in training, so $k=2,3$ are unseen patterns. Learned rows average three training seeds '
         '(seed s.d.\\ $\\leq0.002$) except the transformers (seed 1 as pre-registered; Table~\\ref{tab:seeds} adds a second seed). Last column: share of the achievable gap (FA-Gaussian to Bayes-optimal) closed at $k=2$. '
         'Best non-privileged entry per column in bold. $^\\dagger$Post hoc, scored at $k=2$ only. '
         '$^\\ddagger$Designed after protocol v2 (Section~\\ref{sec:exp-v3}); descriptive on this panel and not bolded.}',
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
    for name in ('lift1', 'pfn', 'pfnall', 'anchor_mlp', 'nlfa', 'lct'):
        v = vals[name][2]
        if v is not None:
            mac(f'Gap{name.replace("_", "")}', f'{100 * (fa2.mean() - v.mean()) / (fa2.mean() - bop2.mean()):.0f}')

# ------------------------------------------------------------------ Table 2: generalisation panels, k=2
ROWS2 = [('Oracle', 'oracle'), ('FA-Gaussian', 'fa1'), ('EM-Gaussian', 'em_gauss'), ('NL-FA', 'nlfa'), ('BLR', 'blr_cc'),
         ('Lifted cavity (ours)', 'lift1'), ('Transformer', 'pfn'), ('Residual MLP', 'anchor_mlp'), ('\\emph{Transformer, all masks}', 'pfnall'),
         ('\\emph{TabPFN v2, pretrained}', 'tabpfn_v2'), ('\\emph{Lifted cavity transformer (v3)}', 'lct')]
cols = ['F2', 'F3', 'F4', 'F5', 'F6', 'F7']
lines = ['\\begin{table}[t]', '\\centering\\small', '\\setlength{\\tabcolsep}{5pt}', '\\caption{\\label{tab:shift}Generalization beyond the training distribution (fresh panels of 128 tasks). '
         'Mean test NLL with two missing query sensors. Every learned model was trained on $P{=}5$ sensors, nonlinearity 0.4 and $n{=}48$ support rows; '
         'the residual MLP is defined only for five sensors. Best non-privileged entry per column in bold; italic rows are '
         'controls or descriptive (the lifted cavity transformer was designed after protocol v2) and are not bolded.}',
         '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
         'Method & ' + ' & '.join(PHEAD[c] for c in cols) + '\\\\', '\\midrule']
tab = {n: [cells(n, PANEL[c], 2) for c in cols] for _, n in ROWS2}
ROWS2 = [r for r in ROWS2 if r[1] not in ('tabpfn_v2', 'lct') or any(v is not None for v in tab[r[1]])]
bestc = {}
for i, c in enumerate(cols):
    cand = [(round(float(tab[n][i].mean()), 3), n) for _, n in ROWS2 if n not in ('oracle', 'pfnall', 'tabpfn_v2', 'lct') and tab[n][i] is not None]
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
         ('Ridge (complete case)', 'ref', 'ridge_cc'), ('GP, complete case$^\\dagger$', 'ph', 'gp'), ('$k$-nearest neighbours$^\\dagger$', 'ph', 'knn'),
         ('Lifted cavity, zero-shot', 'zs', 'lift1_s1'), ('Transformer, zero-shot', 'zs', 'pfn_s1'), ('TabPFN v2, pretrained', 'zs', 'tabpfn_v2'),
         ('\\textbf{Lifted cavity, fine-tuned (ours)}', 'ft', 'lift1_s{1,2,3}_ft'), ('\\quad no synthetic pretraining', 'ft', 'lift1_untrained_ft'),
         ('\\quad no cavity input', 'ft', 'lift1_static_s1_ft'), ('\\quad scalar sites ($K=0$)', 'ft', 'lift0_s1_ft'),
         ('Transformer, fine-tuned', 'ft', 'pfn_s1_ft'), ('Residual MLP, fine-tuned', 'ft', 'anchor_mlp_s{1,2,3}_ft'),
         ('\\emph{Lifted cavity transformer, fine-tuned}$^\\ddagger$', 'ft', 'lct_s1_ft')]


def real_cells(ds, kind, name, e, metric='nll', seed=2027):
    tag = f'real_{ds}_s{seed}'
    if panel(tag) is None:
        return None
    if kind == 'ref':
        p = panel(tag)
        v = p['refs'][e][name][metric]
        return None if np.isnan(v).all() else v
    if kind == 'ph':
        f = PH / name / f'cells_{tag}.npz'
        return np.load(f)[f'e{e}_{metric}'] if (f.exists() and f'e{e}_{metric}' in np.load(f)) else None
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
         'Fine-tuning uses only earlier periods and one recipe for every model; lifted-cavity and residual-MLP rows average three seeds. Best entry per column in bold. '
         '$^\\dagger$Post hoc. $^\\ddagger$Designed after protocol v2 (Section~\\ref{sec:exp-v3}); post hoc here and not bolded.}',
         '\\begin{tabular}{' + colspec + '}', '\\toprule', ' & '.join(head1) + '\\\\', ' & '.join(head2) + '\\\\', '\\midrule']
rv = {}
for label, kind, name in ROWS4:
    rv[(kind, name)] = [real_cells(ds, kind, name, e) for ds, _, es in REAL for e in es]
ROWS4 = [r for r in ROWS4 if r[2] not in ('tabpfn_v2', 'lct_s1_ft') or any(v is not None for v in rv[(r[1], r[2])])]
ncol = sum(len(es) for _, _, es in REAL)
bestr = {}
for i in range(ncol):
    cand = [(round(float(rv[(k, n)][i].mean()), 3), (k, n)) for _, k, n in ROWS4 if rv[(k, n)][i] is not None and n != 'lct_s1_ft']
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
    if panel(f'real_{ds}_s2027') is None:
        continue
    keys = panel(f'real_{ds}_s2027')['pool']['keys']
    meta = panel(f'real_{ds}_s2027')['meta']
    mac(f'R{ds.replace("_", "")}episodes', str(meta['episodes']))
    mac(f'R{ds.replace("_", "")}natmiss', f'{100 * meta["natural_query_missing_frac"]:.1f}')
    for e in es:
        l = real_cells(ds, 'ft', 'lift1_s{1,2,3}_ft', e)
        for other, key in (('blr_cc', 'blr'), ('em_gauss', 'em'), ('fa1', 'fa')):
            b = real_cells(ds, 'ref', other, e)
            if l is not None and b is not None:
                m, lo, hi = week_ci(b, l, keys); mac(f'R{ds.replace("_", "")}e{e}vs{key}', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}vs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        for kind_, nm_, key in (('ref', 'fa1', 'fa'), ('ref', 'blr_cc', 'blr'), ('ref', 'em_gauss', 'em'), ('ref', 'nlfa', 'nlfa'), ('ph', 'gp', 'gp'), ('ph', 'knn', 'knn'),
                                ('zs', 'lift1_s1', 'zslift'), ('zs', 'pfn_s1', 'zspfn'), ('ft', 'lift1_s{1,2,3}_ft', 'ftlift'), ('ft', 'pfn_s1_ft', 'ftpfn')):
            v_ = real_cells(ds, kind_, nm_, e)
            if v_ is not None:
                mac(f'R{ds.replace("_", "")}e{e}val{key}', f3(float(v_.mean())))
        lz, pz, pf_, bl_ = (real_cells(ds, 'zs', 'lift1_s1', e), real_cells(ds, 'zs', 'pfn_s1', e), real_cells(ds, 'ft', 'pfn_s1_ft', e),
                            real_cells(ds, 'ref', 'blr_cc', e))
        if lz is not None and pz is not None:
            m, lo, hi = week_ci(pz, lz, keys); mac(f'R{ds.replace("_", "")}e{e}zsliftvspfn', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}zsliftvspfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        if pf_ is not None and bl_ is not None:
            m, lo, hi = week_ci(bl_, pf_, keys); mac(f'R{ds.replace("_", "")}e{e}pfnvsblr', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}pfnvsblrci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        for other, key in (('pfn_s1_ft', 'pfn'), ('lift1_untrained_ft', 'nopre'), ('anchor_mlp_s{1,2,3}_ft', 'mlp'), ('lift1_static_s1_ft', 'static')):
            b = real_cells(ds, 'ft', other, e)
            if l is not None and b is not None:
                m, lo, hi = week_ci(b, l, keys); mac(f'R{ds.replace("_", "")}e{e}vs{key}', f3(m, True)); mac(f'R{ds.replace("_", "")}e{e}vs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')

# ------------------------------------------------------------------ fine-tuning diagnostics (Beijing PM2.5)
def ft_ema(ds, nm):
    f = RR / ds / nm / 'train.json'
    return json.loads(f.read_text()).get('finetune', {}).get('final_ema') if f.exists() else None


el, ep = [ft_ema('beijing', f'lift1_s{s}_ft') for s in (1, 2, 3)], ft_ema('beijing', 'pfn_s1_ft')
if all(v is not None for v in el) and ep is not None:
    mac('FTbeijingtraingap', f'{np.mean(el) - ep:.2f}')
zl, fl = real_cells('beijing', 'zs', 'lift1_s1', 0), real_cells('beijing', 'ft', 'lift1_s1_ft', 0)
zp, fp = real_cells('beijing', 'zs', 'pfn_s1', 0), real_cells('beijing', 'ft', 'pfn_s1_ft', 0)
if all(v is not None for v in (zl, fl, zp, fp)):
    mac('FTbeijingliftgain', f'{zl.mean() - fl.mean():.2f}'); mac('FTbeijingpfngain', f'{zp.mean() - fp.mean():.2f}')
la, pa, ba = real_cells('beijing', 'ft', 'lift1_s{1,2,3}_ft', 0), real_cells('beijing', 'ft', 'pfn_s1_ft', 0), real_cells('beijing', 'ref', 'blr_cc', 0)
if all(v is not None for v in (la, pa, ba)):
    d_ = la - pa
    mac('FTbeijingpfnbetterpct', f'{100 * (d_ > 0).mean():.0f}')
    qb = np.digitize(ba, np.quantile(ba, [.25, .5, .75]))
    for i_, nm_ in enumerate(('One', 'Two', 'Three', 'Four')):
        mac(f'FTbeijinggapqQ{nm_}', f'{d_[qb == i_].mean():.2f}')

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
    for j, part in enumerate(('F', 'B')):
        for i_, metric in enumerate(('cov', 'crps', 'rmse')):
            if row[3 * j + i_] != '--':
                mac(f'Cal{n.replace("_", "")}{part}{metric}', row[3 * j + i_])
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'calib_table.tex').write_text('\n'.join(lines) + '\n')

# ------------------------------------------------------------------ endpoints
for fname, key in (('confirm_v2.json', 'V2'), ('confirm_v3.json', 'V3')):
    f = R / fname
    if f.exists():
        res = json.loads(f.read_text())
        for i, (k, v) in enumerate(res.items()):
            if isinstance(v, dict) and 'gain' in v:
                tagk = k.split('_')[0]
                mac(f'{key}{tagk}gain', f3(v['gain'], True)); mac(f'{key}{tagk}ci', f'[{f3(v["lo"], True)}, {f3(v["hi"], True)}]')
                mac(f'{key}{tagk}pass', 'pass' if v.get('passed') else 'fail')

# ------------------------------------------------------------------ figure: gains over FA-Gaussian by k, and width transfer
INK2 = '#52514e'
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
ks = np.arange(4)
SER = [('lift1', 'Lifted cavity (ours)', '#2a78d6', 'o', '-'), ('pfn', 'Transformer', '#eb6834', 's', '-'),
       ('pfnall', 'Transformer, all masks', '#eb6834', 's', ':'), ('anchor_mlp', 'Residual MLP', '#1baf7a', 'D', '-'),
       ('nlfa', 'NL-FA', '#eda100', 'v', '-'), ('lct', 'Lifted cavity transformer (v3)', '#2a78d6', 'o', '--')]
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
         ('Scalar-cavity network', 'repo_cavity_fresh'), ('Lifted cavity transformer (v3)', 'lct')]
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
v1f = Path(a.v1)
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


# ------------------------------------------------------------------ protocol v3: lifted sites as the transformer's output layer
V3P = dict(G1='v2_s20261301_n256_p5_nl0.4_sr48', G3='v2_s20261303_n128_p16_nl0.4_sr48')
V3SYN = [('G1', k) for k in range(4)] + [('G3', 2)]
V3REAL = [('beijing_no2', 0), ('beijing_no2', 6), ('beijing_co', 0), ('beijing_co', 6)]
ROWS5 = [('FA-Gaussian', 'fa1', ('ref', 'fa1')), ('Bayesian linear regression', 'blr_cc', ('ref', 'blr_cc')),
         ('Lifted cavity network, 8k', 'lift1', ('ft', 'lift1_s{1,2,3}_ft')), ('Transformer, 203k', 'pfn', ('ft', 'pfn_s1_ft')),
         ('\\textbf{Lifted cavity transformer}, 203k', 'lct', ('ft', 'lct_s1_ft'))]
v5 = {}
for label, n, (kind, nm) in ROWS5:
    v5[n] = [cells(n, V3P[p], k) for p, k in V3SYN] + [real_cells(ds, kind, nm, e, seed=3031) if kind else None for ds, e in V3REAL]
ncol5 = len(V3SYN) + len(V3REAL)
best5 = {}
for i in range(ncol5):
    cand = [(round(float(v5[n][i].mean()), 3), n) for _, n, _ in ROWS5 if v5[n][i] is not None]
    best5[i] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
lines = ['\\begin{table}[t]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
         '\\caption{\\label{tab:v3}Protocol v3 (new panels and targets). Mean test NLL. G1: 256 fresh five-sensor tasks by number of missing query sensors $k$; '
         'G3: 128 fresh 16-sensor tasks, $k=2$. Beijing: log NO$_2$ and log CO at each station from the other 11 stations, 2016--2017, natural missingness and six '
         'further sensors removed; learned models fine-tuned with the recipe of protocol v2. Best entry per column in bold.}',
         '\\begin{tabular}{l' + 'c' * ncol5 + '}', '\\toprule',
         ' & \\multicolumn{4}{c}{G1, $P{=}5$} & G3 & \\multicolumn{2}{c}{Beijing NO$_2$} & \\multicolumn{2}{c}{Beijing CO}\\\\',
         'Method & $k{=}0$ & $k{=}1$ & $k{=}2$ & $k{=}3$ & $P{=}16$ & natural & $+6$ & natural & $+6$\\\\', '\\midrule']
for label, n, _ in ROWS5:
    row = []
    for i in range(ncol5):
        v = v5[n][i]
        s_ = f3(None if v is None else float(v.mean()))
        row.append(f'\\textbf{{{s_}}}' if n in best5[i] else s_)
    if n == 'lift1':
        lines.append('\\midrule')
    lines.append(f'{label} & ' + ' & '.join(row) + '\\\\')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'v3_results_table.tex').write_text('\n'.join(lines) + '\n')
for p, k in V3SYN:
    l3, p3, f3_, a3 = cells('lct', V3P[p], k), cells('pfn', V3P[p], k), cells('fa1', V3P[p], k), cells('lift1', V3P[p], k)
    for other, key in ((p3, 'pfn'), (f3_, 'fa'), (a3, 'lift')):
        if l3 is not None and other is not None:
            m, lo, hi = ci(other, l3); mac(f'Vthree{p}k{k}lctvs{key}', f3(m, True)); mac(f'Vthree{p}k{k}lctvs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
    if a3 is not None and p3 is not None:
        m, lo, hi = ci(p3, a3); mac(f'Vthree{p}k{k}liftvspfn', f3(m, True)); mac(f'Vthree{p}k{k}liftvspfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')
for ds in ('beijing_no2', 'beijing_co'):
    pan = panel(f'real_{ds}_s3031')
    if pan is None:
        continue
    keys = pan['pool']['keys']
    mac(f'Vthree{ds.replace("_", "")}episodes', str(pan['meta']['episodes']))
    for e in (0, 3, 6):
        l3 = real_cells(ds, 'ft', 'lct_s1_ft', e, seed=3031)
        for (kind, nm), key in ((('ft', 'pfn_s1_ft'), 'pfn'), (('ft', 'lift1_s{1,2,3}_ft'), 'lift'), (('ref', 'blr_cc'), 'blr'), (('ref', 'fa1'), 'fa')):
            b = real_cells(ds, kind, nm, e, seed=3031)
            if l3 is not None and b is not None:
                m, lo, hi = week_ci(b, l3, keys); mac(f'Vthree{ds.replace("_", "")}e{e}lctvs{key}', f3(m, True)); mac(f'Vthree{ds.replace("_", "")}e{e}lctvs{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        a3, p3 = real_cells(ds, 'ft', 'lift1_s{1,2,3}_ft', e, seed=3031), real_cells(ds, 'ft', 'pfn_s1_ft', e, seed=3031)
        if l3 is not None and a3 is not None and p3 is not None and a3.mean() > p3.mean():
            mac(f'Vthree{ds.replace("_", "")}e{e}recovered', f'{100 * (a3.mean() - l3.mean()) / (a3.mean() - p3.mean()):.0f}')
        for v_, key in ((l3, 'lct'), (a3, 'lift'), (p3, 'pfn')):
            if v_ is not None:
                mac(f'Vthree{ds.replace("_", "")}e{e}val{key}', f3(float(v_.mean())))
        if a3 is not None and p3 is not None:
            m, lo, hi = week_ci(p3, a3, keys); mac(f'Vthree{ds.replace("_", "")}e{e}liftvspfn', f3(m, True)); mac(f'Vthree{ds.replace("_", "")}e{e}liftvspfnci', f'[{f3(lo, True)}, {f3(hi, True)}]')

# ------------------------------------------------------------------ pre-registered endpoints, protocols v2 and v3
EP = [('confirm_v2.json', 'v2', {'E1_F1_k2_lift1_vs_nlfa': ('E1', 'F1, $k=2$', 'lifted cavity network vs NL-FA'),
                                  'E2_F1_k2_lift1_vs_pfn': ('E2', 'F1, $k=2$', 'lifted cavity network vs transformer'),
                                  'E3a_F3_k2_lift1_vs_fa1': ('E3a', 'F3 (16 sensors), $k=2$', 'lifted cavity network vs FA-Gaussian'),
                                  'E3b_F3_k2_lift1_vs_pfn': ('E3b', 'F3 (16 sensors), $k=2$', 'lifted cavity network vs transformer'),
                                  'E4a_beijing_nat_lift1ft_vs_blr': ('E4a', 'Beijing PM$_{2.5}$', 'lifted cavity network (FT) vs BLR'),
                                  'E4b_beijing_nat_lift1ft_vs_pfnft': ('E4b', 'Beijing PM$_{2.5}$', 'lifted cavity network (FT) vs transformer (FT)')}),
      ('confirm_v3.json', 'v3', {'E5_G1_k2_lct_vs_pfn': ('E5', 'G1, $k=2$', 'LCT vs transformer'),
                                  'E6_G3_k2_lct_vs_pfn': ('E6', 'G3 (16 sensors), $k=2$', 'LCT vs transformer'),
                                  'E7a_bjno2_lctft_vs_lift1ft': ('E7a', 'Beijing NO$_2$', 'LCT (FT) vs lifted cavity network (FT)'),
                                  'E7b_bjno2_lctft_noninf_pfnft': ('E7b', 'Beijing NO$_2$', 'LCT (FT) vs transformer (FT), non-inferiority')}),
      ('confirm_v4.json', 'v4', {'E8_H1_k2_lctL_vs_pfnL': ('E8', 'H1, $k=2$', 'LCT-L vs transformer-L'),
                                  'E9_H3_k2_lctL_vs_pfnL': ('E9', 'H3 (16 sensors), $k=2$', 'LCT-L vs transformer-L'),
                                  'E10_bjnew_nat_lctLft_noninf_pfnLft': ('E10', 'Beijing-new', 'LCT-L (FT) vs transformer-L (FT), non-inferiority'),
                                  'E11_bjnew_plus6_lctLft_vs_pfnLft': ('E11', 'Beijing-new, $+6$ removed', 'LCT-L (FT) vs transformer-L (FT)'),
                                  'E12_bjnew_nat_lctLft_vs_tabpfn': ('E12', 'Beijing-new', 'LCT-L (FT) vs TabPFN v2 (in context)')})]
lines = ['\\begin{table}[t]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
         '\\caption{\\label{tab:endpoints}Every pre-registered endpoint of protocols v2--v4, each committed before its panels or targets existed '
         '(protocol v1: Appendix~\\ref{app:audit}). Gain in nats of NLL, positive when the first-named model is better; paired 95\\% intervals, cluster-robust '
         'over weeks on Beijing. v4 endpoints average three training seeds per task. Pass: gain $\\geq0.01$ and lower bound $>0$, except the '
         'non-inferiority endpoints E7b and E10 (lower bound $>-0.02$).}',
         '\\begin{tabular}{llllc}', '\\toprule', 'ID & Data & Comparison & Gain [95\\% CI] & Result\\\\', '\\midrule']
for fname, proto, desc in EP:
    f = R / fname
    res = json.loads(f.read_text()) if f.exists() else {}
    if proto != 'v2':
        lines.append('\\midrule')
    for k, (tagk, data_, comp) in desc.items():
        v = res.get(k)
        if v is None:
            lines.append(f'{tagk} & {data_} & {comp} & pending & --\\\\')
        else:
            verdict = 'pass' if v['passed'] else '\\textbf{fail}'
            lines.append(f"{tagk} & {data_} & {comp} & {f3(v['gain'], True)} [{f3(v['lo'], True)}, {f3(v['hi'], True)}] & {verdict}\\\\")
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'generated' / 'endpoints_table.tex').write_text('\n'.join(lines) + '\n')



# ------------------------------------------------------------------ exploratory (post hoc) fine-tuning variants on Beijing PM2.5
XP = Path(a.explore)
for nm, key in (('lift2_s1_ft', 'Xplifttwo'), ('lift1_s1_ft10k', 'Xpliftlong')):
    f = XP / 'beijing' / nm / 'cells_real_beijing_s2027.npz'
    if f.exists():
        mac(key, f3(float(np.load(f)['e0_nll'].mean())))
v_ = real_cells('beijing', 'ft', 'lift1_s1_ft', 0)
if v_ is not None:
    mac('Xpliftone', f3(float(v_.mean())))
XROWS = [('Lifted cavity network, standard recipe (seed 1)', RR / 'beijing' / 'lift1_s1_ft'), ('\\quad with $K=2$', XP / 'beijing' / 'lift2_s1_ft'),
         ('\\quad with $5\\times$ fine-tuning steps', XP / 'beijing' / 'lift1_s1_ft10k'), ('Transformer, standard recipe', RR / 'beijing' / 'pfn_s1_ft')]
if all((d / 'cells_real_beijing_s2027.npz').exists() for _, d in XROWS):
    lines = ['\\begin{table}[h]', '\\centering\\small', '\\caption{\\label{tab:explore}Post-hoc fine-tuning variants on Beijing PM$_{2.5}$ (test period of protocol v2; '
             'mean test NLL). Neither more nuisance dimensions nor longer fine-tuning closes the gap to the fine-tuned transformer.}',
             '\\begin{tabular}{lccc}', '\\toprule', 'Fine-tuned model & natural & $+3$ & $+6$\\\\', '\\midrule']
    for label, d in XROWS:
        z = np.load(d / 'cells_real_beijing_s2027.npz')
        lines.append(f'{label} & ' + ' & '.join(f3(float(z[f"e{e}_nll"].mean())) for e in (0, 3, 6)) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'explore_table.tex').write_text('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ untrained lifted network (its own anchor's closed form) on F1
fu = Path(a.init) / 'lift1_untrained' / f'cells_{PANEL["F1"]}.npz'
if fu.exists():
    zu = np.load(fu)
    for k in range(4):
        mac(f'Untrainedk{"abcd"[k]}', f3(float(zu[f'k{k}_nll'].mean(1).mean())))


# ------------------------------------------------------------------ appendix figure: training curves of the two transformers
curves = []
for nm, lab, col, ls in (('pfn_s1', 'Transformer, seed 1', '#eb6834', '-'), ('pfn_s2', 'Transformer, seed 2', '#eb6834', ':'),
                         ('lct_s1', 'Lifted cavity transformer, seed 1', '#2a78d6', '-'), ('lct_s2', 'Lifted cavity transformer, seed 2', '#2a78d6', ':')):
    f = R / nm / 'train.json'
    if f.exists():
        tr = json.loads(f.read_text()).get('trace', [])
        if tr:
            curves.append((lab, col, ls, [t['step'] for t in tr], [t['ema_loss'] for t in tr]))
fp_, fl_ = R / 'pfn_s1' / 'train.json', R / 'lct_s1' / 'train.json'
if fp_.exists() and fl_.exists():
    tp_, tl_ = json.loads(fp_.read_text())['trace'], json.loads(fl_.read_text())['trace']
    if tl_ and tl_[-1]['step'] == 30000:
        pf_ = tp_[-1]['ema_loss']; reach = next(t['step'] for t in tl_ if t['ema_loss'] <= pf_)
        mac('Trainlctreach', f'{reach:,}'.replace(',', '{,}'))
        mac('Trainlctgap', f"{np.mean([t['ema_loss'] for t in tp_[-10:]]) - np.mean([t['ema_loss'] for t in tl_[-10:]]):.3f}")
if curves:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for lab, col, ls, xs, ys in curves:
        ax.plot(xs, ys, color=col, ls=ls, lw=1.5, label=lab)
    ax.set_ylim(-0.2, 0.4); ax.set_xlabel('training step (16 fresh tasks $\\times$ 16 queries)', fontsize=8)
    ax.set_ylabel('training NLL (moving average)', fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'figures' / 'training.pdf', bbox_inches='tight')


# ------------------------------------------------------------------ appendix: second training seeds of both transformers (post hoc)
def seed_cells(name, seed, tag, k=2):
    f = R / f'{name}_s{seed}' / f'cells_{tag}.npz'
    return np.load(f)[f'k{k}_nll'].mean(1) if f.exists() else None


SROWS = [('E2', 'F1, $k=2$', 'lifted vs transformer', PANEL['F1'], 'lift1', 'pfn'),
         ('E3b', 'F3, $k=2$', 'lifted vs transformer', PANEL['F3'], 'lift1', 'pfn'),
         ('E5', 'G1, $k=2$', 'LCT vs transformer', V3P['G1'], 'lct', 'pfn'),
         ('E6', 'G3, $k=2$', 'LCT vs transformer', V3P['G3'], 'lct', 'pfn')]
lines, any_s2 = [], False
for tagk, data_, comp, tag, new, base in SROWS:
    cols = []
    for sb, sn in ((1, 1), (2, 2), ('avg', 'avg')):
        def get(nm, sd):
            if nm == 'lift1':
                return cells('lift1', tag, 2)
            if sd == 'avg':
                vs = [seed_cells(nm, s_, tag) for s_ in (1, 2)]
                return np.mean(vs, 0) if all(v is not None for v in vs) else None
            return seed_cells(nm, sd, tag)
        b, n = get(base, sb), get(new, sn)
        if b is None or n is None:
            cols.append('--'); continue
        if sb == 2:
            any_s2 = True
        m, lo, hi = ci(b, n); cols.append(f'{f3(m, True)} [{f3(lo, True)}, {f3(hi, True)}]')
    lines.append(f'{tagk} & {data_} & {comp} & ' + ' & '.join(cols) + '\\\\')
# real: Beijing PM2.5 (v2 target), fine-tuned
bp = panel('real_beijing_s2027')
if bp is not None:
    cols = []
    for sd in (1, 2, 'avg'):
        b = real_cells('beijing', 'ft', f'pfn_s{sd}_ft', 0) if sd != 'avg' else (
            np.mean([real_cells('beijing', 'ft', f'pfn_s{s_}_ft', 0) for s_ in (1, 2)], 0)
            if all(real_cells('beijing', 'ft', f'pfn_s{s_}_ft', 0) is not None for s_ in (1, 2)) else None)
        n = real_cells('beijing', 'ft', 'lift1_s{1,2,3}_ft', 0)
        if b is None or n is None:
            cols.append('--'); continue
        if sd == 2:
            any_s2 = True
        m, lo, hi = week_ci(b, n, bp['pool']['keys']); cols.append(f'{f3(m, True)} [{f3(lo, True)}, {f3(hi, True)}]')
    lines.append('E4b & Beijing PM$_{2.5}$ & lifted vs transformer (FT) & ' + ' & '.join(cols) + '\\\\')
# real: Beijing NO2 (v3 target), fine-tuned
bj = panel('real_beijing_no2_s3031')
if bj is not None:
    keys = bj['pool']['keys']
    def rft(nm):
        return real_cells('beijing_no2', 'ft', nm, 0, seed=3031)
    for tagk, comp, new, base in (('E7a', 'LCT vs lifted (FT)', 'lct', 'lift'), ('E7b', 'LCT vs transformer (FT)', 'lct', 'pfn')):
        cols = []
        for sd in (1, 2, 'avg'):
            def g(nm, sd_):
                if nm == 'lift':
                    return rft('lift1_s{1,2,3}_ft')
                if sd_ == 'avg':
                    vs = [rft(f'{nm}_s{s_}_ft') for s_ in (1, 2)]
                    return np.mean(vs, 0) if all(v is not None for v in vs) else None
                return rft(f'{nm}_s{sd_}_ft')
            b, n = g(base, sd), g(new, sd)
            if b is None or n is None:
                cols.append('--'); continue
            if sd == 2:
                any_s2 = True
            m, lo, hi = week_ci(b, n, keys); cols.append(f'{f3(m, True)} [{f3(lo, True)}, {f3(hi, True)}]')
        lines.append(f'{tagk} & Beijing NO$_2$ & {comp} & ' + ' & '.join(cols) + '\\\\')
if any_s2:
    tab_ = ['\\begin{table}[h]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{2pt}',
            '\\caption{\\label{tab:seeds}Second training seeds of the transformer and the lifted cavity transformer (post hoc). Endpoint comparisons '
            'recomputed with each seed and with both averaged; the lifted cavity network always averages its three seeds. Gains in nats, paired 95\\% '
            'intervals (cluster-robust over weeks on Beijing).}',
            '\\begin{tabular}{llllll}', '\\toprule', 'ID & Data & Comparison & Seed 1 (protocol) & Seed 2 & Seeds averaged\\\\', '\\midrule'] + lines + \
           ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'seeds_table.tex').write_text('\n'.join(tab_) + '\n')
else:
    (OUT / 'generated' / 'seeds_table.tex').write_text('% second seeds not available yet\n')


for ds_ in ('airq_co', 'airq_no2'):
    for e_ in (0, 2):
        la2, pa2, lc2 = (real_cells(ds_, 'ft', nm_, e_) for nm_ in ('lift1_s{1,2,3}_ft', 'pfn_s1_ft', 'lct_s1_ft'))
        if all(v is not None for v in (la2, pa2, lc2)):
            mac(f'Xplct{ds_.replace("_", "")}e{e_}recovered', f'{100 * (la2.mean() - lc2.mean()) / (la2.mean() - pa2.mean()):.0f}')
ls_ = [real_cells('beijing_no2', 'ft', f'lct_s{s_}_ft', 0, seed=3031) for s_ in (1, 2)]
ps_ = [real_cells('beijing_no2', 'ft', f'pfn_s{s_}_ft', 0, seed=3031) for s_ in (1, 2)]
lf_ = real_cells('beijing_no2', 'ft', 'lift1_s{1,2,3}_ft', 0, seed=3031)
if lf_ is not None and all(v is not None for v in ls_ + ps_):
    lm_, pm_ = np.mean(ls_, 0).mean(), np.mean(ps_, 0).mean()
    mac('Vthreeseedavgrecovered', f'{100 * (lf_.mean() - lm_) / (lf_.mean() - pm_):.0f}')
    mac('Vthreeseedtworecovered', f'{100 * (lf_.mean() - ls_[1].mean()) / (lf_.mean() - ps_[1].mean()):.0f}')
la_, pa_, lc_ = real_cells('beijing', 'ft', 'lift1_s{1,2,3}_ft', 0), real_cells('beijing', 'ft', 'pfn_s1_ft', 0), real_cells('beijing', 'ft', 'lct_s1_ft', 0)
if all(v is not None for v in (la_, pa_, lc_)):
    mac('Xplctpmrecovered', f'{100 * (la_.mean() - lc_.mean()) / (la_.mean() - pa_.mean()):.0f}')
    mac('Xplctpmval', f3(float(lc_.mean())))

# ------------------------------------------------------------------ protocol v4: scale, pretrained TabPFN v2, new real targets
V4P = dict(H1='v2_s20261401_n256_p5_nl0.4_sr48', H3='v2_s20261403_n128_p16_nl0.4_sr48')
V4NEW = (('beijing_pm10', 'PM$_{10}$'), ('beijing_so2', 'SO$_2$'), ('beijing_o3', 'O$_3$'))
V4SEEDS = (1, 2, 3)


def v4_syn(name, tag, k):
    """Per-task NLL on a v4 synthetic panel; learned models average the seeds that exist (v4 models need all three)."""
    p = panel(tag)
    if p is None:
        return None, 0
    if name in p['refs'][k]:
        v = p['refs'][k][name]['nll']
        return (None, 0) if np.isnan(v).all() else (v.mean(1), 0)
    if name == 'tabpfn_v2':
        f = R / 'tabpfn_v2' / f'cells_{tag}.npz'
        return (np.load(f)[f'k{k}_nll'].mean(1), 0) if f.exists() else (None, 0)
    fs = [R / f'{name}_s{s_}' / f'cells_{tag}.npz' for s_ in V4SEEDS]
    vs = [np.load(f)[f'k{k}_nll'].mean(1) for f in fs if f.exists()]
    if name in ('pfn_L', 'lct_L') and len(vs) < len(V4SEEDS):
        return None, len(vs)
    return (np.mean(vs, 0), len(vs)) if vs else (None, 0)


def v4_real(name, ds, e):
    """Per-episode NLL on a Beijing-new panel and its week keys; fine-tuned models average three seeds."""
    tag = f'real_{ds}_s4041'
    p = panel(tag)
    if p is None:
        return None, None
    keys = [k_.split('|')[0] for k_ in p['pool']['keys']]
    if name in p['refs'][e]:
        v = p['refs'][e][name]['nll']
        return (None if np.isnan(v).all() else v), keys
    if name == 'tabpfn_v2':
        f = R / 'tabpfn_v2' / f'cells_{tag}.npz'
        return (np.load(f)[f'e{e}_nll'] if f.exists() else None), keys
    fs = [RR / ds / f'{name}_s{s_}_ft' / f'cells_{tag}.npz' for s_ in V4SEEDS]
    if not all(f.exists() for f in fs):
        return None, keys
    return np.mean([np.load(f)[f'e{e}_nll'] for f in fs], 0), keys


def v4_pool(name, e):
    vs, ks = [], []
    for ds, _ in V4NEW:
        v, k_ = v4_real(name, ds, e)
        if v is None:
            return None, None
        vs.append(v); ks += k_
    return np.concatenate(vs), ks


V4ROWS = [('Oracle (true parameters)', 'oracle', 'priv', '--'), ('FA-Gaussian', 'fa1', 'cf', '--'), ('NL-FA (cubic, tuned)', 'nlfa', 'cf', '--'),
          ('Bayesian linear regression', 'blr_cc', 'cf', '--'), ('Ridge (complete case)', 'ridge_cc', 'cf', '--'),
          ('Lifted cavity network', 'lift1', 'small', '8k'), ('Transformer', 'pfn', 'small', '203k'), ('Lifted cavity transformer', 'lct', 'small', '203k'),
          ('Transformer-L', 'pfn_L', 'large', '2.1M'), ('\\textbf{Lifted cavity transformer-L (ours)}', 'lct_L', 'large', '2.1M'),
          ('TabPFN v2, pretrained (in context only)', 'tabpfn_v2', 'pre', '--')]
V4COLS = [('H1', 0), ('H1', 2), ('H1', 3), ('H3', 2)] + [(ds, 0) for ds, _ in V4NEW] + [('pool', 0), ('pool', 6)]


def v4_col(name, col):
    src, k = col
    if src in V4P:
        return v4_syn(name, V4P[src], k)[0]
    if src == 'pool':
        return v4_pool(name, k)[0]
    return v4_real(name, src, k)[0]


v4vals = {n: [v4_col(n, c) for c in V4COLS] for _, n, _, _ in V4ROWS}
if any(v is not None for n in ('pfn_L', 'lct_L', 'tabpfn_v2') for v in v4vals[n]):
    head = ('Method & Params & \\multicolumn{3}{c}{H1 (5 sensors)} & H3 (16) & \\multicolumn{3}{c}{Beijing-new, natural} & \\multicolumn{2}{c}{pooled}\\\\',
            ' & & $k{=}0$ & $k{=}2$ & $k{=}3$ & $k{=}2$ & ' + ' & '.join(l for _, l in V4NEW) + ' & natural & $+6$\\\\')
    nsmall = max(v4_syn('pfn', V4P['H1'], 2)[1], v4_syn('lct', V4P['H1'], 2)[1])
    lines = ['\\begin{table}[t]', '\\centering\\scriptsize', '\\setlength{\\tabcolsep}{2.2pt}',
             '\\caption{\\label{tab:v4}Protocol v4: mean test NLL. Synthetic: fresh panels H1 and H3 by number of missing query sensors '
             '(learned models were trained with at most one). Beijing-new: log PM$_{10}$, SO$_2$ and O$_3$ at each of 12 stations from the other 11 '
             '(2016--2017, dequantized), natural missingness and six further stations removed; learned models are fine-tuned with one recipe, TabPFN '
             'is used in context. The -L models average three training seeds, the lifted cavity network three, and the 203k transformers '
             + ('two' if nsmall == 2 else 'three') + '. Best non-privileged entry per column in bold.}',
             '\\begin{tabular}{llccccccccc}', '\\toprule', head[0], head[1], '\\midrule']
    best = {}
    for i in range(len(V4COLS)):
        cand = [(round(float(v4vals[n][i].mean()), 3), n) for _, n, g, _ in V4ROWS if g != 'priv' and v4vals[n][i] is not None]
        best[i] = {n for m, n in cand if m == min(cand)[0]} if cand else set()
    last = None
    for label, n, g, par in V4ROWS:
        if last is not None and g != last:
            lines.append('\\midrule')
        last = g
        row = []
        for i in range(len(V4COLS)):
            v = v4vals[n][i]; s_ = f3(None if v is None else float(v.mean()))
            row.append(f'\\textbf{{{s_}}}' if n in best[i] else s_)
        lines.append(f'{label} & {par} & ' + ' & '.join(row) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'v4_table.tex').write_text('\n'.join(lines) + '\n')
else:
    (OUT / 'generated' / 'v4_table.tex').write_text('% protocol v4 results pending\n')

# v4 macros: endpoints, scale contrasts, TabPFN contrasts
cv4 = R / 'confirm_v4.json'
if cv4.exists():
    res4 = json.loads(cv4.read_text())
    for key, short in (('E8_H1_k2_lctL_vs_pfnL', 'Eeight'), ('E9_H3_k2_lctL_vs_pfnL', 'Enine'), ('E10_bjnew_nat_lctLft_noninf_pfnLft', 'Eten'),
                       ('E11_bjnew_plus6_lctLft_vs_pfnLft', 'Eeleven'), ('E12_bjnew_nat_lctLft_vs_tabpfn', 'Etwelve')):
        v = res4.get(key)
        if v:
            mac(f'Vfour{short}', f3(v['gain'], True)); mac(f'Vfour{short}ci', f"[{f3(v['lo'], True)}, {f3(v['hi'], True)}]")
            mac(f'Vfour{short}result', 'passes' if v['passed'] else 'fails')
    mac('Vfourpassed', str(sum(bool(res4.get(k_, {}).get('passed')) for k_ in res4 if k_.startswith('E'))))
for src in ('H1', 'H3'):
    for k in ((0, 1, 2, 3) if src == 'H1' else (2,)):
        tag = V4P[src]
        got = {n: v4_syn(n, tag, k)[0] for n in ('pfn', 'lct', 'pfn_L', 'lct_L', 'lift1', 'fa1', 'tabpfn_v2', 'oracle')}
        for a_, b_, key in (('pfn_L', 'lct_L', 'lctLvspfnL'), ('pfn', 'lct', 'lctvspfn'), ('lct', 'lct_L', 'lctLvslct'), ('pfn', 'pfn_L', 'pfnLvspfn'),
                            ('tabpfn_v2', 'lct_L', 'lctLvstab'), ('tabpfn_v2', 'pfn_L', 'pfnLvstab'), ('lift1', 'lct_L', 'lctLvslift'),
                            ('fa1', 'lct_L', 'lctLvsfa'), ('fa1', 'tabpfn_v2', 'tabvsfa'), ('lct_L', 'oracle', 'oraclevslctL')):
            if got[a_] is not None and got[b_] is not None:
                m, lo, hi = ci(got[a_], got[b_])
                mac(f'Vfour{src}k{k}{key}', f3(m, True)); mac(f'Vfour{src}k{k}{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
        for n in got:
            if got[n] is not None:
                mac(f'Vfour{src}k{k}val{n.replace("_", "")}', f3(float(got[n].mean())))
for e in (0, 3, 6):
    got = {n: v4_pool(n, e) for n in ('pfn_L', 'lct_L', 'lift1', 'tabpfn_v2', 'blr_cc', 'fa1')}
    for a_, b_, key in (('pfn_L', 'lct_L', 'lctLvspfnL'), ('tabpfn_v2', 'lct_L', 'lctLvstab'), ('tabpfn_v2', 'pfn_L', 'pfnLvstab'),
                        ('lift1', 'lct_L', 'lctLvslift'), ('blr_cc', 'lct_L', 'lctLvsblr'), ('blr_cc', 'tabpfn_v2', 'tabvsblr'), ('lift1', 'pfn_L', 'pfnLvslift')):
        if got[a_][0] is not None and got[b_][0] is not None:
            m, lo, hi = week_ci(got[a_][0], got[b_][0], [k_ + '|' for k_ in got[a_][1]])
            mac(f'Vfourbjnewe{e}{key}', f3(m, True)); mac(f'Vfourbjnewe{e}{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
    for n, (v, _) in got.items():
        if v is not None:
            mac(f'Vfourbjnewe{e}val{n.replace("_", "")}', f3(float(v.mean())))
for ds, _ in V4NEW:
    p_ = panel(f'real_{ds}_s4041')
    if p_ is not None:
        mac(f'Vfour{ds.replace("_", "")}episodes', str(p_['meta']['episodes']))
    for e in (0, 6):
        got = {n: v4_real(n, ds, e) for n in ('pfn_L', 'lct_L', 'tabpfn_v2', 'lift1')}
        for a_, b_, key in (('pfn_L', 'lct_L', 'lctLvspfnL'), ('tabpfn_v2', 'lct_L', 'lctLvstab'), ('tabpfn_v2', 'pfn_L', 'pfnLvstab')):
            if got[a_][0] is not None and got[b_][0] is not None:
                m, lo, hi = week_ci(got[a_][0], got[b_][0], [k_ + '|' for k_ in got[a_][1]])
                mac(f'Vfour{ds.replace("_", "")}e{e}{key}', f3(m, True)); mac(f'Vfour{ds.replace("_", "")}e{e}{key}ci', f'[{f3(lo, True)}, {f3(hi, True)}]')
for nm in ('pfn_L', 'lct_L'):
    tj = [R / f'{nm}_s{s_}' / 'train.json' for s_ in V4SEEDS]
    if all(f.exists() for f in tj):
        js = [json.loads(f.read_text()) for f in tj]
        mac(f'Vfour{nm.replace("_", "")}params', f"{js[0]['parameters']:,}".replace(',', '{,}'))
        mac(f'Vfour{nm.replace("_", "")}gpuhours', f"{sum(j_['seconds'] for j_ in js) / 3600:.1f}")
tj = R / 'tabpfn_v2' / 'train.json'
if tj.exists():
    jt = json.loads(tj.read_text())
    mac('Vfourtabpfnpackage', str(jt.get('package', '?')))
    if jt.get('gpu'):
        mac('Vfourgpu', jt['gpu'].replace('Tesla ', '').replace('NVIDIA ', ''))


# ------------------------------------------------------------------ compute table (appendix)
cf = R / 'compute_cost.json'
if cf.exists():
    cc = json.loads(cf.read_text())
    lf = R / 'lct_s1' / 'train.json'
    if lf.exists() and 'lct_s1' in cc:
        cc['lct_s1']['train_seconds'] = json.loads(lf.read_text()).get('seconds')
    CROWS = [('Lifted cavity network', 'lift1_s1'), ('Residual MLP on FA anchor', 'anchor_mlp_s1'), ('Transformer (TabPFN-v2 style)', 'pfn_s1'),
             ('Lifted cavity transformer', 'lct_s1'), ('FA-Gaussian (closed form)', 'fa1'), ('NL-FA (closed form)', 'nlfa')]
    lines = ['\\begin{table}[h]', '\\centering\\small', '\\caption{\\label{tab:compute}Compute. Training on one CPU core (core-hours); inference (ms/task) is single-threaded '
             'time per task for 480 queries (ten masks $\\times$ 48 rows) on F1. Anchors are precomputed in batch and excluded; the closed-form rows include their '
             'unbatched NumPy fits.}',
             '\\begin{tabular}{lrrrr}', '\\toprule', 'Model & Parameters & Training steps & Core-hours & ms/task\\\\', '\\midrule']
    for label, key in CROWS:
        v = cc.get(key)
        if v is None:
            lines.append(f'{label} & -- & -- & -- & --\\\\'); continue
        par = f"{v['parameters']:,}".replace(',', '{,}') if 'parameters' in v else '--'
        st = f"{v['train_steps']:,}".replace(',', '{,}') if v.get('train_steps') else '--'
        hrs = f"{v['train_seconds'] / 3600:.2f}" if v.get('train_seconds') else '--'
        lines.append(f"{label} & {par} & {st} & {hrs} & {v['ms_per_task']:.1f}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'generated' / 'compute_table.tex').write_text('\n'.join(lines) + '\n')
    if 'lift1_s1' in cc and 'pfn_s1' in cc:
        mac('Computetrainratio', f"{cc['pfn_s1']['train_seconds'] / cc['lift1_s1']['train_seconds']:.0f}")

(OUT / 'generated' / 'numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
print(len(macros), 'macros;', ', '.join(sorted(macros)[:12]), '...')
