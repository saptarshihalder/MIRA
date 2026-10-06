"""Generate the markdown result tables for docs/LIFTED_CAVITY_FINDINGS.md from saved scores (no model re-runs).

python make_report.py --runs runs --runs-airq runs_airq --cache cache --oracle oracle_summary.json --repo-panel repo_panel_scores.json
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

REP = Path(__file__).resolve().parents[2] / 'artifacts' / 'reports' / 'lifted_cavity_v1'
CCH = Path(__file__).resolve().parents[2] / 'artifacts' / 'runs' / 'lifted_cavity' / 'cache'
ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=str(REP / 'runs')); ap.add_argument('--runs-airq', default=str(REP / 'airq_co'))
ap.add_argument('--cache', default=str(CCH)); ap.add_argument('--oracle', default=str(REP / 'oracle_audit' / 'oracle_summary.json'))
ap.add_argument('--repo-panel', default=str(REP / 'repo_panel_scores.json')); ap.add_argument('--out', default=str(REP / 'tables.md'))
ap.add_argument('--airq-panel', default=str(CCH / 'airq_s7.pt'))
ap.add_argument('--airq-no2-panel', default=str(CCH / 'airq_no2_s7.pt')); ap.add_argument('--runs-airq-no2', default=str(REP / 'airq_no2'))
a = ap.parse_args()
runs, cache = Path(a.runs), Path(a.cache)
L = []
P = lambda s='': L.append(s)

LABEL = {'oracle': 'Exact Bayes oracle (privileged)', 'oracle_gauss': 'Moment-matched Gaussian oracle (privileged)',
         'pop_linear': 'Population joint Gaussian (privileged)', 'fa1': 'FA-Gaussian, support-only (closed form)',
         'em_gauss': 'EM-Gaussian, support-only (closed form)', 'ridge_cc': 'Complete-case ridge (closed form)',
         'ridge_repo': 'Repo ridge, mean-imputed (closed form)', 'lift1': '**Lifted cavity, K=1**', 'lift2': 'Lifted cavity, K=2',
         'lift1_static': 'Lifted sites, no cavity input', 'lift0': '1-D sites, same machinery (K=0)',
         'anchor_mlp': 'Residual MLP on the same anchor + encoders', 'repo_cavity_fresh': 'Repo anchored cavity, 60k tasks/20k steps',
         'repo_aggregate_fresh': 'Repo anchored aggregate, 60k tasks/20k steps', 'repo_mlp_fresh': 'Repo MLP, 60k tasks/20k steps',
         'repo_cavity_512': 'Repo anchored cavity, repo regime (512 tasks/3k steps)', 'repo_mlp_512': 'Repo MLP, repo regime',
         'repo_cavity_60k3k': 'Repo anchored cavity, 60k tasks/3k steps'}
ORDER = ['oracle', 'oracle_gauss', 'pop_linear', 'lift1', 'lift2', 'anchor_mlp', 'lift1_static', 'fa1', 'em_gauss', 'ridge_cc', 'lift0',
         'repo_aggregate_fresh', 'repo_mlp_fresh', 'repo_cavity_fresh', 'repo_cavity_60k3k', 'repo_cavity_512', 'repo_mlp_512', 'ridge_repo']


def panel_rows(tag):
    panel = torch.load(cache / f'{tag}.pt', weights_only=False)
    refs = panel['refs']; ks = list(refs)
    rows = {n: [{k: refs[k][n] for k in ks}] for n in refs[ks[0]]}
    for run in sorted(runs.iterdir()):
        f = run / f'cells_{tag}.npz'
        if f.exists():
            z = np.load(f); base = re.sub(r'_s\d+$', '', run.name)
            rows.setdefault(base, []).append({k: z[f'k{k}'] for k in ks})
    return rows, ks


def fmt(vals):
    m = np.mean(vals)
    return f'{m:.3f}' if len(vals) == 1 else f'{m:.3f} ± {np.std(vals, ddof=1):.3f}'


# ---- Table A: oracle audit on the repository's closed panels
orc = json.loads(Path(a.oracle).read_text()); rp = json.loads(Path(a.repo_panel).read_text())
P('**Table A. Repository development panels, all ten two-sensor deletions (mean Gaussian NLL, lower is better).**')
P('Rows marked privileged use the true task parameters; every other row sees only the 48 labeled support rows.')
P()
P('| Method | `cavity_site_v1` panel | `anchored_cavity_v1` panel |')
P('|---|---:|---:|')
for key, lab in (('oracle_nll', 'Exact Bayes oracle (privileged)'), ('pop_linear_nll', 'Population joint Gaussian (privileged)')):
    P(f'| {lab} | {orc["cavity_site_v1"][key]:.3f} | {orc["anchored_cavity_v1"][key]:.3f} |')
for lab, key in (('Lifted cavity K=1, seed 1 (this work)', 'lift1_s1'), ('Residual MLP on same anchor, seed 1', 'anchor_mlp_s1')):
    P(f'| {lab} | {rp["cavity_site_v1"].get(key, float("nan")):.3f} | {rp["anchored_cavity_v1"].get(key, float("nan")):.3f} |')
for key, lab in (('fa1_nll', 'FA-Gaussian, support-only'), ('em_gauss_nll', 'EM-Gaussian, support-only'), ('ridge_cc_nll', 'Complete-case ridge')):
    P(f'| {lab} | {orc["cavity_site_v1"][key]:.3f} | {orc["anchored_cavity_v1"][key]:.3f} |')
for lab, pat in (('Repo ridge (the gate\'s control)', 'ridge'), ('Repo cavity (3 seeds)', 'cavity_'), ('Repo full aggregate (3 seeds)', 'aggregate_'),
                 ('Repo static sites (3 seeds)', 'static_'), ('Repo larger MLP (3 seeds)', 'mlp_')):
    cells = []
    for p in ('cavity_site_v1', 'anchored_cavity_v1'):
        v = [x['nll'] for k, x in orc[p]['models'].items() if (k == pat if pat == 'ridge' else k.startswith(pat))]
        cells.append(f'{min(v):.3f}' if len(v) == 1 else (f'{min(v):.3f} (all seeds collapsed)' if max(v) - min(v) < 5e-4 else f'{min(v):.3f}–{max(v):.3f}'))
    P(f'| {lab} | {cells[0]} | {cells[1]} |')
P()

# ---- Table B: main fresh panel
tag = 'panel_s20261007_n256_nl0.4_sr48_sm0.2'
rows, ks = panel_rows(tag)
P('**Table B. Fresh panel (256 new tasks, seed 20261007). Mean NLL by number of missing query sensors. '
  'Learned models were trained only on masks with at most one missing sensor, so k = 2 and 3 are unseen patterns. '
  'Learned rows: mean ± s.d. over training seeds (n in brackets). Last column: share of the FA-Gaussian→oracle gap closed at k = 2.**')
P()
P('| Method | k=0 | k=1 | k=2 | k=3 | gap closed (k=2) |')
P('|---|---:|---:|---:|---:|---:|')
fa2, or2 = rows['fa1'][0][2].mean(), rows['oracle'][0][2].mean()
for name in ORDER:
    if name not in rows:
        continue
    lab = LABEL.get(name, name) + (f' [{len(rows[name])}]' if name not in ('oracle', 'oracle_gauss', 'pop_linear', 'fa1', 'em_gauss', 'ridge_cc', 'ridge_repo') else '')
    vals = {k: [r[k].mean() for r in rows[name]] for k in ks}
    gap = (fa2 - np.mean(vals[2])) / (fa2 - or2)
    P(f'| {lab} | ' + ' | '.join(fmt(vals[k]) for k in ks) + f' | {gap:.2f} |')
P()


# paired seed-averaged CI for lift1 vs anchor_mlp and vs fa1 at each k
def paired(a_rows, b_rows, k):
    a_ = np.mean([r[k] for r in a_rows], 0).mean(1); b_ = np.mean([r[k] for r in b_rows], 0).mean(1)
    d = b_ - a_; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return f'{m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]'


P('Paired task-level 95% intervals for the seed-averaged lifted cavity (positive = lifted cavity better):')
P()
P('| Comparison | k=0 | k=1 | k=2 | k=3 |')
P('|---|---:|---:|---:|---:|')
for other in ('fa1', 'em_gauss', 'anchor_mlp', 'lift1_static', 'lift0', 'repo_cavity_fresh'):
    if other in rows and 'lift1' in rows:
        P(f'| vs {LABEL[other]} | ' + ' | '.join(paired(rows['lift1'], rows[other], k) for k in ks) + ' |')
P()

# ---- Table C: shift panels
P('**Table C. Out-of-training-distribution panels (128 new tasks each). Share of the FA-Gaussian→oracle NLL gap closed, '
  'averaged over k = 0–3 (negative = worse than the closed form). Mean over available training seeds.**')
P()
shifts = [('panel_s20261008_n128_nl0.4_sr48_sm0.2_p8', '8 sensors (trained on 5)'), ('panel_s20261010_n128_nl0.8_sr48_sm0.2', 'Nonlinearity 0.8 (trained 0.4)'),
          ('panel_s20261009_n128_nl0.0_sr48_sm0.2', 'Linear sensors (trained 0.4)'), ('panel_s20261011_n128_nl0.4_sr16_sm0.2', '16 support rows (trained 48)'),
          ('panel_s20261012_n128_nl0.4_sr96_sm0.2', '96 support rows (trained 48)')]
names = ['lift1', 'anchor_mlp', 'lift1_static', 'lift0', 'em_gauss']
P('| Panel | oracle NLL (k=2) | FA-Gaussian NLL (k=2) | ' + ' | '.join(LABEL[n].strip('*') for n in names) + ' |')
P('|---|---:|---:|' + '---:|' * len(names))
for tag, lab in shifts:
    if not (cache / f'{tag}.pt').exists():
        continue
    rows, ks = panel_rows(tag)
    cells = []
    for n in names:
        if n not in rows:
            cells.append('n/a'); continue
        g = np.mean([np.mean([(rows['fa1'][0][k].mean() - r[k].mean()) / (rows['fa1'][0][k].mean() - rows['oracle'][0][k].mean()) for k in ks]) for r in rows[n]])
        cells.append(f'{g:.2f}' + (f' [{len(rows[n])}]' if n != 'em_gauss' else ''))
    P(f'| {lab} | {rows["oracle"][0][2].mean():.3f} | {rows["fa1"][0][2].mean():.3f} | ' + ' | '.join(cells) + ' |')
P()

# ---- Table D: Air Quality (computed from per-week cells; fine-tuned rows averaged over available pretraining seeds)
def airq_table(panel_path, ft_dir, title):
    part = torch.load(panel_path, weights_only=False)['split']['test']
    refs = part['refs']; n = part['pool']['n']
    em = {k: refs[k]['em_gauss'].mean(1) for k in refs}
    rows = [('EM-Gaussian, closed form (reference)', {k: refs[k]['em_gauss'].mean(1) for k in refs}),
            ('FA-Gaussian, closed form', {k: refs[k]['fa1'].mean(1) for k in refs}),
            ('Complete-case ridge', {k: refs[k]['ridge_cc'].mean(1) for k in refs}),
            ('Repo ridge, mean-imputed', {k: refs[k]['ridge_repo'].mean(1) for k in refs})]
    ft_dir = Path(ft_dir)
    for pat, lab in (('lift1_s[0-9]_ft', 'Lifted cavity, synthetic pretrain + fine-tune'), ('lift1_untrained_ft', 'Lifted cavity, fine-tune only (no pretrain)'),
                     ('lift1_static_s[0-9]_ft', 'Lifted sites, no cavity input, fine-tuned'), ('lift0_s[0-9]_ft', '1-D sites (K=0), fine-tuned'),
                     ('anchor_mlp_s[0-9]_ft', 'Residual MLP on same anchor, fine-tuned'), ('anchor_mlp_untrained_ft', 'Residual MLP, fine-tune only'),
                     ('repo_cavity_fresh_s[0-9]_ft', 'Repo 1-D cavity (retrained), fine-tuned')):
        fs = sorted(ft_dir.glob(pat + '/cells_airq_test.npz'))
        if fs:
            zs = [np.load(f) for f in fs]
            rows.append((lab + f' [{len(fs)}]', {k: np.mean([z[f'k{k}'] for z in zs], 0).mean(1) for k in refs}))
    P(title)
    P()
    P('| Method | NLL k=0 | NLL k=2 | gain vs EM, k=2 [95% CI] | gain vs EM, k=3 [95% CI] | weeks better (k=2) |')
    P('|---|---:|---:|---:|---:|---:|')
    for lab, r in rows:
        cells = []
        for k in (2, 3):
            d = em[k] - r[k]; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
            cells.append('–' if lab.startswith('EM-') else f'{m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]')
        wk = '–' if lab.startswith('EM-') else f'{int(((em[2] - r[2]) > 0).sum())}/{n}'
        P(f'| {lab} | {r[0].mean():.3f} | {r[2].mean():.3f} | {cells[0]} | {cells[1]} | {wk} |')
    P()


airq_table(a.airq_panel, a.runs_airq, '**Table D. UCI Air Quality, 23 later test weeks (windows starting 6 Oct 2004 – 23 Mar 2005), target log CO(GT), '
           '48 labeled support hours per week; fine-tuning used only weeks before 1 Oct 2004. Brackets: number of pretraining seeds averaged. '
           'Gains are paired over test weeks (positive = better than EM-Gaussian).**')
if Path(a.airq_no2_panel).exists() and Path(a.runs_airq_no2).exists():
    airq_table(a.airq_no2_panel, a.runs_airq_no2, '**Table E. Same protocol, target log NO2(GT): the frozen confirmation endpoint R1 (one pretraining seed).**')
Path(a.out).write_text('\n'.join(L) + '\n')
print('\n'.join(L))
