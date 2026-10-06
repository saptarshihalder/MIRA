"""Paper figure: NLL vs number of missing sensors. (a) synthetic confirmation panel with oracle; (b) Air Quality CO test weeks.

python make_figure.py --runs runs --cache cache --airq-ft runs_airq --out lifted_cavity_main.png
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REP, CCH = ROOT / 'artifacts' / 'reports' / 'lifted_cavity_v1', ROOT / 'artifacts' / 'runs' / 'lifted_cavity' / 'cache'
ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=str(REP / 'runs')); ap.add_argument('--cache', default=str(CCH))
ap.add_argument('--airq-ft', default=str(REP / 'airq_co')); ap.add_argument('--airq-panel', default=str(CCH / 'airq_s7.pt'))
ap.add_argument('--panel', default='panel_s20261101_n256_nl0.4_sr48_sm0.2'); ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'figures' / 'lifted_cavity_main.png'))
a = ap.parse_args()

INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb'
SERIES = [('lift1', 'Lifted cavity (ours)', '#2a78d6', 'o'), ('anchor_mlp', 'Residual MLP on same anchor', '#eb6834', 's'),
          ('fa1', 'FA-Gaussian, closed form', '#1baf7a', 'D'), ('em_gauss', 'EM-Gaussian, closed form', '#eda100', 'v'),
          ('repo_cavity_fresh', 'Repo 1-D cavity (retrained)', '#e87ba4', '^')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 10})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), facecolor=SURF)
ks = [0, 1, 2, 3]


def seed_mean(name, tag, runs):
    vals = [np.load(d / f'cells_{tag}.npz') for d in sorted(Path(runs).glob(f'{name}_s[0-9]')) if (d / f'cells_{tag}.npz').exists()]
    return [np.mean([v[f'k{k}'].mean() for v in vals]) for k in ks] if vals else None


# (a) synthetic
ax = axes[0]
panel = torch.load(Path(a.cache) / f'{a.panel}.pt', weights_only=False)['refs']
orc = [panel[k]['oracle'].mean() for k in ks]
ax.plot(ks, orc, color=INK2, lw=1.5, ls='--', zorder=2); ax.annotate('exact Bayes oracle', (1, orc[1]), xytext=(4, -12), textcoords='offset points', va='top', color=INK2, fontsize=8)
for name, label, color, marker in SERIES:
    y = [panel[k][name].mean() for k in ks] if name in panel[0] else seed_mean(name, a.panel, a.runs)
    if y is None:
        continue
    ax.plot(ks, y, color=color, lw=2, marker=marker, ms=5, label=label, zorder=3, markeredgecolor=SURF, markeredgewidth=.8)
lo, hi = ax.get_ylim()
ax.axvspan(1.5, 3.3, color='#efeeea', zorder=0)
ax.set_title('(a) Synthetic tasks, fresh panel', loc='left')
ax.set_xlabel('missing sensors at test (of 5); shaded = unseen in training'); ax.set_ylabel('test NLL (lower is better)')

# (b) Air Quality
ax = axes[1]
aq = torch.load(a.airq_panel, weights_only=False)['split']['test']['refs']
ft = Path(a.airq_ft)
for name, label, color, marker in SERIES:
    if name in aq[0]:
        y = [aq[k][name].mean() for k in ks]
    else:
        fs = sorted(ft.glob(f'{name}_s[0-9]_ft/cells_airq_test.npz'))
        if not fs:
            continue
        zs = [np.load(f) for f in fs]; y = [np.mean([z[f'k{k}'].mean() for z in zs]) for k in ks]
    ax.plot(ks, y, color=color, lw=2, marker=marker, ms=5, label=label, zorder=3, markeredgecolor=SURF, markeredgewidth=.8)
lo, hi = ax.get_ylim()
ax.axvspan(1.5, 3.3, color='#efeeea', zorder=0)
ax.set_title('(b) Air Quality, 23 later weeks (log CO)', loc='left')
ax.set_xlabel('missing sensors at test (of 5); shaded = unseen in training')
for ax in axes:
    ax.set_facecolor(SURF); ax.set_xticks(ks); ax.grid(axis='y', color='#e4e3df', lw=.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlim(-.2, 3.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0), labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, .86))
fig.savefig(a.out, dpi=200, facecolor=SURF)
print(a.out)
