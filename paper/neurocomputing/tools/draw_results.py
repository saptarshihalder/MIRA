"""Result figures of the Neurocomputing manuscript, drawn from the confirm_v*.json files in ../data
(copied from artifacts/reports/lifted_cavity_paper/runs on branch lifted-site-pool-v5). No model is run and no
number is copied by hand. Adapted from experiments/lifted_cavity/make_paper_figs.py for the two column CAS layout
(text width 6.84 in) with plain language titles.

    python tools/draw_results.py

Original notes:

    python make_paper_figs.py --runs artifacts/reports/lifted_cavity_paper/runs --out paper/upt

figures/pool_fresh.pdf     Figure 2: (a) robustness to failing inputs on the four 2026 networks (v11-v15);
                           (b) gain of the frozen v15 pool over TabPFN-3.5 on the fresh post-release tasks (v16, v17)
figures/crossfit.pdf       appendix: two-fold (v13 rule) vs four-fold (v15 rule) stacking, per network and condition
figures/weights.pdf        appendix: the stacking weight on UPT-SF against the two models' relative skill
figures/fresh_curves.pdf   appendix: mean NLL by inputs dropped on each fresh network
Figures are drawn at print size (the NeurIPS text is 5.5 in wide) and included without scaling. Per-network intervals
are t-intervals over network-windows (4 or 11 per network; descriptive); the pooled fresh endpoints are the
pre-registered ones (97.5%).
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = 6.84
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': .6,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'lines.linewidth': 1.2, 'lines.markersize': 3.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
Z = {0.95: 1.959964, 0.975: 2.241403}
OLD = [('dwd_p', 'Pressure'), ('aurn_pm25', 'PM$_{2.5}$'), ('usgs_ny', 'River NY'), ('usgs_wa', 'River WA')]
FRESH = [('dwd_p_f', 'Pressure', 16), ('dwd_t2m_f', 'Temperature', 17), ('aurn_pm25_f', 'PM$_{2.5}$', 16),
         ('aurn_no2_f', 'NO$_2$', 17), ('usgs_ny_f', 'River NY', 16), ('usgs_wa_f', 'River WA', 16), ('usgs_q_f', 'River PA', 17)]
CONDS = [(0, 'all inputs', 'o', '#08306b'), (3, '3 dropped', 's', '#2171b5'), (6, '6 dropped', '^', '#6baed6')]
NETCOL = {'Pressure': '#d62728', 'Temperature': '#ff7f0e', 'PM$_{2.5}$': '#2ca02c', 'NO$_2$': '#8c564b',
          'River NY': '#1f77b4', 'River WA': '#17becf', 'River PA': '#9467bd'}


def t_interval(v):
    """95% t-interval over clusters, from a confirm entry (gain, normal-theory interval at its level, n clusters)."""
    se = (v['hi'] - v['lo']) / (2 * Z[v.get('level', 0.95)])
    h = stats.t.ppf(.975, v['n'] - 1) * se
    return v['gain'] - h, v['gain'] + h


def robustness(ax, c12, c13, c15):
    """Panel (a): mean NLL over the four v10 networks relative to TabPFN v2 with all inputs (as make_paper_pool did)."""
    dd, d13, d15 = c12['descriptive_defined_episodes'], c13['descriptive_tabpfn35_defined_episodes'], c15['descriptive']
    nets = [n for n, _ in OLD]
    series = [('TabPFN v2', 'C3', 'o-', 1.0, lambda n, e: dd[f'{n}|e{e}']['tabpfn']),
              ('FA $\\oplus$ TabPFN v2', 'C4', 'o-', 1.0, lambda n, e: dd[f'{n}|e{e}']['fa']),
              ('PFN-L $\\oplus$ TabPFN v2', 'C2', 'o-', 1.0, lambda n, e: dd[f'{n}|e{e}']['pfnL']),
              ('UPT-SF $\\oplus$ TabPFN v2', 'C0', 'o-', 1.8, lambda n, e: dd[f'{n}|e{e}']['uptsf']),
              ('TabPFN-3.5', 'C1', 's--', 1.0, lambda n, e: d13[f'{n}|e{e}']['tabpfn35']['nll']),
              ('UPT-SF $\\oplus$ TabPFN-3.5', 'C9', 's--', 1.8, lambda n, e: d15[f'{n}|e{e}']['selected'])]
    for lab, col, sty, lw, get in series:
        ax.plot([0, 3, 6], [np.mean([get(n, e) - dd[f'{n}|e0']['tabpfn'] for n in nets]) for e in (0, 3, 6)], sty,
                color=col, lw=lw, label=lab, ms=3)
    ax.axhline(0, color='.6', lw=.5, zorder=0)
    ax.set_ylim(top=1.32)                           # room for the legend above the curves
    ax.set_xticks([0, 3, 6]); ax.set_xlabel('additional inputs dropped')
    ax.set_ylabel('NLL $-$ NLL(TabPFN v2, all inputs)')
    ax.set_title('(a) Four networks recorded in 2026', loc='left')
    ax.legend(loc='upper left', frameon=False, fontsize=6, handlelength=2.2, borderaxespad=.2)
    ax.grid(alpha=.25, lw=.4)


def forest(ax, c16, c17):
    """Panel (b): gain of the frozen v15 pool over TabPFN-3.5 on every fresh network, and pooled (pre-registered)."""
    pooled = [c17[k] for k in ('G1_pool_vs_tabpfn35_nll_e0_all7', 'G2_pool_vs_tabpfn35_nll_e3_all7',
                               'G2_pool_vs_tabpfn35_nll_e6_all7')]
    ys, labels = [0.0], ['all seven networks$^{*}$']
    for j, ((e, _, mk, col), v) in enumerate(zip(CONDS, pooled)):
        y = (j - 1) * .26
        ax.plot([v['lo'], v['hi']], [y, y], color=col, lw=1.1)
        ax.plot(v['gain'], y, mk, color=col, mfc=col if v['passed'] else 'white', mew=.9, ms=3.6)
    for i, (net, lab, proto) in enumerate(FRESH):
        src = (c16 if proto == 16 else c17)['secondary']
        yc = 1.45 + i
        ys.append(yc); labels.append(lab)
        for j, (e, _, mk, col) in enumerate(CONDS):
            v = src[f'{net}|pool_vs_tabpfn35_nll_e{e}']
            lo, hi = t_interval(v)
            y = yc + (j - 1) * .26
            ax.plot([lo, hi], [y, y], color=col, lw=.9)
            ax.plot(v['gain'], y, mk, color=col, ms=3)
    ax.axhline(.72, color='.75', lw=.5)
    ax.axvline(0, color='.4', lw=.6, ls='--', zorder=0)
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('NLL gain of the pool over TabPFN-3.5 (nats)')
    ax.set_title("(b) Tasks recorded after the release of TabPFN-3.5", loc='left')
    ax.legend(handles=[Line2D([], [], marker=mk, color=col, lw=.9, ms=3, label=lab) for _, lab, mk, col in CONDS],
              loc='lower right', frameon=False, fontsize=6, handlelength=1.6)
    ax.grid(axis='x', alpha=.25, lw=.4)


def crossfit(axs, c15, c17):
    """Two-fold (v13 rule) vs four-fold (v15 rule) stacking: the pool's gain over TabPFN-3.5, per network x condition."""
    panels = [('(a) Original test periods, four networks', OLD, c15['descriptive'], 'pool13', 'selected'),
              ('(b) Tasks recorded after the release, seven networks', [(n, lab) for n, lab, _ in FRESH], c17['descriptive'],
               'pool_v13rule', 'pool_P3')]
    for ax, (title, nets, d, two, four) in zip(axs, panels):
        ys, labels, y = [], [], 0
        for n, lab in nets:
            col = NETCOL[lab]
            for e, short in ((0, 'all'), (3, '$-3$'), (6, '$-6$')):
                r = d[f'{n}|e{e}']
                g2, g4 = r['tabpfn35'] - r[two], r['tabpfn35'] - r[four]
                ax.annotate('', xy=(g4, y), xytext=(g2, y), arrowprops=dict(arrowstyle='-|>', lw=.8, color=col,
                                                                            mutation_scale=6, shrinkA=1.5, shrinkB=1.5))
                ax.plot(g2, y, 'o', mfc='white', mec=col, mew=.8, ms=3.2)
                ax.plot(g4, y, 'o', color=col, ms=3.2)
                ys.append(y); labels.append(f'{lab}, {short}'); y += 1
            y += .6
        ax.axvline(0, color='.4', lw=.6, ls='--', zorder=0)
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6.3); ax.invert_yaxis()
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('NLL gain of the pool over TabPFN-3.5 (nats)')
        ax.set_title(title, loc='left')
        ax.grid(axis='x', alpha=.25, lw=.4)
    axs[1].legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='.3', ms=3.2, label='twofold cross fit'),
                           Line2D([], [], marker='o', ls='', color='.3', ms=3.2, label='fourfold cross fit')],
                  loc='lower right', frameon=False, fontsize=6)


def weights(ax, c13, c15, c17):
    """Mean stacking weight on UPT-SF against NLL(UPT-SF) - NLL(TabPFN-3.5), per network x condition."""
    d13, d15, d17 = c13['descriptive_tabpfn35_defined_episodes'], c15['descriptive'], c17['descriptive']
    for n, lab in OLD:
        x = [d13[f'{n}|e{e}']['upt_sf']['nll'] - d13[f'{n}|e{e}']['tabpfn35']['nll'] for e in (0, 3, 6)]
        ax.plot(x, [d15[f'{n}|e{e}']['w_selected'] for e in (0, 3, 6)], 'o', mfc='white', mec=NETCOL[lab], mew=.8, ms=3.6)
    for n, lab, _ in FRESH:
        x = [d17[f'{n}|e{e}']['uptsf'] - d17[f'{n}|e{e}']['tabpfn35'] for e in (0, 3, 6)]
        ax.plot(x, [d17[f'{n}|e{e}']['w_P3'] for e in (0, 3, 6)], 'o', color=NETCOL[lab], ms=3.6)
    ax.set_xscale('symlog', linthresh=.1, linscale=.8)
    ax.set_xticks([-.1, 0, .1, 1, 3]); ax.set_xticklabels(['$-0.1$', '0', '0.1', '1', '3'])
    ax.axvline(0, color='.4', lw=.6, ls='--', zorder=0)
    ax.set_ylim(0, 1); ax.set_xlim(-.2, 4)
    ax.set_xlabel('NLL of UPT-SF alone minus NLL of TabPFN-3.5')
    ax.set_ylabel('mean stacking weight on UPT-SF')
    hs = [Line2D([], [], marker='o', ls='', color=c, ms=3.4, label=lab) for lab, c in NETCOL.items()]
    hs += [Line2D([], [], marker='o', ls='', color='.3', ms=3.4, label='after release (v16, v17)'),
           Line2D([], [], marker='o', ls='', mfc='white', mec='.3', ms=3.4, label='original periods (v13, v15)')]
    ax.legend(handles=hs, loc='upper right', frameon=False, fontsize=5.4, ncol=2, columnspacing=.5, handletextpad=.2)
    ax.grid(alpha=.25, lw=.4)


def curves(fig, c17):
    """Mean NLL by inputs dropped on each fresh network: TabPFN-3.5, UPT-SF alone and the frozen v15 pool."""
    d = c17['descriptive']
    axs = fig.subplots(2, 4).ravel()
    lines = [('TabPFN-3.5', 'tabpfn35', 'C1', 's--', 1.0), ('UPT-SF alone', 'uptsf', '.45', 'd:', 1.0),
             ('UPT-SF $\\oplus$ TabPFN-3.5 (v15 rule)', 'pool_P3', 'C9', 'o-', 1.6)]
    for i, (ax, (n, lab, _)) in enumerate(zip(axs, FRESH)):
        for nm, key, col, sty, lw in lines:
            ax.plot([0, 3, 6], [d[f'{n}|e{e}'][key] for e in (0, 3, 6)], sty, color=col, lw=lw, ms=3, label=nm)
        ax.set_title(lab, loc='left'); ax.set_xticks([0, 3, 6]); ax.grid(alpha=.25, lw=.4)
        if i % 4 == 0:
            ax.set_ylabel('mean NLL')
        if i >= 3:
            ax.set_xlabel('inputs dropped')
    axs[-1].axis('off')
    axs[-1].legend(*axs[0].get_legend_handles_labels(), loc='center', frameon=False, fontsize=6.5)


def main():
    ap = argparse.ArgumentParser(); here = Path(__file__).resolve().parents[1]
    ap.add_argument('--runs', default=str(here / 'data')); ap.add_argument('--out', default=str(here))
    a = ap.parse_args()
    R, F = Path(a.runs), Path(a.out) / 'figures'; F.mkdir(parents=True, exist_ok=True)
    c = {v: json.loads((R / f'confirm_v{v}.json').read_text()) for v in (12, 13, 15, 16, 17)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw={'width_ratios': [1, 1.08]})
    robustness(a1, c[12], c[13], c[15]); forest(a2, c[16], c[17])
    fig.tight_layout(w_pad=1.2); fig.savefig(F / 'pool_fresh.pdf'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(W, 4.0), gridspec_kw={'width_ratios': [1, 1]})
    crossfit(axs, c[15], c[17])
    fig.tight_layout(w_pad=1.5); fig.savefig(F / 'crossfit.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    weights(ax, c[13], c[15], c[17])
    fig.tight_layout(); fig.savefig(F / 'weights.pdf'); plt.close(fig)

    fig = plt.figure(figsize=(W, 3.1))
    curves(fig, c[17])
    fig.tight_layout(h_pad=.8, w_pad=.6); fig.savefig(F / 'fresh_curves.pdf'); plt.close(fig)
    print('wrote', ', '.join(p.name for p in sorted(F.glob('*.pdf'))))


if __name__ == '__main__':
    main()
