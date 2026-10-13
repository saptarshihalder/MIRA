"""Pool tables, numbers and the robustness figure of paper/upt from confirm_v11.json and confirm_v12.json.

    python make_paper_pool.py --runs artifacts/reports/lifted_cavity_paper/runs --out paper/upt
"""
import argparse, json
from pathlib import Path

NETS = [('usgs_ny', 'USGS NY (rivers)'), ('usgs_wa', 'USGS WA (rivers)'), ('dwd_p', 'DWD pressure'), ('aurn_pm25', 'AURN PM$_{2.5}$')]
PARTNERS = [('uptsf', 'UPT-SF'), ('pfnL', 'PFN-L'), ('fa', 'FA-Gauss.')]


def ci(v, d=2):
    return f"{v['gain']:+.{d}f} [{v['lo']:+.{d}f}, {v['hi']:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R, O = Path(a.runs), Path(a.out); (O / 'generated').mkdir(parents=True, exist_ok=True); (O / 'figures').mkdir(exist_ok=True)
    c11 = json.loads((R / 'confirm_v11.json').read_text()); c12 = json.loads((R / 'confirm_v12.json').read_text())
    c10 = json.loads((R / 'confirm_v10.json').read_text())
    dd = c12['descriptive_defined_episodes']

    # ---- table: mean NLL of each pool and TabPFN, per network and condition (TabPFN-defined episodes)
    t = [r'\begin{table}[t]\centering\small',
         r'\caption{\textbf{Pooling TabPFN v2 with a partner on four fresh 2026 networks} (mean test NLL, lower is better, on the '
         r'episodes where TabPFN is defined). In parentheses: the mean stacking weight on the partner, chosen per episode by '
         r'cross-fitting the episode\textquotesingle s own labelled rows. Every pool uses the same rule; best per row in bold.}',
         r'\label{tab:pool}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{llcccc}', r'\toprule',
         r'Network & Inputs & UPT-SF$\,\oplus\,$TabPFN & PFN-L$\,\oplus\,$TabPFN & FA$\,\oplus\,$TabPFN & TabPFN v2 \\', r'\midrule']
    for n, lab in NETS:
        for e, cond in ((0, 'all'), (3, '$-3$'), (6, '$-6$')):
            r = dd[f'{n}|e{e}']
            vals = {m: r[m] for m, _ in PARTNERS}; vals['tabpfn'] = r['tabpfn']
            best = min(vals.values())
            cells = []
            for m in ('uptsf', 'pfnL', 'fa', 'tabpfn'):
                s = f'{vals[m]:+.3f}'
                s = r'\textbf{' + s + '}' if vals[m] == best else s
                cells.append(s + (f" ({r[m + '_w']:.2f})" if m != 'tabpfn' else ''))
            t.append(f"{lab if e == 0 else ''} & {cond} & " + ' & '.join(cells) + r' \\')
        if n != NETS[-1][0]:
            t.append(r'\addlinespace')
    t += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (O / 'generated' / 'pool_table.tex').write_text('\n'.join(t))

    # ---- table: the pre-registered fresh-data endpoints (v10 W, v11 X, v12 Y)
    rows = [('v11', 'UPT-SF$\\oplus$TabPFN vs TabPFN v2', c11, ('X1_pool_vs_tabpfn_nll_e0', 'X2_pool_vs_tabpfn_nll_e3', 'X2_pool_vs_tabpfn_nll_e6')),
            ('v11', 'UPT-SF$\\oplus$TabPFN vs UPT-SF alone', c11, ('X3_pool_vs_uptsf_nll_e0', None, None)),
            ('v12', 'UPT-SF$\\oplus$TabPFN vs PFN-L$\\oplus$TabPFN', c12,
             ('Y1_uptsfpool_vs_pfnLpool_nll_e0', 'Y1_uptsfpool_vs_pfnLpool_nll_e3', 'Y1_uptsfpool_vs_pfnLpool_nll_e6')),
            ('v12', 'UPT-SF$\\oplus$TabPFN vs FA$\\oplus$TabPFN', c12,
             ('Y2_uptsfpool_vs_fapool_nll_e0', 'Y2_uptsfpool_vs_fapool_nll_e3', 'Y2_uptsfpool_vs_fapool_nll_e6')),
            ('v10', 'UPT-SF alone vs TabPFN v2', c10, ('W1_sf_vs_tabpfn_nll_e0', 'W2_sf_vs_tabpfn_nll_e3', 'W2_sf_vs_tabpfn_nll_e6'))]
    t = [r'\begin{table}[t]\centering\small',
         r'\caption{\textbf{Pre-registered endpoints on the four fresh 2026 networks.} NLL gain in nats (baseline $-$ ours; '
         r'positive favours the first method), paired 95\% interval clustered by network-window; \checkmark: pass '
         r'(gain $\geq 0.01$ and lower bound $>0$). Protocol v12 tests the mechanism on the same networks after v11.}',
         r'\label{tab:fresh}', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llccc}', r'\toprule',
         r'Protocol & Comparison & All inputs & 3 dropped & 6 dropped \\', r'\midrule']
    for proto, lab, d, ks in rows:
        cells = []
        for k in ks:
            if k is None:
                cells.append('--'); continue
            v = d[k]
            cells.append(ci(v) + (r' \checkmark' if v['passed'] else r' $\times$'))
        t.append(f'{proto} & {lab} & ' + ' & '.join(cells) + r' \\')
    t += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (O / 'generated' / 'fresh_endpoints_table.tex').write_text('\n'.join(t))

    # ---- figure: NLL relative to TabPFN with all inputs, averaged over the four networks, by inputs dropped
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for m, lab, col in (('tabpfn', 'TabPFN v2', 'C3'), ('fa', 'FA $\\oplus$ TabPFN', 'C4'), ('pfnL', 'PFN-L $\\oplus$ TabPFN', 'C2'),
                        ('uptsf', 'UPT-SF $\\oplus$ TabPFN', 'C0')):
        ys = [sum(dd[f'{n}|e{e}'][m] - dd[f'{n}|e0']['tabpfn'] for n, _ in NETS) / len(NETS) for e in (0, 3, 6)]
        ax.plot([0, 3, 6], ys, 'o-', label=lab, color=col, lw=2 if m == 'uptsf' else 1.2)
    ax.set_xticks([0, 3, 6]); ax.set_xlabel('additional input sensors dropped'); ax.set_ylabel('NLL $-$ NLL(TabPFN, all inputs)')
    ax.grid(alpha=.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(O / 'figures' / 'pool_robustness.pdf'); plt.close(fig)

    # ---- numbers
    num = [r'\newcommand{\XOne}{' + ci(c11['X1_pool_vs_tabpfn_nll_e0']) + '}',
           r'\newcommand{\XTwoThree}{' + ci(c11['X2_pool_vs_tabpfn_nll_e3']) + '}',
           r'\newcommand{\XTwoSix}{' + ci(c11['X2_pool_vs_tabpfn_nll_e6']) + '}',
           r'\newcommand{\XThree}{' + ci(c11['X3_pool_vs_uptsf_nll_e0']) + '}',
           r'\newcommand{\YOneZero}{' + ci(c12['Y1_uptsfpool_vs_pfnLpool_nll_e0']) + '}',
           r'\newcommand{\YOneSix}{' + ci(c12['Y1_uptsfpool_vs_pfnLpool_nll_e6']) + '}',
           r'\newcommand{\YTwoZero}{' + ci(c12['Y2_uptsfpool_vs_fapool_nll_e0']) + '}',
           r'\newcommand{\YTwoSix}{' + ci(c12['Y2_uptsfpool_vs_fapool_nll_e6']) + '}',
           r'\newcommand{\FreshEpisodes}{' + f"{c11['X3_pool_vs_uptsf_nll_e0']['episodes']:,}".replace(',', '{,}') + '}']
    (O / 'generated' / 'numbers_pool.tex').write_text('\n'.join(num) + '\n')
    print('\n'.join(num))


if __name__ == '__main__':
    main()
