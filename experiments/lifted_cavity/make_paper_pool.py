"""Pool tables, numbers and the robustness figure of paper/upt from confirm_v11.json, confirm_v12.json and (when it
exists) confirm_v13.json (TabPFN-3.5).

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
    p13 = R / 'confirm_v13.json'
    c13 = json.loads(p13.read_text()) if p13.exists() else None
    p14 = R / 'confirm_v14.json'
    c14 = json.loads(p14.read_text()) if p14.exists() else None
    V14 = {'uptsite': 'UPT site$\\,\\otimes\\,$TabPFN-3.5', 'uptcol': 'TabPFN-3.5 + UPT column', 'uptcol2': 'TabPFN-3.5 + UPT columns',
           'uptres': 'TabPFN-3.5 on UPT residual'}
    p15 = R / 'confirm_v15.json'
    c15 = json.loads(p15.read_text()) if p15.exists() else None
    p16 = R / 'confirm_v16.json'
    c16 = json.loads(p16.read_text()) if p16.exists() else None

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
    if c13:
        rows[2:2] = [('v13', 'UPT-SF$\oplus$TabPFN-3.5 vs TabPFN-3.5', c13, ('Z1_pool_vs_tabpfn35_nll_e0', 'Z2_pool_vs_tabpfn35_nll_e3', 'Z2_pool_vs_tabpfn35_nll_e6')),
                     ('v13', 'UPT-SF$\oplus$TabPFN-3.5 vs UPT-SF alone', c13, ('Z3_pool_vs_uptsf_nll_e0', None, None))]
    if c15:                                         # protocol v15: learning-curve-aware stacking (V1, V2), right after v13
        rows[4:4] = [('v15', 'UPT-SF$\\oplus$TabPFN-3.5, 4-fold stacking, vs TabPFN-3.5', c15,
                      ('V1_sel_vs_tabpfn35_nll_e0', 'V2_sel_vs_tabpfn35_nll_e3', 'V2_sel_vs_tabpfn35_nll_e6'))]
    if c14:                                         # protocol v14: the selected variant (R1, R2) and, secondary, vs the v13 pool
        lab = V14[c14['selected']['selected']]
        c14s = dict(c14, **c14['secondary'])
        at = 4 + (1 if c15 else 0)
        rows[at:at] = [('v14', f'{lab} vs TabPFN-3.5', c14s, ('R1_uptcol_vs_tabpfn35_nll_e0', 'R2_uptcol_vs_tabpfn35_nll_e3', 'R2_uptcol_vs_tabpfn35_nll_e6')),
                     ('v14$^\\dagger$', f'{lab} vs UPT-SF$\\oplus$TabPFN-3.5', c14s,
                      ('uptcol_vs_pool35_nll_e0', 'uptcol_vs_pool35_nll_e3', 'uptcol_vs_pool35_nll_e6'))]
    if c16:                                         # protocol v16: the frozen v15 rule on fresh tasks after TabPFN-3.5's release
        rows[0:0] = [('v16', 'UPT-SF$\\oplus$TabPFN-3.5 vs TabPFN-3.5, fresh tasks', c16,
                      ('F1_pool_vs_tabpfn35_nll_e0', 'F2_pool_vs_tabpfn35_nll_e3', 'F2_pool_vs_tabpfn35_nll_e6'))]
    t = [r'\begin{table}[t]\centering\small',
         r'\caption{\textbf{Pre-registered endpoints on the four fresh 2026 networks.} NLL gain in nats (baseline $-$ ours; '
         r'positive favours the first method), paired 95\% interval clustered by network-window; \checkmark: pass '
         r'(gain $\geq 0.01$ and lower bound $>0$). Protocols v12 and v13 (TabPFN-3.5, September 2026) test the mechanism '
         r'and a newer foundation model on the same networks after v11' + (r'; v14 tests the variant selected on development '
         r'networks, and $^\dagger$ marks a secondary comparison.}' if c14 else '.}'),
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
    if c13:                                         # TabPFN-3.5 and its pool, on the same reference (TabPFN v2, all inputs)
        d13 = c13['descriptive_tabpfn35_defined_episodes']
        get = {'tabpfn35': lambda n, e: d13[f'{n}|e{e}']['tabpfn35']['nll'], 'pool35': lambda n, e: d13[f'{n}|e{e}']['pool35']['nll']}
        lines = [('tabpfn35', 'TabPFN-3.5', 'C1')]
        if c15:                                     # protocol v15: the learning-curve-aware pool replaces the v13 pool
            get['pool15'] = lambda n, e: c15['descriptive'][f'{n}|e{e}']['selected']
            lines.append(('pool15', 'UPT-SF $\\oplus$ TabPFN-3.5 (v15)', 'C9'))
        else:
            lines.append(('pool35', 'UPT-SF $\\oplus$ TabPFN-3.5', 'C9'))
        for m, lab, col in lines:
            ys = [sum(get[m](n, e) - dd[f'{n}|e0']['tabpfn'] for n, _ in NETS) / len(NETS) for e in (0, 3, 6)]
            ax.plot([0, 3, 6], ys, 's--', label=lab, color=col, lw=1.2 if m == 'tabpfn35' else 2)
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
    if c13:
        num += [r'\newcommand{\ZOne}{' + ci(c13['Z1_pool_vs_tabpfn35_nll_e0']) + '}',
                r'\newcommand{\ZTwoThree}{' + ci(c13['Z2_pool_vs_tabpfn35_nll_e3']) + '}',
                r'\newcommand{\ZTwoSix}{' + ci(c13['Z2_pool_vs_tabpfn35_nll_e6']) + '}',
                r'\newcommand{\ZThree}{' + ci(c13['Z3_pool_vs_uptsf_nll_e0']) + '}']
    if c15:
        v3 = c15['V3_noninferior_every_network']['tests']
        num += [r'\newcommand{\VOne}{' + ci(c15['V1_sel_vs_tabpfn35_nll_e0']) + '}',
                r'\newcommand{\VTwoThree}{' + ci(c15['V2_sel_vs_tabpfn35_nll_e3']) + '}',
                r'\newcommand{\VTwoSix}{' + ci(c15['V2_sel_vs_tabpfn35_nll_e6']) + '}',
                r'\newcommand{\VThreePass}{' + f"{sum(v['passed'] for v in v3.values())}" + '}',
                r'\newcommand{\VThreeN}{' + f'{len(v3)}' + '}',
                r'\newcommand{\VPoolZero}{' + ci(c15['secondary']['sel_vs_pool13_nll_e0'], 3) + '}']
    if c16:
        s16 = c16['secondary']
        num += [r'\newcommand{\FOne}{' + ci(c16['F1_pool_vs_tabpfn35_nll_e0']) + '}',
                r'\newcommand{\FTwoThree}{' + ci(c16['F2_pool_vs_tabpfn35_nll_e3']) + '}',
                r'\newcommand{\FTwoSix}{' + ci(c16['F2_pool_vs_tabpfn35_nll_e6']) + '}',
                r'\newcommand{\FEpisodes}{' + f"{c16['F1_pool_vs_tabpfn35_nll_e0']['episodes']:,}".replace(',', '{,}') + '}',
                r'\newcommand{\FUptZero}{' + ci(s16['pool_vs_uptsf_nll_e0']) + '}']
    if c14:
        num += [r'\newcommand{\ROne}{' + ci(c14['R1_uptcol_vs_tabpfn35_nll_e0']) + '}',
                r'\newcommand{\RTwoThree}{' + ci(c14['R2_uptcol_vs_tabpfn35_nll_e3']) + '}',
                r'\newcommand{\RTwoSix}{' + ci(c14['R2_uptcol_vs_tabpfn35_nll_e6']) + '}',
                r'\newcommand{\RPoolZero}{' + ci(c14['secondary']['uptcol_vs_pool35_nll_e0']) + '}',
                r'\newcommand{\RPoolSix}{' + ci(c14['secondary']['uptcol_vs_pool35_nll_e6']) + '}']
    (O / 'generated' / 'numbers_pool.tex').write_text('\n'.join(num) + '\n')
    print('\n'.join(num))


if __name__ == '__main__':
    main()
