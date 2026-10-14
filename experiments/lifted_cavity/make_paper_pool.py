"""Pool tables and numbers of paper/upt from confirm_v11.json, confirm_v12.json and (when they exist) confirm_v13-v17.json
(TabPFN-3.5). The figures are drawn by make_paper_figs.py.

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
    V14S = {'uptsite': 'log pool', 'uptcol': 'UPT column', 'uptcol2': 'UPT columns', 'uptres': 'UPT residual'}
    p15 = R / 'confirm_v15.json'
    c15 = json.loads(p15.read_text()) if p15.exists() else None
    p16 = R / 'confirm_v16.json'
    c16 = json.loads(p16.read_text()) if p16.exists() else None
    p17 = R / 'confirm_v17.json'
    c17 = json.loads(p17.read_text()) if p17.exists() else None

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
    rows = [('v11', 'pool vs TabPFN v2', c11, ('X1_pool_vs_tabpfn_nll_e0', 'X2_pool_vs_tabpfn_nll_e3', 'X2_pool_vs_tabpfn_nll_e6')),
            ('v11', 'pool vs UPT-SF alone', c11, ('X3_pool_vs_uptsf_nll_e0', None, None)),
            ('v12', 'pool vs PFN-L$\\oplus$TabPFN', c12,
             ('Y1_uptsfpool_vs_pfnLpool_nll_e0', 'Y1_uptsfpool_vs_pfnLpool_nll_e3', 'Y1_uptsfpool_vs_pfnLpool_nll_e6')),
            ('v12', 'pool vs FA$\\oplus$TabPFN', c12,
             ('Y2_uptsfpool_vs_fapool_nll_e0', 'Y2_uptsfpool_vs_fapool_nll_e3', 'Y2_uptsfpool_vs_fapool_nll_e6')),
            ('v10', 'UPT-SF alone vs TabPFN v2', c10, ('W1_sf_vs_tabpfn_nll_e0', 'W2_sf_vs_tabpfn_nll_e3', 'W2_sf_vs_tabpfn_nll_e6'))]
    if c13:
        rows[2:2] = [('v13', 'pool vs TabPFN-3.5', c13, ('Z1_pool_vs_tabpfn35_nll_e0', 'Z2_pool_vs_tabpfn35_nll_e3', 'Z2_pool_vs_tabpfn35_nll_e6')),
                     ('v13', 'pool vs UPT-SF alone', c13, ('Z3_pool_vs_uptsf_nll_e0', None, None))]
    if c15:                                         # protocol v15: learning-curve-aware stacking (V1, V2), right after v13
        rows[4:4] = [('v15', '4-fold pool vs TabPFN-3.5', c15,
                      ('V1_sel_vs_tabpfn35_nll_e0', 'V2_sel_vs_tabpfn35_nll_e3', 'V2_sel_vs_tabpfn35_nll_e6'))]
    if c14:                                         # protocol v14: the selected variant (R1, R2) and, secondary, vs the v13 pool
        lab = V14S[c14['selected']['selected']]
        c14s = dict(c14, **c14['secondary'])
        at = 4 + (1 if c15 else 0)
        rows[at:at] = [('v14', f'{lab} vs TabPFN-3.5', c14s, ('R1_uptcol_vs_tabpfn35_nll_e0', 'R2_uptcol_vs_tabpfn35_nll_e3', 'R2_uptcol_vs_tabpfn35_nll_e6')),
                     ('v14$^\\dagger$', f'{lab} vs pool', c14s,
                      ('uptcol_vs_pool35_nll_e0', 'uptcol_vs_pool35_nll_e3', 'uptcol_vs_pool35_nll_e6'))]
    if c17:                                         # protocol v17: all fresh post-release tasks, seven networks, 97.5%
        rows[0:0] = [('v16+17', 'the same, seven networks', c17,
                      ('G1_pool_vs_tabpfn35_nll_e0_all7', 'G2_pool_vs_tabpfn35_nll_e3_all7', 'G2_pool_vs_tabpfn35_nll_e6_all7'))]
    if c16:                                         # protocol v16: the frozen v15 rule on fresh tasks after TabPFN-3.5's release
        rows[0:0] = [('v16', '4-fold pool vs TabPFN-3.5, new tasks', c16,
                      ('F1_pool_vs_tabpfn35_nll_e0', 'F2_pool_vs_tabpfn35_nll_e3', 'F2_pool_vs_tabpfn35_nll_e6'))]
    t = [r'\begin{table}[t]\centering\small',
         r'\caption{\textbf{Pre-registered endpoints on the fresh 2026 networks.} NLL gain in nats (baseline $-$ ours; '
         r'positive favours the first method) $\pm$ the half-width of its paired 95\% interval, clustered by '
         r'network-window; \checkmark: pass (gain $\geq 0.01$ and lower bound $>0$). Pool: UPT-SF$\oplus$TabPFN v2 (v11, '
         r'v12) or TabPFN-3.5 (September 2026; v13 on). Protocols v11--v15 use the four v10 networks' +
         (r'; v14 tests the log pool selected on development networks ($^\dagger$: secondary)' if c14 else '') +
         (r". v16 scores the frozen v15 rule on new tasks recorded after TabPFN-3.5's release; v16+17 adds the three v9 "
          r'networks and, as a second look, uses 97.5\% intervals.}' if c17 else '.}'),
         r'\label{tab:fresh}', r'\setlength{\tabcolsep}{3pt}', r'\begin{adjustbox}{max width=\linewidth}',
         r'\begin{tabular}{llccc}', r'\toprule', r'Protocol & Comparison & All inputs & 3 dropped & 6 dropped \\', r'\midrule']
    for proto, lab, d, ks in rows:
        cells = []
        for k in ks:
            if k is None:
                cells.append('--'); continue
            v, dg = d[k], 3 if proto in ('v16', 'v16+17') else 2
            cells.append(f"${v['gain']:+.{dg}f}\\pm{(v['hi'] - v['lo']) / 2:.{dg}f}$" + (r' \checkmark' if v['passed'] else r' $\times$'))
        t.append(f'{proto} & {lab} & ' + ' & '.join(cells) + r' \\')
    t += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    (O / 'generated' / 'fresh_endpoints_table.tex').write_text('\n'.join(t))

    # (the figures, including the robustness curves, are drawn by make_paper_figs.py)

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
        num += [r'\newcommand{\FOne}{' + ci(c16['F1_pool_vs_tabpfn35_nll_e0'], 3) + '}',
                r'\newcommand{\FTwoThree}{' + ci(c16['F2_pool_vs_tabpfn35_nll_e3'], 3) + '}',
                r'\newcommand{\FTwoSix}{' + ci(c16['F2_pool_vs_tabpfn35_nll_e6'], 3) + '}',
                r'\newcommand{\FEpisodes}{' + f"{c16['F1_pool_vs_tabpfn35_nll_e0']['episodes']:,}".replace(',', '{,}') + '}',
                r'\newcommand{\FUptZero}{' + ci(s16['pool_vs_uptsf_nll_e0']) + '}']
    if c17:
        g = c17['G1_pool_vs_tabpfn35_nll_e0_all7']
        num += [r'\newcommand{\GOne}{' + ci(g, 3) + '}',
                r'\newcommand{\GTwoThree}{' + ci(c17['G2_pool_vs_tabpfn35_nll_e3_all7'], 3) + '}',
                r'\newcommand{\GTwoSix}{' + ci(c17['G2_pool_vs_tabpfn35_nll_e6_all7'], 3) + '}',
                r'\newcommand{\GEpisodes}{' + f"{g['episodes']:,}".replace(',', '{,}') + '}',
                r'\newcommand{\GClusters}{' + f"{g['n']}" + '}']
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
