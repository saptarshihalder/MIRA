"""Tables, numbers and the robustness figure of paper/upt from the committed confirm_v*.json files (no hand-copied numbers).

    python make_paper_upt.py --runs artifacts/reports/lifted_cavity_paper/runs --out paper/upt [--counts counts.json]
--counts: optional {network: [episodes, windows, sensors]} measured on the panels (colab cell in the README of paper/upt).
"""
import argparse, json
from pathlib import Path
import numpy as np

NETS = [  # key, label, domain, source of data, test period
    ('metr_la', 'METR-LA', 'traffic speed', 'Los Angeles loop detectors~\\citep{li2018dcrnn}', '2012'),
    ('pems_bay', 'PEMS-BAY', 'traffic speed', 'Bay Area loop detectors~\\citep{li2018dcrnn}', '2017'),
    ('intel', 'Intel Lab', 'indoor temperature', 'Intel Berkeley lab motes~\\citep{intellab2004}', '2004'),
    ('pems03', 'PeMS03', 'traffic flow', 'Sacramento detectors~\\citep{song2020stsgcn}', '2018'),
    ('pems08', 'PeMS08', 'traffic flow', 'San Bernardino detectors~\\citep{guo2019astgcn}', '2016'),
    ('air_cn', 'China air', 'PM$_{2.5}$', '400 stations outside Beijing~\\citep{zheng2015forecasting}', '2015'),
    ('dwd_t2m', 'DWD temp.', 'air temperature', 'German weather service stations~\\citep{dwd2026}', '2026'),
    ('aurn_no2', 'AURN NO$_2$', 'NO$_2$', 'UK air-quality network~\\citep{carslaw2012openair}', '2026'),
    ('usgs_q', 'USGS PA', 'river discharge', 'Pennsylvania stream gauges~\\citep{usgs2026nwis}', '2026'),
    ('usgs_ny', 'USGS NY', 'river discharge', 'New York stream gauges~\\citep{usgs2026nwis}', '2026'),
    ('usgs_wa', 'USGS WA', 'river discharge', 'Washington stream gauges~\\citep{usgs2026nwis}', '2026'),
    ('dwd_p', 'DWD pressure', 'air pressure', 'German weather service stations~\\citep{dwd2026}', '2026'),
    ('aurn_pm25', 'AURN PM$_{2.5}$', 'PM$_{2.5}$', 'UK air-quality network~\\citep{carslaw2012openair}', '2026'),
]
PROTO = {'metr_la': 'v6', 'pems_bay': 'v6', 'intel': 'v6', 'pems03': 'v7', 'pems08': 'v7', 'air_cn': 'v7',
         'dwd_t2m': 'v9', 'aurn_no2': 'v9', 'usgs_q': 'v9', 'usgs_ny': 'v10', 'usgs_wa': 'v10', 'dwd_p': 'v10', 'aurn_pm25': 'v10'}
FAIL = '\\textbf{fail}'
COLS = [('sf', 'UPT-SF'), ('pfn_L', 'PFN-L'), ('tabpfn_v2', 'TabPFN v2'), ('tnp', 'TNP-D'), ('ignnk_cal', 'Krig.\\ (cal.)'),
        ('fa1', 'FA-Gauss.')]


def f3(x):
    return '--' if x is None else f'{x:+.2f}' if abs(x) < 10 else f'{x:+.1f}'


def row_val(desc, net, e, m):
    r = desc.get(f'{net}|e{e}', {}).get(m)
    return None if not r else r.get('nll')


def endpoint_rows(name, d, prefix_map):
    rows = []
    for k, v in d.items():
        if not isinstance(v, dict) or 'gain' not in v:
            continue
        lab = prefix_map(k)
        if lab:
            rows.append((name, lab, v))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--counts')
    a = ap.parse_args()
    R, O = Path(a.runs), Path(a.out); (O / 'generated').mkdir(parents=True, exist_ok=True)
    c = {v: json.loads((R / f'confirm_v{v}.json').read_text()) for v in (6, 7, 8, 9, 10) if (R / f'confirm_v{v}.json').exists()}
    d10 = c[10]
    desc = {**d10['descriptive_old'], **d10['descriptive_v10']}

    # ---- main table: NLL by network at all sensors and at +6 dropped
    lines = ['\\begin{table}[t]\\centering\\small', '\\caption{\\textbf{Mean test NLL (nats; lower is better)} with all sensors '
             '(left of the slash) and with six extra sensors dropped (right). Learned models: mean over three seeds. '
             'Best per row in bold. Networks recorded in 2026 are below the rule; the last four are protocol v10, never used '
             'before its endpoints were scored. On USGS NY the UPT-SF mean is dominated by 15 episodes with a constant '
             'support target (\\S\\ref{sec:results}).}', '\\label{tab:main}', '\\setlength{\\tabcolsep}{3pt}',
             '\\begin{tabular}{l' + 'c' * len(COLS) + '}', '\\toprule', 'Network & ' + ' & '.join(l for _, l in COLS) + ' \\\\', '\\midrule']
    for key, lab, *_ in NETS:
        if key == 'dwd_t2m':
            lines.append('\\midrule')
        cells = []
        for e in (0, 6):
            vals = {m: row_val(desc, key, e, m) for m, _ in COLS}
            best = min((v for v in vals.values() if v is not None), default=None)
            cells.append({m: (('\\textbf{' + f3(v) + '}') if v is not None and v == best else f3(v)) for m, v in vals.items()})
        lines.append(lab + ' & ' + ' & '.join(f'{cells[0][m]}/{cells[1][m]}' for m, _ in COLS) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (O / 'generated' / 'main_table.tex').write_text('\n'.join(lines))

    # ---- counts over networks: who wins NLL at each condition
    wins = {}
    for e in (0, 3, 6):
        w = {'sf_vs_tabpfn': 0, 'sf_vs_pfnL': 0, 'n': 0}
        for key, *_ in NETS:
            s, t, p = row_val(desc, key, e, 'sf'), row_val(desc, key, e, 'tabpfn_v2'), row_val(desc, key, e, 'pfn_L')
            if s is None:
                continue
            w['n'] += 1
            w['sf_vs_tabpfn'] += int(t is not None and s < t)
            w['sf_vs_pfnL'] += int(p is not None and s < p)
        wins[e] = w

    # ---- robustness figure: pooled-over-networks mean NLL by condition, relative to all sensors
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for m, lab, col in (('sf', 'UPT-SF', 'C0'), ('pfn_L', 'PFN-L', 'C2'), ('tabpfn_v2', 'TabPFN v2', 'C3'), ('tnp', 'TNP-D', 'C1'),
                        ('ignnk_cal', 'kriging (cal.)', 'C4')):
        ys = []
        for e in (0, 3, 6):
            v = [row_val(desc, k, e, m) for k, *_ in NETS]
            v0 = [row_val(desc, k, 0, 'sf') for k, *_ in NETS]
            ok = [i for i in range(len(v)) if v[i] is not None and v0[i] is not None]
            ys.append(np.mean([v[i] - v0[i] for i in ok]) if ok else np.nan)
        ax[0].plot([0, 3, 6], ys, 'o-', label=lab, color=col)
        deg = []
        for e in (0, 3, 6):
            v = [(row_val(desc, k, e, m), row_val(desc, k, 0, m)) for k, *_ in NETS]
            v = [x - y for x, y in v if x is not None and y is not None]
            deg.append(np.mean(v) if v else np.nan)
        ax[1].plot([0, 3, 6], deg, 'o-', label=lab, color=col)
    ax[0].set_xlabel('extra sensors dropped'); ax[0].set_ylabel('NLL $-$ NLL(UPT-SF, all sensors)')
    ax[1].set_xlabel('extra sensors dropped'); ax[1].set_ylabel('NLL increase vs. own all-sensor NLL')
    for x in ax:
        x.set_xticks([0, 3, 6]); x.grid(alpha=.3)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(O / 'figures' / 'robustness.pdf'); plt.close(fig)

    # ---- every pre-registered endpoint
    def lab6(k):
        return {'P1_real_upt_vs_pfn': 'UPT vs transformer (PFN), NLL', 'P2_real_upt_noninf_pfn2pool': 'UPT non-inferior to 2-model pool',
                'P3_real_upt_vs_tabpfn': 'UPT vs TabPFN v2, NLL, all sensors', 'P4_syn16_upt_vs_pfn': 'UPT vs PFN, synthetic, 16 sensors'}.get(k)

    def lab7(k):
        m = {'Q1_upt_vs_tabpfn_e3': 'UPT vs TabPFN v2, NLL, +3 dropped', 'Q2_upt_vs_tabpfn_e6': 'UPT vs TabPFN v2, NLL, +6 dropped',
             'Q3_upt_vs_pfn_e0': 'UPT vs transformer (PFN), NLL', 'Q4_upt64_noninf_upt_e0': 'float32 sites non-inferior to float64',
             'S1_uptL_vs_tabpfn_e0': '(sec.) UPT-L vs TabPFN v2, all sensors', 'S1_uptL_vs_tabpfn_e3': '(sec.) UPT-L vs TabPFN v2, +3',
             'S1_uptL_vs_tabpfn_e6': '(sec.) UPT-L vs TabPFN v2, +6', 'S2_uptL_vs_upt_e0': '(sec.) UPT-L vs UPT'}
        return m.get(k)

    def lab8(k):
        b = {'tnp': 'TNP-D', 'cnp': 'CNP', 'ignnk_cal': 'calibrated kriging'}
        for kk, nm in b.items():
            for e, s in (('e0', 'all sensors'), ('e3', '+3 dropped'), ('e6', '+6 dropped')):
                if k.endswith(f'_vs_{kk}_{e}') and k[0] == 'R':
                    return f'UPT vs {nm}, NLL, {s}'
        return None

    def lab9(k):
        return {'V1_star_vs_tabpfn_nll_e0': 'UPT$^\\ast$ vs TabPFN v2, NLL, all sensors', 'V2_star_vs_tabpfn_nll_e3': 'UPT$^\\ast$ vs TabPFN v2, NLL, +3',
                'V2_star_vs_tabpfn_nll_e6': 'UPT$^\\ast$ vs TabPFN v2, NLL, +6', 'V3_star_vs_pfnL_nll_e0': 'UPT$^\\ast$ vs PFN-L (matched scale), NLL',
                'V4_star_vs_tnp_nll_e0': 'UPT$^\\ast$ vs TNP-D, NLL', 'V5_star_vs_krigcal_nll_e0': 'UPT$^\\ast$ vs calibrated kriging, NLL',
                'V6_star_vs_tabpfn_crps_e0': 'UPT$^\\ast$ vs TabPFN v2, CRPS$/\\sigma$'}.get(k)

    def lab10(k):
        return {'W1_sf_vs_tabpfn_nll_e0': 'UPT-SF vs TabPFN v2, NLL, all sensors', 'W2_sf_vs_tabpfn_nll_e3': 'UPT-SF vs TabPFN v2, NLL, +3',
                'W2_sf_vs_tabpfn_nll_e6': 'UPT-SF vs TabPFN v2, NLL, +6', 'W3_sf_vs_pfnLsf_nll_e0': 'UPT-SF vs PFN-L (matched scale), NLL',
                'W4_sf_vs_tabpfn_crps_e0': 'UPT-SF vs TabPFN v2, CRPS$/\\sigma$', 'W5_sf_vs_tnp_nll_e0': 'UPT-SF vs TNP-D, NLL',
                'W6_sf_vs_krigcal_nll_e0': 'UPT-SF vs calibrated kriging, NLL', 'W7_sf_vs_uptstar_nll_e0': 'UPT-SF vs UPT$^\\ast$ (effect of SF), NLL'}.get(k)
    rows = []
    for v, fn, nets in ((6, lab6, 'METR-LA, PEMS-BAY, Intel'), (7, lab7, 'PeMS03, PeMS08, China air'), (8, lab8, 'the six networks above'),
                        (9, lab9, 'DWD temp., AURN NO$_2$, USGS PA'), (10, lab10, 'USGS NY, USGS WA, DWD pressure, AURN PM$_{2.5}$')):
        if v in c:
            rows += [(f'v{v}', nets, lab, r) for _, lab, r in endpoint_rows(v, c[v], fn)]
    t = ['\\begin{table}[h]\\centering\\scriptsize', '\\caption{\\textbf{Every pre-registered endpoint} of the protocols that '
         'evaluate UPT. Gain = baseline $-$ UPT (positive favours UPT), paired 95\\% interval clustered by network-window.}',
         '\\label{tab:all-endpoints}', '\\begin{tabular}{llrcc}', '\\toprule', 'Protocol & Endpoint & Gain & 95\\% CI & Result \\\\', '\\midrule']
    last = None
    for v, nets, lab, r in rows:
        if v != last:
            t.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{v}: {nets}}}}} \\\\'); last = v
        res = 'pass' if r['passed'] else FAIL
        t.append(f' & {lab} & {r["gain"]:+.3f} & [{r["lo"]:+.3f}, {r["hi"]:+.3f}] & {res} \\\\')
    t += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (O / 'generated' / 'all_endpoints_table.tex').write_text('\n'.join(t))
    npass = sum(r['passed'] for *_, r in rows); ntot = len(rows)

    # ---- networks table
    counts = json.loads(Path(a.counts).read_text()) if a.counts else {}
    t = ['\\begin{table}[h]\\centering\\scriptsize', '\\caption{\\textbf{The thirteen sensor networks.} Test period year; '
         'protocol that introduced the network; test episodes and network-time-window clusters.}', '\\label{tab:networks}',
         '\\begin{tabular}{llllccc}', '\\toprule', 'Network & Quantity & Source & Test & Protocol & Episodes & Windows \\\\', '\\midrule']
    for key, lab, dom, src, yr in NETS:
        n = counts.get(key, ['--', '--', '--'])
        t.append(f'{lab} & {dom} & {src} & {yr} & {PROTO[key]} & {n[0]} & {n[1]} \\\\')
    t += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (O / 'generated' / 'networks_table.tex').write_text('\n'.join(t))

    # ---- per-network appendix tables: NLL / CRPS/sigma / RMSE/sigma / cov90 at e0, e3, e6
    t = []
    for e in (0, 3, 6):
        t += ['\\begin{table}[h]\\centering\\scriptsize', f'\\caption{{Per-network metrics, {"all sensors" if e == 0 else f"+{e} sensors dropped"}: '
              'NLL / CRPS$/\\sigma$ / RMSE$/\\sigma$ / 90\\% coverage.}', '\\setlength{\\tabcolsep}{2pt}',
              '\\begin{tabular}{l' + 'c' * 5 + '}', '\\toprule', 'Network & UPT-SF & PFN-L & TabPFN v2 & TNP-D & Krig.\\ (cal.) \\\\', '\\midrule']
        for key, lab, *_ in NETS:
            cell = []
            for m in ('sf', 'pfn_L', 'tabpfn_v2', 'tnp', 'ignnk_cal'):
                r = desc.get(f'{key}|e{e}', {}).get(m)
                if not r:
                    cell.append('--'); continue
                cell.append(f"{r['nll']:+.2f}/" + (f"{r['crps_std']:.3f}" if 'crps_std' in r else '--') + f"/{r['rmse_std']:.3f}/" +
                            (f"{r['cov90']:.2f}" if 'cov90' in r else '--'))
            t.append(lab + ' & ' + ' & '.join(cell) + ' \\\\')
        t += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (O / 'generated' / 'per_network_tables.tex').write_text('\n'.join(t))

    # ---- numbers
    eps = sum(int(v[0]) for v in counts.values()) if counts else 0
    W = {k: d10.get(k, {}) for k in ('W1_sf_vs_tabpfn_nll_e0', 'W2_sf_vs_tabpfn_nll_e3', 'W2_sf_vs_tabpfn_nll_e6', 'W3_sf_vs_pfnLsf_nll_e0',
                                     'W4_sf_vs_tabpfn_crps_e0', 'W5_sf_vs_tnp_nll_e0', 'W6_sf_vs_krigcal_nll_e0', 'W7_sf_vs_uptstar_nll_e0')}
    num = [f'\\newcommand{{\\NumProtocols}}{{eleven}}', f'\\newcommand{{\\NumNetworks}}{{thirteen}}', f'\\newcommand{{\\NumFreshNetworks}}{{Seven}}',
           f'\\newcommand{{\\NumEpisodes}}{{{eps:,}}}'.replace(',', '{,}'), f'\\newcommand{{\\NumEndpointsPass}}{{{npass}}}',
           f'\\newcommand{{\\NumEndpoints}}{{{ntot}}}']
    for e in (0, 3, 6):
        nm = {0: 'Zero', 3: 'Three', 6: 'Six'}[e]
        num += [f'\\newcommand{{\\WinsTab{nm}}}{{{wins[e]["sf_vs_tabpfn"]}}}', f'\\newcommand{{\\WinsPfn{nm}}}{{{wins[e]["sf_vs_pfnL"]}}}',
                f'\\newcommand{{\\WinsN{nm}}}{{{wins[e]["n"]}}}']
    for k, v in W.items():
        tag = ''.join(ch for ch in k.split('_')[0] if ch.isalnum()) + ('Three' if k.endswith('e3') else 'Six' if k.endswith('e6') else '')
        tag = tag.replace('1', 'One').replace('2', 'Two').replace('3', 'Three').replace('4', 'Four').replace('5', 'Five').replace('6', 'Six').replace('7', 'Seven')
        if 'gain' in v:
            num.append(f'\\newcommand{{\\{tag}}}{{{v["gain"]:+.3f} [{v["lo"]:+.3f}, {v["hi"]:+.3f}]}}')
    (O / 'generated' / 'numbers_auto.tex').write_text('\n'.join(num) + '\n')
    print(json.dumps(dict(endpoints=f'{npass}/{ntot}', wins=wins), indent=1))


if __name__ == '__main__':
    main()
