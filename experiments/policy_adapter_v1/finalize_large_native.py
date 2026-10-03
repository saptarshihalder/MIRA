"""Curate actual GPU results and failure records; derive descriptive tables."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path.write_text(value,encoding='utf-8',newline='\n')


def interval(values):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean())
    radius=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean,low=mean-radius,high=mean+radius,groups=len(values))


if __name__=='__main__':
    raw=ROOT/'artifacts/runs/large_native_a100_v3'
    out=ROOT/'artifacts/reports/large_native_a100_v3'
    if out.exists():
        raise ValueError('Preserve curated outputs')
    shutil.copytree(raw/'results',out)
    shutil.copy2(raw/'wrapper_runtime.json',out/'wrapper_runtime.json')
    audit=json.loads((out/'independent_audit.json').read_text())
    rows=audit['development_rows']
    contrasts={}
    for task,means in audit['development_group_mean_nll'].items():
        groups=sorted({row['group'] for row in rows if row['task']==task})
        contrasts[task]={name:interval([np.mean([row['nll'][name]-row['nll']['native_scratch']
                                               for row in rows if row['task']==task and row['group']==group])
                                       for group in groups]) for name in means if name!='native_scratch'}
    summary=dict(group_mean_nll=audit['development_group_mean_nll'],comparator_minus_neural=contrasts,
                 independent_audit='independent_audit.json',trials=9,primary='native_scratch',
                 inference='Descriptive t intervals over fixed held-out regions after averaging three seeds; correlated health tasks, two domains, conditional on trained sources; not confirmation')
    write(out/'summary.json',json.dumps(summary,indent=2))
    names=['frozen','native_scratch','native_linear','native_linear_converged','global_simplex','global_platt','support_intercept','support_platt','cv_select','moment']
    tasks=['brfss_diabete4','brfss_asthma3','road_ksi']
    text='# Larger native A100 development\n\nNine actual NVIDIA A100-SXM4-40GB trials: three tasks × three training seeds.\nFour official 2024/2025 files contain 1,016,280 raw records; latest cohorts were released in 2026.\nThe health tasks share respondents. Three seeds are training variation, not independent datasets.\nSource fitting uses 50,000 labels per trial; each neural/Adam-linear scorer gets 800 CUDA updates.\n\n| Method | Diabetes | Asthma | Road KSI |\n|---|---:|---:|---:|\n'
    for name in names:
        text+='| '+name+' | '+' | '.join(f"{summary['group_mean_nll'][task][name]:.6f}" for task in tasks)+' |\n'
    text+='\nNLL in nats, lower is better; equally weighted held-out regions and seeds.\nDevelopment has ten states per health target and six police-force groups for road collisions.\nThe model improves road loss relative to the frozen source, but support Platt and global simplex stacking perform better.\nHealth effects are tiny; diabetes neural loss is slightly worse than frozen. Distinct neural novelty remains unsupported.\n\nAll 1,470 probability arrays, checkpoints, source hashes, public-row labels and group/year boundaries pass independent audit.\nMaximum NLL discrepancy 7.05e-8; NumPy checkpoint prediction discrepancy 2.02e-7.\nAll nine contextual linear fits attain the 1e-7 gradient tolerance; worst simplex KKT residual is 2.01e-6.\nXGBoost host/device prediction fallback is retained: CUDA training is verified, every inference kernel is not asserted to use CUDA.\n\nThe first A100 call stopped on a calibration L-BFGS line search after four complete trials; all partial results are preserved.\nA separately frozen Newton repair uses exactly the same ridge objective, inputs, partitions, seeds and models.\nIts first package lacked a required historical source mount and failed before child training.\nThe final complete mount finished all nine trials in 62.020 seconds. Failures are included in the ledger.\nNo automatic retry or scientific selection was performed.\n\nReservations are $16.95 within the latest $20 cap, with $3 protected for reproduction.\nProvider app UI reports $0.04/$0.02/$0.05 for the three attempts (rounded and not verified final invoices).\nThe successful app remains deployed with zero active GPU containers, no schedule and no public endpoint.\n\n[Live Modal app](https://modal.com/apps/saptarshihalder/main/deployed/mira-large-native-a100-final)\n\n[CDC 2025](https://www.cdc.gov/brfss/annual_data/annual_2025.html), [CDC 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html), [DfT](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data).\nRaw input hashes, codebook/layout rules and fixed protocol are committed; raw data remain ignored.\nBRFSS age is already imputed, reporting/state composition changes, and road-specification migration creates structural absence.\nThese results do not identify causal acquisition-policy effects, population prevalence, clinical effectiveness or venue readiness.\n'
    write(out/'report.md',text)
    for version in (1,2):
        destination=ROOT/f'artifacts/reports/large_native_a100_v{version}_failure'
        destination.mkdir()
        source=ROOT/f'artifacts/runs/large_native_a100_v{version}'
        for filename in ('runner.log','wrapper_runtime.json'):
            if (source/filename).exists():
                shutil.copy2(source/filename,destination/filename)
        if (source/'results').exists():
            shutil.copytree(source/'results',destination/'partial_results')
        write(destination/'failure.json',json.dumps(dict(version=version,preserved=True,
              reason='Calibration L-BFGS-B line search; four complete trials and partial fifth' if version==1 else 'Missing frozen v1 wrapper at /opt/mira/infra; remote input check failed before child training',
              scope='Failed run; not a complete evaluation; charges retained'),indent=2))
    billing=dict(observed_date='2026-10-03',provider='Modal',precision='USD rounded to 2 decimal places',
                 apps=[dict(app_id=identity,display_usd=value,status='rounded provider UI, not final invoice') for identity,value in
                       [('ap-29sdBNwyGIJse79T72LRNz',.04),('ap-FGMu0nPPjGLxIT7Ds63p70',.02),('ap-raKw2bH4mQsaJMMJXkJwvJ',.05)]],
                 sum_display_usd=.11,final_invoice_verified=False,actual_charged_usd=None,reservations_retained=True)
    write(ROOT/'artifacts/manifests/provider_billing_large_native.json',json.dumps(billing,indent=2))
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    entries=json.loads(ledger.read_text())
    by_run={f'large_native_a100_v{version}':(identity,value) for version,identity,value in
            [(1,'ap-29sdBNwyGIJse79T72LRNz',.04),(2,'ap-FGMu0nPPjGLxIT7Ds63p70',.02),(3,'ap-raKw2bH4mQsaJMMJXkJwvJ',.05)]}
    for entry in entries:
        if entry['run_id'] in by_run:
            entry.update(app_id=by_run[entry['run_id']][0],provider_display_usd=by_run[entry['run_id']][1],
                         provider_cost_status='rounded/pending final invoice; full reservation retained')
    write(ledger,json.dumps(entries,indent=2))
    print(json.dumps(dict(curated=str(out),group_mean_nll=summary['group_mean_nll'],contrasts=contrasts['road_ksi'])))
