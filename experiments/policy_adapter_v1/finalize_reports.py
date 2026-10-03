"""Readable progress notes from retained evidence; no fitting or selection."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
def write(path,text):
    path.write_text(text.replace('\r\n','\n'),encoding='utf-8',newline='\n')
def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8-sig'))
native=read('artifacts/reports/policy_native_gpu_v1/report.json')
synthetic=read('artifacts/reports/policy_mixture_v1/report.json')
audit=read('artifacts/reports/policy_native_gpu_v1/independent_audit.json')
ledger=read('artifacts/manifests/compute_ledger.json')
reserved=sum(entry['reserved_usd'] for entry in ledger)
assert reserved==11.05 and ledger[-1]['status']=='completed'
for relative in ('policy_adapter_v1_freeze.json','policy_mixture_v1_freeze.json','policy_native_v1_freeze.json'):
    for name,digest in read('artifacts/manifests/'+relative)['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for name,digest in read('configs/policy_gpu_native_v1.json')['file_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

for folder,result in (('policy_mixture_v1',synthetic),('policy_native_gpu_v1',native)):
    items=result['results']
    metric='nll' if folder=='policy_native_gpu_v1' else 'expected'
    lines=['# '+folder.replace('_',' '),'', '| Method | Mean NLL |','|---|---:|']
    for name in items[0][metric]:
        lines.append(f"| {name} | {sum(e[metric][name] for e in items)/len(items):.6f} |")
    if metric=='nll':
        lines += ['', 'Actual Tesla T4 run: XGBoost source training and three correction models on CUDA.',
            'The fine-tuned primary, scratch neural and linear scorers received 400 updates each.',
            'Native source/meta-training/validation/development patients are disjoint. The source backbone',
            'uses 4,096 labels; fixed source context128; native episodes32 training/16 validation/32 development.',
            'Each episode has64 labeled target support and256 queries. Final epochs are fixed;',
            'only validation labels select the operational guard threshold.', '',
            'Primary fine-tuning loses to scratch and linear models. Descriptive scratch gain over frozen',
            'is .006491 [.002546,.010437], but its gain over linear is .000449 [-.000388,.001285].',
            'Synthetic pretraining and a distinctive nonlinear advantage are not established.',
            'Intervals condition on this single dataset, trained source backbone and initialization.', '',
            'Independent audit passes: all256 probability arrays, patient boundaries and hashes.',
            'NLL discrepancy <=2.98e-8; interval discrepancy <=1.61e-9; NumPy scorer <=1.79e-7.',
            'Preserve all14 warnings: one XGBoost host-device prediction fallback and13 low-class-count CV warnings.',
            'Do not claim every inference kernel ran on CUDA. No verified hospital/date split or clinical claim.', '',
            'Source: UCI Diabetes130 (CC BY4.0), https://doi.org/10.24432/C5230J.',
            'Official download hashes/token audit: artifacts/manifests/diabetes_native_source.json.',
            'A proposed70/15/15 preparation split was not adopted; executed protocol uses60/20/20.', '',
            'Frozen protocol commit68f5dcd; Modal app ap-yd9YbpSmYeKwFYgpgepUj5.',
            'One successful remote invocation,16.877 wrapper seconds. Pre-dispatch Windows console failure',
            'is retained within the same$.50 reservation. No remote retry. Total reservations11.05/26;',
            '3 reproduction reserve protected. Provider display rounds to$.00; final invoice unverified.',
            'Runtime active-compute estimate ~$.00351 excludes startup/build/other charges.', '',
            'An earlier CPU native screen failed at a baseline optimizer limit after2 completed episodes;',
            'its partial outputs/failure manifest remain. This GPU screen uses separately frozen controls.']
    else:
        lines += ['', '961-parameter neural predictive mixture, equally pretrained linear scorer, calibrated guard',
            'and support-only controls. Fixed60-epoch checkpoint;768 training worlds/1,536 episodes.',
            'Development48 independent worlds/five cells each;2,400 saved probability arrays.',
            'Moment gain .015773 [.008210,.023337], null harm .001930. Calibrated-guard gain',
            '.003645 [-.000058,.007349] fails the full gate. No confirmatory or safety claim.',
            'Independent saved-score audit passes exactly. OOF fold ordering and moment-selector differences',
            'remain documented. Protocol a6033c0; checkpoints447b488 before evaluation.']
    write(ROOT/'artifacts/reports'/folder/'report.md','\n'.join(lines)+'\n')

status=subprocess.check_output(['git','show','578ce94:STATUS.md'],cwd=ROOT).decode('utf-8')
history=status[status.index('## Completed evidence'):]
header='''# MIRA status — October 3, 2026

Target: a NeurIPS-quality draft by October20; trained-model contribution required. Real native-data CUDA training is complete, but methodological novelty and venue readiness remain unresolved. Strict token rationing continues.

## Current checkpoint: read this before continuing

- **Actual GPU training:** `artifacts/reports/policy_native_gpu_v1/`. Frozen protocol68f5dcd; one Modal Tesla T4 run. XGBoost trained4,096 source labels on CUDA; fine-tuned/scratch neural models (961 parameters) and linear scorer (13) each received400 CUDA updates. Torch2.8.0+cu128, source booster cuda:0, wrapper16.877 seconds. All checks verified; do not claim every XGBoost inference kernel ran on CUDA.
- **Native outcome:** NLL frozen.313739, synthetic fixed.318891, fine-tuned.311691, scratch.307248, linear.307697. Predeclared fine-tuned primary loses to scratch/linear. Descriptive scratch gain over frozen .006491 [.002546,.010437], but over linear .000449 [-.000388,.001285]. Synthetic pretraining and distinctive neural benefit remain unsupported.
- **Native boundaries:** UCI Diabetes101,766 encounters;69,990 eligible patient representatives after official death/hospice exclusions and label-independent encounter selection. Executed patient split60/20/20; source fit/context/meta-training, validation and development are disjoint. Native episodes32 training/16 validation/32 development, with64 support and256 queries each. IDs excluded. Missing '?' and unmeasured-test 'None' semantics preserved. No hospital/date fields, verified acquisition-policy shift or clinical claim. Proposed70/15/15 preparation split was not adopted.
- **Independent GPU audit:** all256 probability arrays, source/protocol/raw/checkpoint/backbone hashes and patient boundaries pass. Float32 NLL discrepancy <=2.98e-8, intervals <=1.61e-9, NumPy scorers <=1.79e-7. Keep14 warnings (XGBoost prediction-device mismatch plus13 class-count CV warnings). Episode intervals are conditional on one dataset/shared trained source.
- **Synthetic uncertainty learner:** `artifacts/reports/policy_mixture_v1/`, protocol a6033c0, checkpoints447b488.961 parameters,768 paired-policy worlds/1,536 training episodes,60 fixed epochs,30.656 CPU seconds. New development48 worlds/240 episodes/2,400 probability arrays. Gain over moment .015773 [.008210,.023337], null harm .001930; calibrated-guard gain .003645 [-.000058,.007349] fails the full gate. Exact score audit passes. OOF order dependence and guard/original-selector differences remain.
- **Earlier failures retained:** dense11,176-parameter context learner improves over ordinary/no-shift models but loses to the moment correction and harms null tasks. Compiler matches its heuristic. Historical324 backbone cells remain fixed (162 GPU/162 CPU). Local native HistGB screen failed at a support-CV baseline optimizer limit after2 completed episodes; preserve partial results. Seventeen new boundary/worker checks pass; no repeated old experiments.
- **Budget:** reservations$11.05 across23 calls, with$3 reproduction reserve inside$26. New$.50 native CUDA phase is closed after one successful remote invocation. A Windows launcher encoding failure occurred before dispatch, with zero GPU tasks observed, and is retained in the same reservation. UTF8 repair did not retry a remote function. Provider UI displays rounded$0.00; final charge remains unverified. Active-compute estimate~$.00351 excludes other charges. Prior$1 compiler pilot/$.50 repair remain closed.
- **Artifacts/access:** official public CSV/ZIP/mapping are retained under ignored `artifacts/runs/diabetes_native_raw`; tracked source audit/DOI/hashes permit re-download. BrowserOS was connected and used for public data and billing. Colab Pro was not verified or used. Same open `paper/main.tex` includes actual evidence; native compilation still fails with `Unable to find standard directories for platform`. Source checks pass; PDF/layout unverified.

Next: compare converged linear-mixture and global/support calibration controls with identical native meta-label access. Then justify a variable-width, query-conditioned learner across naturally incomplete datasets with defensible acquisition boundaries; a fixed four-mask parity bank is insufficient novelty evidence. Read `docs/POLICY_ADAPTER_NEXT_GATE.md` and independent audits. Do not relabel used panels as untouched or relax failed gates. Any new scientific experiment needs a new protocol/seeds; paid expansion needs useful evidence, committed bounds and reservations. Explicit user GPU engineering requests remain distinct from efficacy/confirmation. Seeds98000–98019 remain unused. Daily automation follows this checkpoint. No email, submission or acceptance/safety promise.

'''
# Keep historical evidence and its old dated reservations as history, rather than deleting it.
spacing={'October20':'October 20','protocol68f5dcd':'protocol 68f5dcd','trained4,096':'trained 4,096',
    'received400':'received 400','Torch2.8.0':'Torch 2.8.0','wrapper16.877':'wrapper 16.877',
    'frozen.313739':'frozen .313739','fixed.318891':'fixed .318891','fine-tuned.311691':'fine-tuned .311691',
    'scratch.307248':'scratch .307248','linear.307697':'linear .307697','Diabetes101,766':'Diabetes 101,766',
    ';69,990':'; 69,990','split60/20/20':'split 60/20/20','split70/15/15':'split 70/15/15',
    'episodes32':'episodes 32','with64':'with 64','and256':'and 256','all256':'all 256','Keep14':'Keep 14',
    'plus13':'plus 13','checkpoints447b488':'checkpoints 447b488','.961 parameters':'. 961 parameters',
    ',768 paired':', 768 paired',',60 fixed':', 60 fixed',',30.656':', 30.656','development48':'development 48',
    'worlds/240':'worlds / 240','after2':'after 2','dense11,176':'dense 11,176','Historical324':'Historical 324',
    'reservations$11.05':'reservations $11.05','across23':'across 23','with$3':'with $3','inside$26':'inside $26',
    'New$.50':'New $.50','estimate~$.00351':'estimate ~$.00351','Prior$1':'Prior $1','Seeds98000':'Seeds 98000'}
for before,after in spacing.items(): header=header.replace(before,after)
write(ROOT/'STATUS.md',header+history)
marker='\n## October3: uncertainty mixture and real native CUDA training\n'
decision=(ROOT/'decision.md').read_text(encoding='utf-8').split(marker)[0]
note='''
The 961-parameter uncertainty learner improves over the synthetic moment correction, but its strongest guard comparison crosses zero. The full gate remains failed. On the user's explicit GPU instruction, a separately frozen $0.50 native engineering/training phase ran successfully on a Modal Tesla T4. XGBoost source training and three correction models used CUDA; each correction received 400 updates. Patient boundaries, raw-data semantics, checkpoints and all 256 native arrays pass independent audit within float32 rounding.

The fine-tuned primary loses to native scratch and linear models. Descriptive scratch gain over frozen is .006491 [.002546,.010437]; gain over the 13-parameter linear control is .000449 [-.000388,.001285]. This supports limited native utility, without establishing nonlinear novelty or a synthetic-pretraining benefit. Preserve warnings and earlier CPU/launcher failures. Reservations $11.05/$26 retain the $3 reproduction reserve; provider charge remains pending/rounded. No remote retry or confirmation. Next compare converged/calibration controls before a broader trainable successor. Native PDF compilation remains blocked by its environment.
'''
write(ROOT/'decision.md',decision+marker+note)
next_gate='''# Next gate for a trained-model contribution

Keep the user-required trainable-model core and every negative result. The compiler matches a heuristic; the dense context learner loses to the moment correction. The 961-parameter uncertainty mixture has limited synthetic/native utility, but distinct methodological novelty remains unresolved.

## Evidence that constrains the next step

Synthetic gain over moment is .015773 [.008210,.023337], with null harm .001930. The calibrated-guard contrast .003645 [-.000058,.007349] fails the full gate. Do not add tasks until this test passes or relax its threshold after seeing results. A new study needs one fixed protocol and fresh development.

Actual Modal T4 native training completed under the user's explicit GPU instruction. The predeclared fine-tuned primary loses to native scratch and linear models. Descriptive scratch gain over frozen is .006491 [.002546,.010437], but gain over the 13-parameter linear model is .000449 [-.000388,.001285]. Synthetic pretraining and a distinctive nonlinear benefit are unsupported. Native patient boundaries and checkpoints pass audit. Diabetes has no verified hospital/date fields: retrospective patient-grouped utility does not establish policy transport or clinical effectiveness.

## Required successor work

1. Fit converged linear stacking and global/support calibration with identical native meta-label access. Equal Adam updates do not establish equal optimization. Determine whether ordinary calibration explains the improvement.
2. Verify more naturally incomplete datasets, acquisition/outcome timing and genuine site/time boundaries where available. Freeze cross-dataset partitions before training; do not infer chronology from numeric IDs or missingness rates.
3. Consider a variable-width, query-conditioned learner over observed values and masks, conditioned on labeled source/target support. Train on predictive loss. Require improvement beyond calibrated shrinkage, conditional-interaction models and equally pretrained contextual learners. The current sixteen handcrafted candidates are a restrictive prior; greater synthetic complexity alone is insufficient novelty evidence.
4. Check targeted prior art before claiming architecture/objective novelty. Preserve query-label boundaries, checkpoints, negative results and label/pretraining budgets. Use a new protocol, seed namespace, useful-effect threshold and finite stopping rule. Existing panels are used development; confirmation seeds 98000–98019 remain unused.

Reservations are $11.05 within $26, protecting $3 for reproduction. The new $0.50 native GPU phase and old phases are closed. Include launcher failures and provider charges; a displayed rounded $0.00 is not a verified zero invoice. Scientific paid expansion needs a useful gate and separately committed bounds; explicit engineering requests remain distinct from efficacy/confirmation.

Readiness still requires broad native evidence, current backbones, appropriate task uncertainty, clean reproduction, formatting/authorship and native PDF verification. Preserve the same open manuscript. No acceptance, universal-improvement, safety or clinical-effectiveness promise.
'''
write(ROOT/'docs/POLICY_ADAPTER_NEXT_GATE.md',next_gate)
sprint=(ROOT/'SPRINT.md').read_text(encoding='utf-8')
parts=sprint.split('\n')
for i,line in enumerate(parts):
    if line.startswith('Primary track:'):
        parts[i]='Primary track: a trained-model contribution evaluated against strong simple controls. The 961-parameter uncertainty mixture improves over synthetic moments, but its calibrated-guard advantage is uncertain. Actual Modal T4 native Diabetes training is complete: source XGBoost and 400 CUDA updates each for fine-tuned/scratch neural and linear corrections. Native pretraining does not help; scratch neural and linear models perform similarly. Read docs/POLICY_ADAPTER_NEXT_GATE.md before a successor. Compare calibration and converged controls before broader query-dependent learning across native datasets. Historical representation evidence does not establish a causal pretraining-prior effect.'
write(ROOT/'SPRINT.md','\n'.join(parts))
print('Readable evidence reports generated; all four frozen input manifests passed.')
