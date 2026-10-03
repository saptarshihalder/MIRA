"""Generate concise reports from retained scores; never train or select a model."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/reports/policy_adapter_v1'
report = json.loads((OUT/'report.json').read_text())
training = json.loads((OUT/'training.json').read_text())
audit = json.loads((OUT/'independent_audit.json').read_text())
lines = ['# Predictive-loss context adapter: failed development gate', '',
    'Protocol/code freeze `ebac808`; final checkpoints committed `4514490` before development.',
    'CPU training only: 768 independent paired-policy worlds, 1,536 episodes, 40 fixed epochs,',
    '128 validation episodes, 47.062 seconds. Context/no-shift models11,176 parameters; ordinary12,601.',
    'Evaluation:48 independent worlds ×5 policy cells,240 episodes,2,160 saved prediction arrays.',
    'No additional paid call or confirmation. Prior $10.55 reservations and $3 reproduction reserve unchanged.', '',
    '| Method | Expected NLL | Comparator minus context (nominal 95% task interval) |',
    '|---|---:|---:|']
for name in report['expected_nll']['null']:
    loss = sum(r[name] for r in report['expected_nll'].values())/5
    contrast = report['gains'].get(name)
    value = (f"{contrast['mean']:.6f} [{contrast['low']:.6f}, {contrast['high']:.6f}]" if contrast else 'reference')
    lines.append(f'| {name} | {loss:.6f} | {value} |')
lines += ['', 'The context learner beats ordinary and no-shift neural controls, but loses to the centered-moment',
    'correction by .044606 NLL and has .045750 mean null harm against the .01 limit. The gate fails.',
    'Expected NLL averages paired policy cells within outcome worlds before uncertainty. Intervals are',
    'developmental and nominal; they do not support a confirmatory multiple-comparison claim.', '',
    'Independent audit reconstructs all saved expected/empirical scores and task intervals exactly.',
    'Oracle reconstruction from saved float32 U differs by at most3.35e-8. Six boundary checks passed',
    'using runpy (pytest is absent in the isolated CPU environment).', '',
    'Limitations: sufficient always-observed U, four-bit/full-parity structural prior, engineered moments',
    'in addition to a learned row encoder, limited target labels and non-identical optimization/capacity.',
    'Only rate/block/value extremes exceed pretraining support;17/48 reversal cells have null source signal.',
    'Pooled ridge penalizes the source intercept while exempting the final target-domain intercept.',
    'No real-data, TFM, clinical, causal-policy-identification, safety or venue-readiness claim.', '',
    'Reproduction with the pinned CPU environment (Torch2.8.0+cpu,NumPy2.3.5,sklearn1.8.0):', '',
    '```powershell',
    "& '../.venv-compiler-cpu/Scripts/python.exe' experiments/policy_adapter_v1/pilot.py train --out artifacts/runs/policy_reproduction_v1",
    "& '../.venv-compiler-cpu/Scripts/python.exe' experiments/policy_adapter_v1/pilot.py evaluate --out artifacts/runs/policy_reproduction_v1",
    '```', '', 'Do not overwrite existing outputs; clean reproduction remains a later readiness gate.',
    'Next design requirements: `docs/POLICY_ADAPTER_NEXT_GATE.md`.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
freeze = json.loads((ROOT/'artifacts/manifests/policy_adapter_v1_freeze.json').read_text(encoding='utf-8-sig'))
for relative, digest in freeze['files'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
source = (ROOT/'paper/main.tex').read_text()
stack = []
for match in re.finditer(r'\\(begin|end)\{([^}]+)\}', source):
    kind, environment = match.groups()
    if kind=='begin':
        stack.append(environment)
    else:
        assert stack.pop()==environment
assert not stack
citations = {item.strip() for match in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', source) for item in match.split(',')}
references = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', source))
assert citations <= set(re.findall(r'\\bibitem\{([^}]+)\}', source))
assert references <= set(re.findall(r'\\label\{([^}]+)\}', source))
manifest = dict(path=str(ROOT/'paper/main.tex'), source_sha256=hashlib.sha256((ROOT/'paper/main.tex').read_bytes()).hexdigest(),
    native_compiler_status='compile-failed', diagnostic='Unable to find standard directories for platform',
    pdf_verified=False, source_checks=dict(environments_balanced=True,citations_resolved=True,references_resolved=True),
    policy_protocol_files_rechecked=len(freeze['files']),
    limitation='Source checks do not establish native compilation or PDF layout; same editor remains open.')
(ROOT/'artifacts/manifests/paper_compile.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Report generated; frozen new protocol files and manuscript source checks passed.')
