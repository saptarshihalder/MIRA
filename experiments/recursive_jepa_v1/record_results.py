"""Curate actual engineering evidence and synchronize notes; no training calls."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', newline='\n', encoding='utf-8')


def main():
    source = ROOT / 'artifacts/runs/recursive_jepa_v1_gpu/results'
    target = ROOT / 'artifacts/reports/recursive_jepa_v1_gpu'
    audit = json.loads((target / 'independent_audit.json').read_text(encoding='utf-8'))
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if not audit['passed'] or report['device'] != 'cuda' or report['gate_pass'] is not None:
        raise ValueError('Require audited engineering-only CUDA evidence')
    for file in source.rglob('*'):
        if file.is_file():
            destination = target / file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    shutil.copy2(source.parent / 'wrapper_runtime.json', target / 'wrapper_runtime.json')
    apps = json.loads((ROOT / 'artifacts/manifests/modal_apps_after_recursive_jepa.json').read_text(encoding='utf-8-sig'))
    app = next(item for item in apps if item['description'] == 'mira-recursive-jepa-v1')
    if app['tasks'] != '0':
        raise ValueError('Expected completed worker with zero active tasks')
    resources = json.loads((ROOT / 'artifacts/manifests/modal_billing_after_recursive_jepa.json').read_text(encoding='utf-8-sig'))
    rows = [row for row in resources if row['object_id'] == app['app_id']]
    charged = sum((Decimal(row['cost']) for row in rows), Decimal(0))
    ledger = ROOT / 'artifacts/manifests/compute_ledger.json'
    entries = json.loads(ledger.read_text(encoding='utf-8'))
    entry = next(item for item in entries if item['run_id'] == 'recursive_jepa_v1_gpu')
    entry.update(app_id=app['app_id'], gpu_verified=report['gpu'], cuda_updates_per_trained_model=64,
        provider_reported_resource_cost_usd=float(charged) if rows else None,
        provider_report_source='artifacts/manifests/modal_billing_after_recursive_jepa.json',
        provider_cost_status='Resource report may lag; no matching rows means unavailable, not zero cost. Final invoice unverified; $.80 provision retained',
        active_tasks=0)
    save(ledger, entries)
    provisions = sum(Decimal(str(item['reserved_usd'])) for item in entries)
    if provisions != Decimal('14.85'):
        raise ValueError('Revisit budget notes rather than silently overwriting new work')
    summary = dict(engineering_only=True, gate_pass=None, gpu=report['gpu'],
        runner_seconds=report['seconds'], wrapper_seconds=entry['seconds'],
        trainable_parameters=6880, stored_parameters=8448, cuda_updates_per_model=64,
        controls=list(report['training_models']), audit_passed=True,
        interpretation='JEPA prediction loss falls but latent spread stays low and downstream changes are small. No efficacy, anti-collapse, novelty or robustness claim.',
        current_provisions_usd=float(provisions), reproduction_reserve_usd=3,
        unallocated_usd=2.15, provider_invoice_verified=False)
    save(target / 'summary.json', summary)
    manifest = {str(path.relative_to(target)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in target.rglob('*') if path.is_file() and path.name != 'artifact_manifest.json'}
    save(target / 'artifact_manifest.json', dict(files=manifest, protocol_sha256=report['protocol_sha256']))
    note = '''## Recursive JEPA engineering checkpoint — October 3

Implemented and actually trained a 6,880-trainable-parameter, three-step tied recursive label-memory model on Modal NVIDIA A100-SXM4-40GB. Support-only residual/class memories drive nonlinear embeddings and bounded differentiable head updates; supervised meta-query loss backpropagates through every step. A support-conditioned EMA teacher predicts embeddings across nested observed-only missingness views. Query labels never enter inference or the teacher target. JEPA/VICReg/MAML motivate ingredients; novelty is unestablished.

Three controls completed64 CUDA updates each (full, supervised-only, one-step); runner13.835s/wrapper24.838s. Seven grouped unit tests and a separate two-update CPU engineering smoke pass. Independent audit verifies16 files/96 probability arrays and48 CPU checkpoint predictions (max error1.19e-7), all label boundaries, source logits, teacher states and exact frozen intervention. Saved raw predictions, all checkpoints, traces and audit are in artifacts/reports/recursive_jepa_v1_gpu. Protocol96258a1f was committed252c348 before the paid call. The app has zero active tasks, no schedule or endpoint, and no retries.

Full JEPA auxiliary MSE1.036854→.003989; student mean std .022198→.029807 stays far below variance floor1 and final recurrent query std is .015229 on the logged batch. This is low-spread/collapse-risk evidence, not robust label representation. Diagnostic NLL full/frozen: sign-flip .616819/.617999; nonlinear .592116/.593328; no-shift .600430/.601214; ignorable .646807/.647985. Supervised-only and one-step scores are almost identical; support logistic sign-flip .503530 is much stronger. Two correlated development worlds cannot establish safety, significance or neural novelty. Different recurrence/deep-supervision/auxiliary costs prevent a causal ablation claim. Gate is deliberately null, not passed. Earlier failed full gates remain failed.

Next: diagnose representation geometry and nonlinear label discrimination on CPU before another GPU protocol. Use per-world variance/effective rank, held-out support class separation and shuffled-label controls; do not solve a low-variance auxiliary merely by inflating embedding norm. Consider a learned field interaction/attention representation that preserves task-relevant cross-feature structure, then independently validated frozen fallback. Freeze a fresh computationally matched protocol and per-regime/null gate before scientific GPU expansion. Confirmation98000-98019 remains untouched; no native extension is activated.

Conservative provisions14.85 plus protected reproduction3 within20 leave2.15 unallocated. New $.80 retained; provider report has not yet posted matching rows, so charge is unavailable, not zero. Preserve source/editor; no email or submission.

'''
    status = ROOT / 'STATUS.md'
    current = status.read_text(encoding='utf-8')
    if '## Recursive JEPA engineering checkpoint' not in current:
        anchor = '## Current successor checkpoint'
        current = current.replace(anchor, note + anchor, 1)
        status.write_text(current, newline='\n', encoding='utf-8')
    for name in ('SPRINT.md', 'decision.md'):
        path = ROOT / name
        current = path.read_text(encoding='utf-8')
        if '## Recursive JEPA engineering checkpoint' not in current:
            path.write_text(current.rstrip() + '\n\n' + note, newline='\n', encoding='utf-8')
    paper = ROOT / 'paper/main.tex'
    current = paper.read_text(encoding='utf-8')
    if r'\subsection{Recursive label-memory engineering prototype}' not in current:
        section = r'''
\subsection{Recursive label-memory engineering prototype}
We implemented a separate6,880-parameter encoder/refiner with three tied steps. Source/target labeled support class prototypes and current residual memories refine nonlinear query embeddings; bounded support-gradient head updates receive outer meta-query backpropagation. A stop-gradient EMA encoder supplies representation targets across nested additional missingness views, using only originally observed values and support-label context. No query label enters the teacher target or inference. The auxiliary is JEPA-inspired and support-conditioned, rather than fully self-supervised\cite{ijepa,vicreg,maml}. Finite unrolling and a bounded correction do not guarantee contraction, calibrated predictions, monotonic improvement or safety; exact frozen output is available as an intervention, not yet a validated gate.

An engineering-only Modal A100 pilot trains the full model, supervised-only control and one-step control for64 updates each, with no score-based selection. All have6,880 trainable parameters; their effective auxiliary training, deep supervision and recurrent compute differ. The full auxiliary MSE falls1.036854 to .003989, but student mean standard deviation remains .029807 against variance floor1, and recurrent query spread is .015229 on the logged batch. Diagnostic full/frozen NLL is .616819/.617999 for sign flip, .592116/.593328 for nonlinear shift, .600430/.601214 for no shift and .646807/.647985 for ignorable shift. Control neural scores are nearly identical; support logistic reaches .503530 on sign flip. These two correlated synthetic worlds establish neither usefulness nor anti-collapse performance; no efficacy gate was run or passed. Independent audit replays all48 checkpoint predictions and verifies96 probability arrays. The next gate concerns discriminative embedding geometry, matched computational controls and a support-validated frozen fallback, followed by fresh development before any confirmation.

'''
        current = current.replace(r'\section{Controlled data and oracle}', section + r'\section{Controlled data and oracle}', 1)
        current = current.replace(r'Current conservative provisions are \$14.05', r'Current conservative provisions including the subsequent engineering pilot are \$14.85', 1)
        bibliography = r'''
\bibitem{ijepa}
M. Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. 2023.
\url{https://arxiv.org/abs/2301.08243}.
\bibitem{vicreg}
A. Bardes, J. Ponce, and Y. LeCun. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. 2021.
\url{https://arxiv.org/abs/2105.04906}.
\bibitem{maml}
C. Finn, P. Abbeel, and S. Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. 2017.
\url{https://arxiv.org/abs/1703.03400}.
'''
        current = current.replace(r'\end{thebibliography}', bibliography + r'\end{thebibliography}', 1)
        paper.write_text(current, newline='\n', encoding='utf-8')
    stack = []
    for action, name in re.findall(r'\\(begin|end)\{([^}]+)\}', current):
        if action == 'begin':
            stack.append(name)
        elif not stack or stack.pop() != name:
            raise ValueError('LaTeX environments do not balance')
    if stack:
        raise ValueError('Unclosed environment')
    citations = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', current) for key in group.split(',')}
    if not citations <= set(re.findall(r'\\bibitem\{([^}]+)\}', current)):
        raise ValueError('Unresolved citation')
    for protocol in ('bridge_v1', 'bridge_v2', 'recursive_jepa_v1'):
        for name, digest in json.loads((ROOT / f'configs/{protocol}.json').read_text(encoding='utf-8'))['file_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
