"""Curate actual GPU evidence and update the existing research notes/manuscript."""
import hashlib
import json
import shutil
from pathlib import Path
from decimal import Decimal
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write(path, text):
    path.write_text(text, encoding='utf-8', newline='\n')


def main():
    summaries = {}
    for version in (1, 2):
        raw = ROOT / f'artifacts/runs/bridge_v{version}_a100/results'
        target = ROOT / f'artifacts/reports/bridge_v{version}_a100'
        target.mkdir(parents=True, exist_ok=True)
        report = json.loads((raw / 'report.json').read_text())
        audit = json.loads((target / 'independent_audit.json').read_text())
        if not audit['passed']:
            raise ValueError('Independent audit must pass before publication of scores')
        primary = 'bridge' if version == 1 else 'contextual_regularizer'
        regimes = ('sign_flip', 'nonlinear_shift', 'no_shift', 'ignorable_shift')
        means = {g: {m: float(np.mean([r['metrics'][m]['nll'] for r in report['results'] if r['regime'] == g]))
                     for m in report['results'][0]['metrics']} for g in regimes}
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((raw / 'predictions').glob('*.npz'))}
        for path in raw.glob('*'):
            if path.is_file():
                shutil.copy2(path, target / path.name)
        summary = dict(version=version, primary=primary, gate_pass=report['gate_pass'],
                       regime_mean_nll=means, shifted_nll_gains=report['shifted_nll_gains'],
                       null_harms=report['null_harms'], gpu=report['gpu'], updates=report['updates'],
                       parameters=report['training_models'][primary]['parameters'],
                       protocol_sha256=report['protocol_sha256'], independent_audit=audit,
                       statistical_units=12, development_files=144, validation_files=48,
                       scope='Synthetic adaptive development, not confirmation or native/causal efficacy')
        write(target / 'prediction_manifest.json', json.dumps(manifest, indent=2) + '\n')
        write(target / 'summary.json', json.dumps(summary, indent=2) + '\n')
        lines = [f'# Bridge v{version} A100 development', '', f"Main model: {primary}; {summary['parameters']} trained parameters, {summary['updates']} CUDA updates. Gate passed: **{report['gate_pass']}**.", '',
                 'Four validation and twelve development latent seeds; widths6/10/14 and four regimes. Average widths/regimes within seed before paired bootstrap. These are controlled, retrospectively label-dependent masks; no natural-data claim.', '',
                 '| Model | Sign flip NLL | Nonlinear shift | No shift | Ignorable shift |', '|---|---:|---:|---:|---:|']
        for model in means['sign_flip']:
            lines.append('| ' + model + ' | ' + ' | '.join(f'{means[g][model]:.6f}' for g in regimes) + ' |')
        lines.extend(['', 'Paired shifted gains (control minus primary):', ''])
        for model, effect in report['shifted_nll_gains'].items():
            lines.append(f"- {model}: {effect['mean']:.6f} [{effect['low']:.6f}, {effect['high']:.6f}].")
        lines.extend(['', 'Null harm (primary minus frozen):', ''])
        for regime, effect in report['null_harms'].items():
            lines.append(f"- {regime}: {effect['mean']:.6f} [{effect['low']:.6f}, {effect['high']:.6f}].")
        lines.extend(['', f"Independent audit: {audit['probability_arrays_checked']} score arrays and all observed data/masks/labels/row boundaries; CPU replay on {audit['cpu_checkpoint_predictions']} checkpoint predictions. Full prediction archives are retained locally under ignored artifacts/runs; prediction_manifest.json identifies every file.", '',
                      f'Reproduce: python experiments/bridge_v{version}/run.py --config configs/bridge_v{version}.json --out NEW_DIRECTORY --device cuda (or cpu). Protocol and source hashes are frozen; never overwrite an existing run.', '',
                      'Four-step v2 solver is not claimed converged; full support logistic is converged. Meta-regularization/optimization layers are established prior art. Venue readiness and universal safety remain unresolved.'])
        write(target / 'report.md', '\n'.join(lines) + '\n')
        summaries[version] = summary

    manifest_root = ROOT / 'artifacts/manifests'
    resources = json.loads((manifest_root / 'modal_billing_after_bridge_v2.json').read_text(encoding='utf-8-sig'))
    apps = json.loads((manifest_root / 'modal_apps_after_bridge_v2.json').read_text(encoding='utf-8-sig'))
    ledger = manifest_root / 'compute_ledger.json'
    entries = json.loads(ledger.read_text())
    for version in (1, 2):
        app = next(a for a in apps if a['description'] == f'mira-bridge-v{version}-a100')
        if int(app['tasks']) != 0:
            raise ValueError('GPU tasks still active')
        cost = sum(Decimal(r['cost']) for r in resources if r['object_id'] == app['app_id'])
        entry = next(e for e in entries if e['run_id'] == f'bridge_v{version}_a100')
        entry.update(app_id=app['app_id'], provider_reported_resource_cost_usd=float(cost),
                     provider_report_source='artifacts/manifests/modal_billing_after_bridge_v2.json',
                     provider_cost_status='Reported resource cost; final invoice unverified, full $.80 provision retained',
                     gpu_verified=summaries[version]['gpu'], cuda_updates_per_trained_model=summaries[version]['updates'])
    write(ledger, json.dumps(entries, indent=2) + '\n')
    reserved = sum(float(e['reserved_usd']) for e in entries)
    policy = json.loads((manifest_root / 'compute_policy.json').read_text())
    if reserved + policy['reproduction_reserve_usd'] > policy['total_cap_usd']:
        raise ValueError('Budget policy violated')
    s2 = summaries[2]
    if '\\subsection{Trainable support-conditioned successors}' in (ROOT / 'paper/main.tex').read_text():
        print(json.dumps(dict(curated=True, manuscript_already_updated=True, provisions=reserved)))
        return
    observation = ('Bridge v1 fails; v2 support-conditioned regularization has a real controlled shifted gain over support logistic '
                   f"{s2['shifted_nll_gains']['support_logistic']['mean']:.6f} "
                   f"[{s2['shifted_nll_gains']['support_logistic']['low']:.6f},{s2['shifted_nll_gains']['support_logistic']['high']:.6f}] "
                   'and target-only conditioning .003300 [.000862,.005632]. However it harms the ignorable null by .060856 [.050163,.070476] '
                   'and nonlinear-shift NLL .619693 exceeds frozen .594366. The aggregate shifted gain does not establish robustness. Both gates fail; neither candidate advances to native confirmation.')
    checkpoint = ('## Current successor checkpoint — October 3\n\n' + observation + '\n\n'
        '- Actual Modal A100 training: v1 five800-update fits, v2 four600-update fits and train-only global-grid selection. Reports/checkpoints/audits are in artifacts/reports/bridge_v1_a100 and bridge_v2_a100. Both deployed apps have zero active tasks and no schedule/endpoint. No automatic remote retries. A pre-dispatch source guard caught concurrent reporting cleanup; the final protocol was recommitted before any GPU call.\n'
        f'- Budget: ${reserved:.2f} conservative provisions plus $3 reproduction reserve within $20; ${20-reserved-3:.2f} unallocated. Resource billing reconciles stopped oversized failures while retaining $.50 each. Each new GPU phase retains $.80; reported charges remain unverified as final invoices.\n'
        '- Next: read docs/BRIDGE_NEXT_GATE.md. Do not repeat either closed candidate or treat adaptive development as confirmation. Diagnose frozen-fallback capacity and nonlinear feature expressivity on CPU before a new trained protocol. Seeds98000-98019 remain untouched. Neural-specific conditional benefit is now observed synthetically; general native utility, architecture novelty and venue readiness remain unresolved.\n\n')
    status = ROOT / 'STATUS.md'
    old = status.read_text()
    old = old.replace('## Current checkpoint: read this before continuing', '## Earlier native checkpoint (historical budget; retained evidence)', 1)
    marker = '## Earlier native checkpoint (historical budget; retained evidence)'
    write(status, old.replace(marker, checkpoint + marker, 1))
    for filename in ('SPRINT.md', 'decision.md', 'docs/POLICY_ADAPTER_NEXT_GATE.md'):
        path = ROOT / filename
        write(path, path.read_text().rstrip() + '\n\n## Trained successor update — October 3\n\n' + observation +
              f'\n\nProvisions ${reserved:.2f} plus protected reproduction $3 within $20. Current next gate: docs/BRIDGE_NEXT_GATE.md. No completed phase is repeated; all negatives and confirmation separation persist.\n')

    tex = ROOT / 'paper/main.tex'
    source = tex.read_text()
    original_intro = 'Our primary candidate is MIRA-Compiler, a trained selector of reversible missingness coordinates for a frozen predictor.'
    source = source.replace(original_intro, 'Our current trainable candidate learns source/target-support-conditioned regularization for a differentiable logistic correction. Its controlled shifted benefit is qualified by substantial null and nonlinear-regime failures. Earlier MIRA-Compiler is a trained selector of reversible missingness coordinates for a frozen predictor.', 1)
    source = source.replace('Latest total Modal cap is \\$20, with conservative reservations \\$16.95',
                            'At that native checkpoint, the total Modal cap was \\$20, with conservative reservations \\$16.95', 1)
    source = source.replace('General native-missingness benefit, identified acquisition-policy transport and venue readiness remain unresolved.',
                            'A later learned support regularizer improves averaged controlled-shift losses over strong controls, but harms ignorable missingness and nonlinear-shift cases. General native-missingness benefit, identified acquisition-policy transport and venue readiness remain unresolved.', 1)
    section = r'''
\subsection{Trainable support-conditioned successors}
The fixed candidate mixture motivated a variable-width query-conditioned bridge with 20,801 parameters, bounded shared potentials, and correction
\begin{equation}
 z_{\mathrm{target}}(q)=z_{\mathrm{frozen}}(q)+h_{\theta}(q,S_t)-h_{\theta}(q,S_s).
\end{equation}
Potential subtraction guarantees identity, reversal and composition only at the same observed query. It does not guarantee correct adaptation. Five models received 800 CUDA updates on controlled development tasks. Shifted gain over frozen was .001602 [-.001330,.004660]; support interaction logistic was better by .027011 [.016530,.037520], and target-only conditioning by .003339 [.000681,.005855]. The frozen gate failed.

A separately frozen revision fits target labels explicitly while learning coordinate-specific positive regularization from source and target support moments. A shared 1,857-parameter MLP outputs penalties for the basis containing intercept, frozen logit, observed values, masks and all ordered value--mask products. Four differentiable float64 Newton updates approximate
\begin{equation}
 \min_b\;\frac{1}{n_t}\sum_{i\in S_t}\ell(z_i+\psi_i^\top b,y_i)
 +\frac{1}{2}\sum_j\lambda_{\omega,j}(S_s,S_t)b_j^2.
\end{equation}
Prediction is $\sigma(z_q+\psi_q^\top b^{(4)})$. No query label or mechanism coefficient conditions prediction; synthetic labels enter the evaluator's observation generator and meta-training loss. This is controlled retrospective masking, not natural missingness. The explicit interaction basis is shared by every solver control and cannot be credited as a learned feature representation. Optimization layers and learned regularization have established precedents~\cite{r2d2,metaoptnet,metareg}; the support-conditioning advantage is the candidate contribution under test.

Four learned variants receive the same episodes and 600 updates: source/target-conditioned, target-only-conditioned, feature-type-only, and global-scalar penalties. The type-only network has the same stored size but effectively emits five type penalties. Fixed .1 regularization and eleven global penalties selected using training meta-query loss only are additional controls. Source-fit, support, meta-query, validation and development generator streams are disjoint. Four validation and twelve development latent seeds each span widths6/10/14 and four regimes. Bootstrap intervals resample the twelve seeds after averaging correlated widths and regimes. Width14 is structural extrapolation, not an external dataset.

\begin{center}
\begin{tabular}{lrrrr}
\toprule
Model & Sign flip & Nonlinear & No shift & Ignorable\\
\midrule
Frozen & .600367 & .594366 & .600610 & .661772\\
Support interaction logistic & .491318 & .624173 & .496648 & .719699\\
Fixed/train-grid regularizer & .498337 & .608472 & .500623 & .707019\\
Target-only learned & .458689 & .613399 & .456088 & .728806\\
Source/target learned & .445794 & .619693 & .444034 & .722628\\
\bottomrule
\end{tabular}
\end{center}
Lower NLL is better. Averaged shifted gain is .025002 [.021761,.028315] over support logistic, .020661 [.017715,.023771] over fixed/train-grid regularization, and .003300 [.000862,.005632] over the target-only learner. These are conditional synthetic gains. Ignorable-null harm is .060856 [.050163,.070476], far above the predeclared .001 limit, and nonlinear-shift loss exceeds frozen. Thus the full gate fails; a positive aggregate does not establish robustness or identified policy transport. No natural-data or confirmation extension follows this result. All outcomes, source hashes, checkpoints and probabilities are retained. Both versions trained on NVIDIA A100-SXM4-40GB; independent audits reconstruct all scores and data boundaries and replay checkpoints. The four-step solver reports numerical gradients without a convergence guarantee. Both candidate protocols preceded their respective GPU scores, and the second was explicitly motivated by the first failed development result. Current conservative provisions are \$14.05 with \$3 protected for reproduction within the \$20 total cap; reported resource charges are distinct from unverified final invoices.

'''
    source = source.replace('\\section{Controlled data and oracle}', section + '\\section{Controlled data and oracle}', 1)
    references = r'''
\bibitem{r2d2}
L. Bertinetto, J. F. Henriques, P. Torr, and A. Vedaldi.
Meta-learning with differentiable closed-form solvers. ICLR, 2019.
\url{https://arxiv.org/abs/1805.08136}.
\bibitem{metaoptnet}
K. Lee, S. Maji, A. Ravichandran, and S. Soatto.
Meta-learning with differentiable convex optimization. CVPR, 2019.
\url{https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html}.
\bibitem{metareg}
Y. Balaji, S. Sankaranarayanan, and R. Chellappa.
MetaReg: Towards domain generalization using meta-regularization. NeurIPS, 2018.
\url{https://proceedings.neurips.cc/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html}.
'''
    source = source.replace('\\end{thebibliography}', references + '\\end{thebibliography}', 1)
    write(tex, source)
    print(json.dumps(dict(v1_gate=summaries[1]['gate_pass'], v2_gate=summaries[2]['gate_pass'],
                         provisions=reserved, updated_manuscript=str(tex))))


if __name__ == '__main__':
    main()
