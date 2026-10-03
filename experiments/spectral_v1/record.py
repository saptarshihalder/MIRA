import hashlib,json,shutil,re
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    source=ROOT/'artifacts/runs/spectral_v1_gpu/results';target=ROOT/'artifacts/reports/spectral_v1_gpu'
    audit=json.loads((target/'audit.json').read_text());assert audit['passed']
    report=json.loads((source/'report.json').read_text())
    for file in source.rglob('*'):
        if file.is_file():
            destination=target/file.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,destination)
    apps=json.loads((ROOT/'artifacts/manifests/modal_apps_after_spectral.json').read_text(encoding='utf-8-sig'))
    app=next(x for x in apps if x['description']=='mira-spectral-v1');assert app['tasks']=='0'
    resource=json.loads((ROOT/'artifacts/manifests/modal_billing_after_spectral.json').read_text(encoding='utf-8-sig'))
    ledger=ROOT/'artifacts/manifests/compute_ledger.json';entries=json.loads(ledger.read_text())
    for entry in entries:
        if entry['run_id']=='spectral_v1_gpu':
            cost=[r for r in resource if r['object_id']==app['app_id']]
            entry.update(phase='spectral_development',app_id=app['app_id'],active_tasks=0,gpu_verified=report['gpu'],cuda_updates_per_trained_model=500,provider_reported_resource_cost_usd=float(sum((Decimal(r['cost']) for r in cost),Decimal(0))) if cost else None,provider_cost_status='Report may lag; unavailable is not zero; final invoice unverified and full.80 retained')
    ledger.write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8',newline='\n')
    note='''## Spectral learned-operator checkpoint — October 3

Read docs/SPECTRAL_FINDINGS.md. A257-parameter learned mode filter and scalar control each completed500 CUDA updates on Modal A100; fresh development8seeds x2widths x4regimes. Neural selectivity improves sign-flip NLL over the trained scalar by.019314 [.004884,.033744] and nonlinear NLL over support logistic by.010659 [.004019,.017299]. However frozen wins on nonlinear and ignorable shifts; ignorable harm.013053 [.006387,.019719]. Full development gate FAILED. Intervals are exploratory and unadjusted. Support-objective descent holds under the stated convex bound and passed numerical checks; it is not test-risk safety. All384 score arrays/192 CPU predictions audit/replay pass. Protocol8383e139 committedf00fce3 before GPU. Artifacts/reports/spectral_v1_gpu includes full predictions/checkpoints.

JEPA diagnosis: solved head on learned pooled features worsens sign-flip NLL to.637938, versus original.616819 and random-solved.630923. The next architecture candidate is a cross-fitted label-response operator: trainable field interactions plus learned spectral gates informed by disjoint support gradient agreement. First test free CPU discrimination of harmful modes; then commit fresh label/compute-matched protocol. See SPECTRAL_FINDINGS for closest prior art, constraints and failure cases. No claim of unique novelty,100% safety or venue readiness; confirmation98000-98019 untouched.

Budget:15.65 provisions +3 protected reproduction leaves1.35 within20. Latest job complete/zero active tasks, no retry; provider charges may lag, full.80 retained. Existing native LaTeX compiler environment failure persists until a successful check. No email/submission.

'''
    for name in ('STATUS.md','SPRINT.md','decision.md'):
        file=ROOT/name;text=file.read_text(encoding='utf-8')
        if '## Spectral learned-operator checkpoint' not in text:
            if name=='STATUS.md':text=text.replace('## Recursive JEPA engineering checkpoint',note+'## Recursive JEPA engineering checkpoint',1)
            else:text=text.rstrip()+'\n\n'+note
            file.write_text(text,encoding='utf-8',newline='\n')
    paper=ROOT/'paper/main.tex';text=paper.read_text(encoding='utf-8')
    if r'\subsection{Learned spectral adaptation}' not in text:
        paragraph=r'''
\subsection{Learned spectral adaptation}
A subsequent257-parameter trained filter learns which source/target support-error eigenmodes to update. With fixed declared value--mask features, let $L(w)$ be offset logistic support loss plus $.05\|w\|^2/2$. Its Hessian is bounded by $H=.25A^\top A/n+.05I=V\operatorname{diag}(h)V^\top$. A learned gate $a_i\in[0,1]$ gives $B=V\operatorname{diag}(a_i/h_i)V^\top$ and $w'=w-B\nabla L(w)$. Since $0\preceq B\preceq H^{-1}$, the quadratic upper bound implies $L(w')\le L(w)-\nabla L(w)^\top B\nabla L(w)/2$. This is support-objective descent in exact arithmetic, not a test-risk guarantee. The inner basis is handcrafted; the shared error-mode filter is learned. Differentiable optimization and learned optimizers with guarantees have established precedents\cite{metaoptnet,learnedopt,pacopt}; novelty of this particular missingness use remains unresolved.

The filter and a scalar control each receive500 Modal A100 CUDA updates, followed by fresh development on eight latent seeds and two widths. The filter gains .019314 [.004884,.033744] NLL over the scalar on sign flip and .010659 [.004019,.017299] over support logistic on nonlinear shift. However its nonlinear and ignorable losses exceed frozen by .009441 and .013053 respectively; the latter harm interval is [.006387,.019719]. Therefore the full development gate fails. These are unadjusted exploratory seed-level t intervals, with widths averaged within seed. All384 probability arrays and192 CPU checkpoint replays pass audit. A proposed next design combines learned field interactions with disjoint-support gradient agreement, preserving the inner descent condition while testing generalization separately. It is untested. Current provisions are \$15.65 plus \$3 protected for reproduction within \$20; provider invoices remain unverified.

'''
        text=text.replace(r'\section{Controlled data and oracle}',paragraph+r'\section{Controlled data and oracle}',1)
        text=text.replace(r'Current conservative provisions including the subsequent engineering pilot are \$14.85',r'At the recursive pilot checkpoint, conservative provisions were \$14.85',1)
        refs=r'''
\bibitem{learnedopt} L. Metz et al. Understanding and correcting pathologies in the training of learned optimizers. ICML,2019. \url{https://proceedings.mlr.press/v97/metz19a.html}.
\bibitem{pacopt} M. Sucker and P. Ochs. PAC-Bayesian Learning of Optimization Algorithms. AISTATS,2023. \url{https://proceedings.mlr.press/v206/sucker23a.html}.
'''
        text=text.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}',1)
        paper.write_text(text,encoding='utf-8',newline='\n')
    citations={k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert citations<=set(re.findall(r'\\bibitem\{([^}]+)\}',text)),citations-set(re.findall(r'\\bibitem\{([^}]+)\}',text))
    files={str(f.relative_to(target)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in target.rglob('*') if f.is_file() and f.name!='manifest.json'}
    (target/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(audit_pass=True,gate_pass=audit['development_gate_pass'],files=len(files),provisions=15.65)))
if __name__=='__main__':main()
