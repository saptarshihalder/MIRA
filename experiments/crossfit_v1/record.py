"""Curate the recovered evidence without changing frozen training sources."""
import hashlib,json,re,shutil,zipfile
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    recovered=ROOT/'artifacts/runs/crossfit_v1_recovered';target=ROOT/'artifacts/reports/crossfit_v1_gpu';original=ROOT/'artifacts/runs/crossfit_v1_gpu'
    report=json.loads((recovered/'report.json').read_text());audit=json.loads((target/'audit.json').read_text());assert audit['passed']
    for file in recovered.rglob('*'):
        if file.is_file():
            destination=target/file.relative_to(recovered);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,destination)
    with zipfile.ZipFile(target/'original_gpu_run.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in original.rglob('*'):
            if file.is_file():archive.write(file,file.relative_to(original))
    apps=json.loads((ROOT/'artifacts/manifests/modal_apps_after_crossfit.json').read_text(encoding='utf-8-sig'));app=next(a for a in apps if a['description']=='mira-crossfit-v1');assert app['tasks']=='0'
    resources=json.loads((ROOT/'artifacts/manifests/modal_billing_after_crossfit.json').read_text(encoding='utf-8-sig'));rows=[r for r in resources if r['object_id']==app['app_id']]
    ledger=ROOT/'artifacts/manifests/compute_ledger.json';entries=json.loads(ledger.read_text());entry=next(e for e in entries if e['run_id']=='crossfit_v1_gpu')
    assert entry['status']=='failed'
    entry.update(app_id=app['app_id'],active_tasks=0,training_completed=True,cuda_updates_per_trained_model=500,evaluation_recovered_on_cpu=True,recovery_path=str(recovered),provider_reported_resource_cost_usd=float(sum((Decimal(r['cost']) for r in rows),Decimal(0))) if rows else None,provider_cost_status='Final invoice unverified; missing resource rows mean unavailable; full.80 retained despite CPU recovery',failure='Platt L-BFGS-B ABNORMAL after all training and132predictions; no remote retry')
    ledger.write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8',newline='\n')
    note='''## Support-verified query operator — October 4

Read docs/CROSSFIT_FINDINGS.md first. A370-parameter query-conditioned trained correction now PASSES the prespecified exploratory synthetic robustness gate on20fresh latent seeds/two widths/all four regimes. Sign-flip gain.045838 [.034527,.057149] over frozen; .032354 [.020077,.044631] over same-rule guarded logistic; .018036 [.008201,.027872] over no-query learner. All80ignorable branch decisions reject adaptation. Nonlinear harm.0000123 is small. However gain.006202 [-.002545,.014948] versus the114parameter scalar/query learner is uncertain; the stronger neural-specific gate FAILS. Ordinary full-support logistic still wins on sign flip while harming null cases. No universal safety, novelty, native utility or venue-readiness claim. Intervals are unadjusted exploratory; confirmation98000-98019 untouched.

Three500-update models trained on Modal CUDA/A100 allocation. Evaluation failed at the Platt numerical control after132cells; all checkpoints preserved. Committed CPU-only recovery solved the SAME convex calibration objective with gradient<8.33e-11; no retrain/remote retry/seed changes. Original GPU run stays FAILED in ledger. All160cells completed;1,600scores,960CPU prediction-array replays,960split decisions and640identity arrays audit pass. Original132GPU neural files agree within1.87e-9. Training traces/exact device-name log were lost at late evaluation abort, disclosed. Artifacts/reports/crossfit_v1_gpu includes original failed-run archive and recovered full predictions.

Next: resolve capacity/task-conditioned scalar and support-only-head controls, then freeze a finite independent replication and native-transfer protocol. Current learned field representation is handcrafted; optional learned interactions need evidence against matched features. See CROSSFIT_FINDINGS. Do not add seeds until significance or spend on another architecture without an unresolved-control rationale. Budget16.45 provisions +3 reproduction leaves.55 under20. Latest app tasks0; bills may lag. Same paper/editor updated; native PDF compiler environment remains unresolved. No email/submission.

'''
    for name in ('STATUS.md','SPRINT.md','decision.md'):
        path=ROOT/name;text=path.read_text(encoding='utf-8')
        if '## Support-verified query operator' not in text:
            text=text.replace('## Spectral learned-operator checkpoint',note+'## Spectral learned-operator checkpoint',1) if name=='STATUS.md' else text.rstrip()+'\n\n'+note
            if name=='STATUS.md':text='# MIRA status — October 4, 2026\n'+text.split('\n',1)[1]
            path.write_text(text,encoding='utf-8',newline='\n')
    paper=ROOT/'paper/main.tex';text=paper.read_text(encoding='utf-8')
    if r'\section{Support-verified query correction}' not in text:
        text=text.replace(r'\title{Training Missingness Corrections for Frozen Tabular Predictors}',r'\title{Learning Support-Verified Corrections for Frozen Tabular Predictors}').replace(r'\date{3 October 2026}',r'\date{4 October 2026}')
        start=text.index(r'\fbox{');end=text.index(r'\end{center}',start)
        text=text[:start]+r'\fbox{\parbox{0.93\columnwidth}{\small\textbf{Working draft.} A trained query correction passes a fresh synthetic development robustness gate. Its full architectural advantage, native transfer and novelty remain unresolved; the manuscript is not submission-ready.}}'+'\n'+text[end:]
        abstract=r'''We study how a trained correction can adapt a frozen tabular predictor while deferring when labeled support does not justify the change. A370-parameter operator combines source/target-conditioned spectral updates with a query-specific attenuation head. Two branches each fit64 target labels and verify on the disjoint64, then average their predictions. Training uses a smooth verification surrogate; inference applies a fixed nominal verification rule. On20 fresh synthetic development seeds and two widths, sign-flip log-loss improves by .04584 [.03453,.05715] over frozen and .03235 [.02008,.04463] over logistic regression with the same verification rule. All80 ignorable-shift branch decisions reject adaptation, and the predeclared exploratory robustness gate passes. Removing the query head loses .01804 [.00820,.02787] on sign flips. However, superiority over a trained scalar-filter/query-head control is uncertain, and unverified full-support logistic remains more accurate on sign flips while harming ignorable cases. These unadjusted seed-level intervals support a conditional adaptation/rejection benefit, not universal dominance or a distribution-free guarantee. Native evaluation of this operator and a distinct architectural novelty claim remain open. Checkpoints, full predictions, prior negative results and a disclosed numerical-control recovery are retained.'''
        a=text.index(r'\begin{abstract}')+len(r'\begin{abstract}');b=text.index(r'\end{abstract}',a);text=text[:a]+'\n'+abstract+'\n'+text[b:]
        a=text.index(r'\section{Question and scope}');b=text.index(r'\section{Information equivalence}',a)
        section=r'''\section{Question and scope}
The primary candidate is a trained support-verified query operator. We ask whether it can retain useful missingness corrections while rejecting updates that fail to generalize beyond their fitting labels. The present positive evidence is controlled synthetic development with a frozen logistic backbone, not native-data or foundation-model confirmation. Missingness adaptation and validation-guided meta-learning have established precedents\cite{zhou2023,rockenschaub2024,ren2018,metaweight}; a combination of these ingredients alone does not establish novelty. Earlier compiler, mixture, representation and native-data experiments are retained as supporting evidence and failure analysis, not as validation of this new operator.

\section{Support-verified query correction}
Source support contains128 labels. Target support is split by fixed even/odd indices into two64-label sets. Each branch fits the six-step spectral correction on one set and verifies the complete candidate on the other; roles are then reversed. The257-parameter mode filter is supplemented with a113-parameter query head. Its inputs are the frozen logit, bounded signed correction, correction magnitude, curvature score and source/target residual-mean difference. It produces a probability mixture of the correction and frozen predictor. All context statistics use labeled support only. The value--mask interaction basis is handcrafted. No query label enters fitting, query features or verification.

For verification examples define paired improvement $d_i=\ell(p_0,y_i)-\ell(p_\theta,y_i)$. The branch is accepted when $\bar d>1.645s_d/\sqrt{64}$, otherwise it returns $p_0$. Branch probabilities are averaged. This is a fixed nominal rule, not a finite-sample or simultaneous safety guarantee; the branches are dependent because support roles are reversed. During meta-training, query BCE receives equal weight from an unguarded ensemble and a smooth surrogate with acceptance weight $\sigma((\bar d-1.645\widehat{\mathrm{SE}})/(\widehat{\mathrm{SE}}+.005))$. The training/inference difference requires a later ablation. The inner convex-objective descent bound applies to the fitting solver, not the final mixture or query risk.

Three models receive500 CUDA updates with identical episodes and batch schedules: the370-parameter query operator,257-parameter no-query-head operator, and114-parameter scalar-filter/query-head control. Ordinary interaction logistic receives the same fitting/verification split and rule. Capacity-matched, task-conditioned scalar and support-only-head controls remain necessary. Development uses20 fresh latent seeds, widths6/10 and four regimes; widths average within seed before unadjusted95\% t intervals. The separate confirmation seeds are untouched.

\begin{table}[t]
\centering\small
\caption{Fresh synthetic development: lower NLL is better. These are adaptive development results, not confirmation.}
\begin{tabular}{lrrrr}
\toprule
Regime & Frozen & Query operator & Guarded logistic & Scalar/query\\
\midrule
Sign flip & .599337 & .553499 & .585853 & .559700\\
Nonlinear shift & .597942 & .597954 & .597942 & .598434\\
No shift & .598576 & .555911 & .587586 & .568979\\
Ignorable shift & .640957 & .640957 & .640801 & .640957\\
\bottomrule
\end{tabular}
\end{table}

The query operator passes the planned synthetic robustness gate: sign-flip gain over frozen is .045838 [.034527,.057149], while the two null upper harm limits remain below .001. Its sign-flip gain over guarded logistic is .032354 [.020077,.044631], and over the no-query learner .018036 [.008201,.027872]. However gain .006202 [-.002545,.014948] over the scalar/query learner is uncertain, so the stronger neural-specific gate fails. Full-support unverified logistic scores .511657 on sign flip but .684695 on ignorable shift: the method offers a different adaptation/rejection tradeoff, not predictive dominance. The main model rejects all80 ignorable branch updates. Nonlinear harm .0000123 also remains small on this panel; general robustness is unproven.

The paid run completed training but failed during evaluation when the Platt control's L-BFGS-B optimizer terminated abnormally after132 cells. A committed CPU-only recovery uses damped exact Newton for the identical convex objective and ridge, with maximum gradient $8.33\times10^{-11}$. All160 cells and original checkpoints are retained;132 original GPU neural files replay within $1.87\times10^{-9}$. No model retraining, cloud retry, seed replacement or case exclusion occurred. Per-update training traces and the exact device-name log were not persisted before the abort; the frozen CUDA-only worker/checkpoint placement and changed checkpoint states establish the completed updates. Independent checks reproduce1,600 score arrays,960 CPU output arrays and960 split decisions. Current conservative provisions are \$16.45 plus \$3 protected for reproduction within \$20; the failed run remains counted. This candidate has no native-data efficacy or venue-readiness claim.

'''
        text=text[:a]+section+text[b:]
        bibliography=r'''
\bibitem{ren2018} M. Ren, W. Zeng, B. Yang, and R. Urtasun. Learning to Reweight Examples for Robust Deep Learning. ICML,2018. \url{https://proceedings.mlr.press/v80/ren18a.html}.
\bibitem{metaweight} J. Shu et al. Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting. NeurIPS,2019. \url{https://proceedings.neurips.cc/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html}.
'''
        text=text.replace(r'\end{thebibliography}',bibliography+r'\end{thebibliography}',1)
        text=text.replace(r'Current provisions are \$15.65',r'At the earlier spectral checkpoint, provisions were \$15.65')
        paper.write_text(text,encoding='utf-8',newline='\n')
    manifest={str(f.relative_to(target)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in target.rglob('*') if f.is_file() and f.name!='manifest.json'}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(robustness_gate=audit['robustness_development_gate'],neural_gate=audit['neural_specific_signflip_gate'],artifact_files=len(manifest),provisions=16.45)))
if __name__=='__main__':main()
