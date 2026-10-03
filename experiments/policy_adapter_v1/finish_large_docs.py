"""Update current notes from the audited large GPU report; retain history."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def write(path,text):
    path.write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':
    status=ROOT/'STATUS.md'
    text=status.read_text(encoding='utf-8')
    prefixes=('- **Latest user request and cap:**','- **Larger protocol frozen, scores pending:**')
    text='\n'.join(line for line in text.splitlines() if not line.startswith(prefixes))+'\n'
    marker='## Current checkpoint: read this before continuing\n'
    update='''

- **Latest Modal cap and access:** $20 total supersedes $26. Authoritative policy: compute_policy.json. Conservative reservations $16.95 across 26 calls preserve $3 for reproduction and leave $0.05 unallocated. Provider UI displays $0.11 for the three new attempts, rounded and not a verified final invoice; reservations remain retained. Successful deployed app mira-large-native-a100-final has zero active GPU containers, no schedule and no endpoint. Failed apps were stopped. No new paid science before reconciling charges or changing the cap; never spend reproduction funds on expansion.
- **Larger A100 development complete:** artifacts/reports/large_native_a100_v3/ and configs/large_native_v3.json. Four official 2024/2025 files contain 1,016,280 raw records; latest cohorts were released during 2026. Nine A100-SXM4-40GB trials across three tasks/seeds train 50,000 CUDA source labels each plus separate meta-training rows. Each neural/Adam-linear fit receives 800 CUDA updates. Held-out 2025 regions include ten states per health target and six police-force groups. Two domains, correlated health tasks and shared seeds are not nine independent datasets.
- **Larger result:** frozen/neural NLL .333466/.333545 diabetes, .395842/.395610 asthma, .611319/.603685 road. Road support Platt .603197 and global simplex .603348 perform better than neural; matched linear .603648 is similar. Road gain over frozen .007634 has conditional region interval [-.001617,.016885]. Distinct neural novelty remains unsupported. Questionnaire/state composition and reporting-specification changes confound causal interpretation.
- **Larger audit/failures:** all 1,470 probability arrays, NumPy checkpoints, prepared labels and group/year boundaries pass (NLL discrepancy <=7.05e-8; predictions <=2.02e-7). All nine contextual linear fits attain 1e-7 gradient tolerance; worst simplex KKT residual 2.01e-6. Nine boundary/numerical tests pass. Preserve calibration failure (four complete trials/partial fifth), equivalent Newton solver repair and missing-mount failure before child training. Final complete mount finishes all nine trials in 62.020 seconds. No automatic retries or results-based scientific selection. CUDA training is verified; host/device prediction warning remains.

The entries below retain the earlier Diabetes and synthetic checkpoint; current budget and larger outcome above supersede their historical limits.
'''
    if '- **Latest Modal cap and access:**' not in text:
        text=text.replace(marker,marker+update,1)
    text=text.replace('Next: compare converged linear-mixture and global/support calibration controls with identical native meta-label access. Then justify',
                      'Next: converged linear, convex stacking and calibration controls are complete; larger results do not establish a neural advantage. Use free CPU diagnostics and targeted prior-art review to justify')
    write(status,text)
    sprint=ROOT/'SPRINT.md'
    text=sprint.read_text(encoding='utf-8').replace('| Existing conservative reservations | 11.05 |\n| Larger native A100 evaluation | 4 |',
          '| Completed conservative reservations including failures | 16.95 |').replace('| Unallocated | 1.95 |','| Unallocated | 0.05 |')
    text=text.replace('Compare calibration and converged controls before broader query-dependent learning across native datasets.',
          'Larger 2026-release A100 development and converged/calibration controls are complete; neural utility is small or matched by simple controls. Diagnose query-dependent learning on CPU before another paid protocol.')
    write(sprint,text)
    decision=ROOT/'decision.md'
    text=decision.read_text(encoding='utf-8')
    if '## Larger 2026-release A100 outcome' not in text:
        text+='''

## Larger 2026-release A100 outcome — October 3

Nine complete A100 trials on BRFSS diabetes/asthma and UK collision severity use 50,000 source labels each, fixed native meta-training and held-out 2025 regions. Neural NLL .333545/.395610/.603685 versus frozen .333466/.395842/.611319. Strong calibration/simplex controls match or improve the neural model; no distinctive nonlinear advantage is established. Conditional road gain .007634 [-.001617,.016885] is uncertain. Two domains, correlated health targets and reporting/state changes constrain inference. This is used development, not confirmation or venue readiness.

All 1,470 probability arrays and query-label/group/year/checkpoint boundaries pass independent audit. Converged contextual linear gradient certificates pass all nine trials; simplex KKT residual <=2.01e-6. Preserve the calibration and cloud-mount failures and all partial scores. Numerical repair changes neither objective, ridges, model, data nor partitions; no scientific selection follows scores. Successful app stays deployed with zero GPU containers; failed apps stopped. Latest cap $20, reservations $16.95, reproduction reserve $3. Provider UI total $0.11 is rounded and unverified as final invoices. No further paid expansion without reconciled headroom or a changed cap. Next use CPU diagnostics and prior art to assess a query-conditioned, variable-width trained successor; do not manufacture a positive result.
'''
    write(decision,text)
    gate=ROOT/'docs/POLICY_ADAPTER_NEXT_GATE.md'
    text=gate.read_text(encoding='utf-8')
    if '## Completed larger gate' not in text:
        text+='''

## Completed larger gate — October 3

The separately frozen larger native A100 development is complete and audited: nine trials, two domains, recent 2026 releases, 1,016,280 raw records, 1,470 saved probability arrays. Global/support calibration and converged linear/convex stacking controls are complete. Health effects are tiny/negative and road gain is matched by simple controls. A distinctive neural contribution is still unsupported. Preserve all failed gates; these panels and partial v1 outcomes remain used development.

Next identify query-conditional or variable-width residual structure that simple calibration cannot represent, using bounded free CPU diagnostics and targeted prior-art review. A new trainable model needs a fixed new protocol and direct superiority against appropriate trained controls. Do not rely on increasing parameter count, dataset size or GPU power alone. Further paid scientific work requires reconciling retained reservations or a changed user cap; $16.95 of the $20 cap is reserved and $3 remains protected. Successful live app has zero active containers; do not repeat completed calls. Same manuscript contains the results; native compiler still has an environment error.
'''
    write(gate,text)
    protocol=ROOT/'docs/LARGE_NATIVE_PROTOCOL.md'
    text=protocol.read_text(encoding='utf-8')
    if '## Execution outcome' not in text:
        text+='''

## Execution outcome and separately frozen repairs

Original source/protocol freeze commit 7c1f513 preceded GPU outcomes. The first call trained four complete trials before an L-BFGS calibration line-search failure. V2 froze a safeguarded Newton solver with the same convex ridge objective and verified stationarity, but cloud input validation found an omitted historical wrapper mount before child training. V3 freezes the complete source mount, the same data/seeds/models/objectives and a $0.40 A100 call with two physical cores, 8 GiB RAM, 60-second startup, 300-second function and 240-second child limits. Its resource bound is $0.22571 for 360 seconds plus other-charge margin. All nine trials completed in 62.020 seconds. Failures remain charged and partial outcomes are preserved; no automatic retry or scientific selection. Total reservations $16.95 include all three attempts; $3 remains protected inside $20. Independent audit, raw probabilities, checkpoints and descriptive results are in artifacts/reports/large_native_a100_v3/.
'''
    write(protocol,text)
    paper=ROOT/'paper/main.tex'
    text=paper.read_text(encoding='utf-8')
    old='These findings support limited predictive adaptation, not an established novel learned-method advantage.'
    new='Nine further A100 trials evaluate health surveys and road collisions from 2026 releases, using 50,000 source labels per fit and held-out 2025 regions. Neural gains are tiny or matched by calibration and convex stacking controls. These findings support limited predictive adaptation, not an established novel learned-method advantage.'
    if 'Nine further A100 trials' not in text:
        text=text.replace(old,new)
    write(paper,text)
