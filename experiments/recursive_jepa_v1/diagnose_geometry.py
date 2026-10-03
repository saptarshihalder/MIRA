"""Used-panel diagnosis only: compare frozen learned/random embeddings and head fits."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
import run

ROOT = run.ROOT
OUT = ROOT / 'artifacts/reports/recursive_jepa_geometry'


def fit_predict(support, query, labels, offset, query_offset, penalty=.1):
    mean, scale = support.mean(0), support.std(0).clip(.01)
    a = np.column_stack((np.ones(len(support)), (support - mean) / scale))
    b = np.column_stack((np.ones(len(query)), (query - mean) / scale))
    def objective(w):
        z = offset + a @ w
        return np.mean(np.logaddexp(0, z) - labels * z) + .5 * penalty * (w @ w), a.T @ (expit(z) - labels) / len(labels) + penalty * w
    result = minimize(objective, np.zeros(a.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=300, gtol=1e-8, ftol=1e-12))
    if not result.success:
        raise ValueError(result.message)
    return expit(query_offset + b @ result.x)


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'artifacts/reports/recursive_jepa_v1_gpu'
    report = json.loads((folder / 'report.json').read_text())
    plan = report['effective_plan']
    trained = run.restore_model('full_jepa', plan, folder / 'full_jepa_weights.npz')
    random = run.make_model('full_jepa', plan).eval()
    rows = []
    for file in sorted((folder / 'predictions').glob('*.npz')):
        with np.load(file, allow_pickle=False) as saved:
            inputs = [torch.as_tensor(saved[k][None].copy(), dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in run.v1.INPUT_KEYS]
            labels = saved['labels'].astype(float)
            row = dict(file=file.name, regime=file.stem.split('_', 2)[2], losses={name:run.v1.metrics(saved[name], labels)['nll'] for name in ('frozen', 'full_jepa', 'support_logistic')}, geometry={})
            for name, model in (('trained', trained), ('random', random)):
                with torch.no_grad():
                    details = model.forward_details(*inputs)
                for representation in ('base_latents',):
                    target = details[representation]['target'][0].numpy().astype(float)
                    query = details[representation]['query'][0].numpy().astype(float)
                    singular = np.linalg.svd(query - query.mean(0), compute_uv=False)
                    mass = singular.square() if hasattr(singular, 'square') else singular ** 2
                    mass = mass / max(mass.sum(), 1e-20)
                    rank = np.exp(-np.sum(mass * np.log(mass.clip(1e-20))))
                    row['geometry'][name] = dict(mean_std=float(query.std(0).mean()), effective_rank=float(rank))
                    probability = fit_predict(target, query, saved['target_y'], saved['target_logit'], saved['query_logit'])
                    row['losses'][name + '_solved'] = run.v1.metrics(probability, labels)['nll']
            rows.append(row)
    means = {regime:{name:float(np.mean([r['losses'][name] for r in rows if r['regime']==regime])) for name in rows[0]['losses']} for regime in run.v1.REGIMES}
    result = dict(scope='post-hoc diagnosis on consumed engineering worlds; no architecture selection or confirmation claim', penalty=.1, rows=rows, means=means)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(means))


if __name__ == '__main__':
    main()
