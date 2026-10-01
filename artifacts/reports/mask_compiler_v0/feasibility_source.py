"""CPU development check; no TFM, novelty or confirmation claim."""
from pathlib import Path
import itertools
import json
import hashlib
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mira.data import Mechanism, Support, QueryInputs, _sample
from mira.artifacts import source_hashes, capture_environment
from mira.metrics import expected_nll

CS = (.1, 1., 10.)


def basis(parity):
    a = np.asarray(parity, dtype=np.uint8)
    if a.ndim != 1 or not np.isin(a, [0, 1]).all() or not a.any():
        raise ValueError("Nonzero binary parity required")
    result = np.eye(len(a), dtype=np.uint8)
    result[np.flatnonzero(a)[0]] = a
    return result


def transform(mask, matrix):
    m = np.asarray(mask)
    a = np.asarray(matrix)
    if m.ndim != 2 or a.shape != (m.shape[1], m.shape[1]):
        raise ValueError("Mask/matrix shape mismatch")
    if not np.isin(m, [0, 1]).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("Binary inputs required")
    if not np.array_equal((a.astype(int) @ a.astype(int)) % 2, np.eye(len(a))):
        raise ValueError("Library requires self-inverse matrices")
    return (m.astype(int) @ a.astype(int).T % 2).astype(np.uint8)


def library(d):
    matrices = []
    for bits in itertools.product((0, 1), repeat=d):
        if any(bits):
            a = basis(bits)
            if not any(np.array_equal(a, b) for b in matrices):
                matrices.append(a)
    identity = np.eye(d, dtype=np.uint8)
    return [identity] + [a for a in matrices if not np.array_equal(a, identity)]


def features(observations, matrix=None, all_parities=False):
    m = observations.mask
    if all_parities:
        codes = np.array([a for a in itertools.product((0, 1), repeat=m.shape[1]) if any(a)])
        z = (m @ codes.T) % 2
    else:
        z = transform(m, matrix)
    return np.column_stack([observations.u, 2 * z.astype(float) - 1])


def fit_support(support, query, matrices, all_parities=False):
    """Support labels only; query type has no targets or oracle metadata."""
    if not isinstance(support, Support) or not isinstance(query, QueryInputs):
        raise TypeError("Restricted learner inputs required")
    labels = support.labels
    folds = list(StratifiedKFold(3, shuffle=True, random_state=91).split(np.zeros(len(labels)), labels))
    best = None
    for index, matrix in enumerate(matrices):
        xc = features(support.observations, matrix, all_parities)
        for c in CS:
            scores = []
            for train, valid in folds:
                model = LogisticRegression(C=c, max_iter=1000).fit(xc[train], labels[train])
                p = np.clip(model.predict_proba(xc[valid])[:, 1], 1e-6, 1 - 1e-6)
                scores.append(float(np.mean(-labels[valid] * np.log(p) - (1-labels[valid]) * np.log1p(-p))))
            score = float(np.mean(scores))
            if best is None or score < best[0]:
                best = (score, index, c)
    score, index, c = best
    model = LogisticRegression(C=c, max_iter=1000).fit(features(support.observations, matrices[index], all_parities), labels)
    p = model.predict_proba(features(query.observations, matrices[index], all_parities))[:, 1]
    return p, {"cv_nll": score, "matrix_index": index, "matrix": matrices[index].tolist(), "C": c,
               "candidate_count": len(matrices), "C_count": len(CS)}


def run(out):
    if out.exists():
        raise ValueError("Use a new output directory; preserve prior results")
    out.mkdir(parents=True)
    started = time.monotonic()
    d = 4
    matrices = library(d)
    masks = np.array(list(itertools.product((0, 1), repeat=d)), dtype=np.uint8)
    for a in matrices:
        z = transform(masks, a)
        assert np.array_equal(transform(z, a), masks)
        assert len(np.unique(z, axis=0)) == 2**d
        assert np.array_equal(z.sum(axis=0), masks.sum(axis=0))
    rows = []
    for order, context, gamma, seed in itertools.product(range(1, 5), (32, 128), (0., .9), range(73000, 73005)):
        rng = np.random.default_rng(seed)
        active = tuple(sorted(rng.choice(d, order, replace=False).tolist()))
        task = Mechanism("compiler_development", d, active, 1, gamma)
        obs, st = _sample(task, context, np.random.default_rng([seed, order, 1]), np.random.default_rng([seed, order, 3]))
        qo, qt = _sample(task, 512, np.random.default_rng([seed, order, 2]), np.random.default_rng([seed, order, 4]))
        support, query = Support(obs, st.labels), QueryInputs(qo)
        code = np.array([int(j in active) for j in range(d)])
        aligned = basis(code)
        random_matrix = matrices[int(np.random.default_rng([seed, order, 9]).integers(len(matrices)))]
        name = f"k{order}_n{context}_g{gamma}_s{seed}"
        np.savez_compressed(out / (name+"_episode.npz"), xc=obs.values, uc=obs.u, mc=obs.mask, yc=st.labels,
                            xq=qo.values, uq=qo.u, mq=qo.mask, yq=qt.labels, oracle=qt.oracle,
                            active=np.array(active), support_ids=np.array([f"{name}:support:{i}" for i in range(context)]),
                            query_ids=np.array([f"{name}:query:{i}" for i in range(512)]))
        # Primitive/compiled logit algebra is exactly checked, not fitted or supplied to a learner.
        z = transform(qo.mask, aligned)
        pivot = int(np.flatnonzero(code)[0])
        parity_signal = (-1)**order * (1 - 2*z[:, pivot].astype(float))
        assert np.array_equal(parity_signal, task.signal(qo.mask, qo.u))
        methods = (("identity", [matrices[0]], False), ("random_basis", [random_matrix], False),
                   ("oracle_aligned_reference", [aligned], False), ("support_selected", matrices, False),
                   ("full_parity_logistic", [matrices[0]], True))
        for method, choices, full in methods:
            p, config = fit_support(support, query, choices, full)
            file = out / (name+"_"+method+".npz")
            np.savez_compressed(file, p=p)
            # Evaluation begins after fitted predictions have been persisted.
            rows.append({"task": name, "order": order, "context": context, "gamma": gamma, "seed": seed,
                         "method": method, "expected_nll": expected_nll(p, qt.oracle),
                         "oracle_nll": expected_nll(qt.oracle, qt.oracle), "selection": config,
                         "prediction_file": file.name, "prediction_sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    (out / "results.json").write_text(json.dumps(rows, indent=2)+"\n")
    table = []
    for order, context, gamma in itertools.product(range(1, 5), (32, 128), (0., .9)):
        values = {method: float(np.mean([r["expected_nll"] for r in rows if (r["order"],r["context"],r["gamma"],r["method"]) == (order,context,gamma,method)])) for method,_,_ in methods}
        table.append({"order":order,"context":context,"gamma":gamma,"losses":values})
    manifest = {"status":"complete","tasks":80,"predictions":400,"development_seeds":list(range(73000,73005)),
                "matrices":len(matrices),"queries":512,"seconds":time.monotonic()-started,"paid_usd":0,
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"core_source_sha256":source_hashes(),
                "environment":capture_environment(out,lock=True),"checks":"All16 masks bijective/round-trip; uniform marginals; exact oracle signal under aligned basis",
                "scope":"CPU coordinate/logistic feasibility only. U+mask features; Gaussian nuisance values deliberately unused. No TFM/novelty/confirmation claim. Aligned basis uses known active set and is privileged. Full-parity baseline has15 mask features versus4; compiler searches12 distinct bases versus baseline one basis, common3 Cs/3 folds."}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out / "summary.json").write_text(json.dumps(table,indent=2)+"\n")
    lines=["# CPU reversible-coordinate feasibility (development)","",manifest["scope"],"","80 tasks/400 predictions; no paid calls. No confidence/acceptance claim.","","| Order | Labels | Gamma | Identity | Random | Aligned reference | Support-selected | Full parity |","|---|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['order']} | {row['context']} | {row['gamma']} | "+" | ".join(f"{row['losses'][m]:.5f}" for m,_,_ in methods)+" |")
    (out / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"tasks":80,"predictions":400,"seconds":manifest["seconds"],"paid_usd":0}))


if __name__ == "__main__":
    run(ROOT / "artifacts/reports/mask_compiler_v0")
