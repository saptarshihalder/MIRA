"""Matched-budget CPU selectors and downstream development comparisons.

Only labeled Support enters selectors. Privileged mechanism teachers are used
for synthetic pretraining and evaluator metrics, never selector inference.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_selector import Compiler, encode, CODES, MATRICES, dataset as prototype_dataset
from feasibility import ROOT, basis, features, CS
from mira.data import Mechanism, Support, QueryInputs, _sample
from mira.metrics import expected_nll

TRAINING = dict(train_start=81000, train_count=2048, validation_start=85000,
                validation_count=256, heldout_start=87000, heldout_count=256,
                epochs=80, learning_rate=.003, batch_size=128, torch_seed=901)
MODEL_DIRECTORY = ROOT / "artifacts/reports/matched_compiler_v1"
SOLVER_SEED = 91


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def private_sample(task, n, rng, value_rng):
    """Evaluator generator wrapper; signature matches the preserved core."""
    return _sample(task, n, rng, value_rng)


def rows(support):
    if not isinstance(support, Support) or support.observations.mask.shape[1] != 4 or not len(support.labels):
        raise TypeError("Nonempty four-bit labeled Support required")
    return np.column_stack([support.observations.u, 2 * support.observations.mask.astype(float) - 1,
                            2 * support.labels - 1]).astype(np.float32)


def matrix_index(matrix):
    return next(i for i, candidate in enumerate(MATRICES) if np.array_equal(matrix, candidate))


def guarded(probability, confidence):
    raw = int(probability.argmax())
    index = raw if float(probability.max()) >= confidence else 0
    return MATRICES[index].copy(), {"index": index, "raw_index": raw,
                                    "max_probability": float(probability.max()), "threshold": confidence}


class LinearSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(17, 12)

    def forward(self, x):
        return self.net(x)

    def select(self, support, confidence=.8):
        self.eval()
        with torch.no_grad():
            probability = self(torch.from_numpy(encode(support))[None]).softmax(-1)[0].numpy()
        return guarded(probability, confidence)


class DeepSetsSelector(nn.Module):
    """6092 parameters: learned U/mask/label row encoder, mean pool and size."""
    def __init__(self):
        super().__init__()
        self.cells = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 72), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(73, 40, bias=False), nn.LayerNorm(40), nn.ReLU(), nn.Linear(40, 12))

    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        pooled = (self.cells(x) * valid[:, :, None]).sum(1) / lengths[:, None]
        return self.head(torch.cat([pooled, torch.log2(lengths.float())[:, None] / 8], dim=1))

    def select(self, support, confidence=.8):
        self.eval()
        x = torch.from_numpy(rows(support))[None]
        with torch.no_grad():
            probability = self(x, torch.tensor([x.shape[1]])).softmax(-1)[0].numpy()
        return guarded(probability, confidence)


def fixed_heuristic(support, confidence=.8):
    moment = encode(support)[:15]
    best = int(moment.argmax())
    matrix = MATRICES[0] if moment[best] < .3 else basis(CODES[best])
    return matrix.copy(), {"index": matrix_index(matrix), "max_moment": float(moment[best]), "threshold": .3}


def load_selector(kind, checkpoint):
    constructors = {"prototype": Compiler, "linear": LinearSelector, "matched_linear": LinearSelector,
                    "deepsets": DeepSetsSelector, "matched_deepsets": DeepSetsSelector}
    model = constructors[kind]()
    with np.load(checkpoint, allow_pickle=False) as state:
        model.load_state_dict({key: torch.from_numpy(state[key].copy()) for key in state.files}, strict=True)
    return model.eval()


def load_selectors(model_dir=MODEL_DIRECTORY):
    directory = Path(model_dir)
    return {name: load_selector(kind, directory / filename) for name, kind, filename in
            (("prototype", "prototype", "prototype_selector.npz"),
             ("matched_linear", "linear", "linear_selector.npz"),
             ("matched_deepsets", "deepsets", "deepsets_selector.npz"))}


def select(model, support):
    return model.select(support, confidence=.8)


def training_data(start, count):
    moments, teachers = prototype_dataset(start, count)
    padded = np.zeros((count, 128, 6), dtype=np.float32)
    lengths = np.empty(count, dtype=np.int64)
    for i, seed in enumerate(range(start, start + count)):
        rng = np.random.default_rng([seed, 777])
        active = tuple(sorted(rng.choice(4, int(rng.integers(1, 5)), replace=False).tolist()))
        gamma = float(rng.choice([0., .25, .5, .9]))
        task = Mechanism("trained_compiler_development", 4, active, int(rng.choice([-1, 1])),
                         gamma, beta=float(rng.uniform(.2, 1.2)))
        n = int(rng.choice([32, 128]))
        obs, targets = private_sample(task, n, np.random.default_rng([seed, 777, 1]), np.random.default_rng([seed, 777, 2]))
        support = Support(obs, targets.labels)
        np.testing.assert_array_equal(encode(support), moments[i])
        teacher = 0 if gamma == 0 else matrix_index(basis([int(j in active) for j in range(4)]))
        assert teacher == teachers[i]
        padded[i, :n] = rows(support)
        lengths[i] = n
    return moments, padded, lengths, teachers


def train(out, prototype):
    if out.exists():
        raise ValueError("Preserve existing outputs; choose a new --out")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    blocks = [training_data(TRAINING[start], TRAINING[count]) for start, count in
              (("train_start", "train_count"), ("validation_start", "validation_count"), ("heldout_start", "heldout_count"))]
    for label, block in zip(("train", "validation", "heldout"), blocks):
        np.savez_compressed(out / f"{label}_inputs.npz", moments=block[0], rows=block[1], lengths=block[2], teacher=block[3])
    # Reproduce the prototype's minibatch RNG after its 6092-parameter initialization.
    torch.manual_seed(TRAINING["torch_seed"])
    Compiler()
    schedule = [torch.randperm(TRAINING["train_count"]) for _ in range(TRAINING["epochs"])]
    np.savez_compressed(out / "batch_schedule.npz", indices=np.stack([order.numpy() for order in schedule]))
    result = {}
    for name, constructor, filename in (("matched_linear", LinearSelector, "linear_selector.npz"),
                                       ("matched_deepsets", DeepSetsSelector, "deepsets_selector.npz")):
        torch.manual_seed(TRAINING["torch_seed"])
        model = constructor()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
        moment, raw, lengths, teacher = [torch.from_numpy(array) for array in blocks[0]]
        losses = []
        model_start = time.monotonic()
        for epoch, order in enumerate(schedule):
            model.train()
            total = 0.
            for indices in order.split(TRAINING["batch_size"]):
                logits = model(raw[indices], lengths[indices]) if isinstance(model, DeepSetsSelector) else model(moment[indices])
                loss = nn.functional.cross_entropy(logits, teacher[indices])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(indices)
            losses.append(total / len(moment))
        np.savez_compressed(out / filename, **{key: value.detach().numpy() for key, value in model.state_dict().items()})
        model.eval()
        accuracies = {}
        for label, block in zip(("validation", "heldout_development"), blocks[1:]):
            with torch.no_grad():
                p = model(torch.from_numpy(block[1]), torch.from_numpy(block[2])) if isinstance(model, DeepSetsSelector) else model(torch.from_numpy(block[0]))
                probability = p.softmax(-1).numpy()
            guarded_indices = np.where(probability.max(1) >= .8, probability.argmax(1), 0)
            accuracies[label] = {"raw_accuracy": float(np.mean(probability.argmax(1) == block[3])),
                                 "guarded_accuracy": float(np.mean(guarded_indices == block[3]))}
            np.savez_compressed(out / f"{name}_{label}.npz", probability=probability, teacher=block[3])
        result[name] = {"parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - model_start,
                        "checkpoint_sha256": sha(out / filename), "accuracy": accuracies, "training_losses": losses}
        json_file(out / "training_progress.json", result)
        print(f"Trained {name}: {result[name]['parameters']} parameters", flush=True)
    shutil.copyfile(prototype, out / "prototype_selector.npz")
    manifest = {"status": "trained", "training": TRAINING, "models": result,
                "prototype_source": str(prototype.resolve()), "prototype_checkpoint_sha256": sha(prototype),
                "batch_schedule_sha256": sha(out / "batch_schedule.npz"), "seconds": time.monotonic() - started,
                "torch": torch.__version__, "source_sha256": sha(__file__), "prototype_source_sha256": sha(HERE / "train_selector.py"),
                "source_dependencies_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                    (HERE / "feasibility.py", ROOT / "src/mira/data.py", ROOT / "src/mira/metrics.py")},
                "checkpoint_format": "Numeric float32 NPZ state dictionaries, strict key/shape loading; no Torch pickle or compiled operators.",
                "paid_usd": 0, "limits": "Same privileged synthetic teacher/tasks/epochs/optimizer/batches; final epoch only. Linear has216 versus6092 parameters. DeepSets has6092 but different raw-row architecture. No evaluation-based checkpoint/threshold selection."}
    json_file(out / "training_manifest.json", manifest)


def logistic_prediction(support, query, choices, l1=False):
    if not isinstance(support, Support) or not isinstance(query, QueryInputs):
        raise TypeError("Support/query-only learner inputs")
    labels = support.labels
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SOLVER_SEED).split(np.zeros(len(labels)), labels))
    best = None
    for index, matrix in enumerate(choices):
        xc = features(support.observations, matrix, all_parities=l1)
        for c in CS:
            errors = []
            for fit, held in folds:
                estimator = LogisticRegression(C=c, solver="liblinear" if l1 else "lbfgs", l1_ratio=1. if l1 else 0.,
                                               random_state=SOLVER_SEED, max_iter=1000).fit(xc[fit], labels[fit])
                errors.append(expected_nll(estimator.predict_proba(xc[held])[:, 1], labels[held]))
            candidate = (float(np.mean(errors)), index, c)
            if best is None or candidate < best:
                best = candidate
    score, index, c = best
    estimator = LogisticRegression(C=c, solver="liblinear" if l1 else "lbfgs", l1_ratio=1. if l1 else 0.,
                                   random_state=SOLVER_SEED, max_iter=1000).fit(features(support.observations, choices[index], all_parities=l1), labels)
    probability = estimator.predict_proba(features(query.observations, choices[index], all_parities=l1))[:, 1]
    return probability, {"matrix_index": matrix_index(choices[index]), "C": c, "cv_nll": score}


def development_episode(seed, order, gamma):
    rng = np.random.default_rng([seed, order, 902])
    active = tuple(sorted(rng.choice(4, order, replace=False).tolist()))
    task = Mechanism("matched_compiler_development", 4, active, int(rng.choice([-1, 1])), gamma)
    obs, st = private_sample(task, 128, np.random.default_rng([seed, order, 902, 1]), np.random.default_rng([seed, order, 902, 3]))
    query, qt = private_sample(task, 256, np.random.default_rng([seed, order, 902, 2]), np.random.default_rng([seed, order, 902, 4]))
    return Support(obs, st.labels), QueryInputs(query), qt, task


def evaluate_development(out):
    models = load_selectors(out)
    predictions = out / "development_predictions"
    predictions.mkdir()
    records, errors = [], []
    started = time.monotonic()
    with threadpool_limits(limits=1):
        for order, gamma, seed in itertools.product(range(1, 5), (0., .9), range(90000, 90064)):
            support, query, targets, task = development_episode(seed, order, gamma)
            name = f"k{order}_g{gamma:g}_s{seed}"
            obs, qo = support.observations, query.observations
            np.savez_compressed(predictions / f"{name}_episode.npz", uc=obs.u, xc=obs.values, mc=obs.mask, yc=support.labels,
                                uq=qo.u, xq=qo.values, mq=qo.mask, yq=targets.labels, oracle=targets.oracle,
                                support_ids=np.array([f"{name}:support:{i}" for i in range(128)]),
                                query_ids=np.array([f"{name}:query:{i}" for i in range(256)]))
            teacher = 0 if gamma == 0 else matrix_index(basis([int(j in task.active) for j in range(4)]))
            choices = {method: select(model, support) for method, model in models.items()}
            choices["fixed_heuristic"] = fixed_heuristic(support)
            choices["identity"] = (MATRICES[0], {"index": 0})
            for method in (*choices, "full_parity_l1", "cv_compiler"):
                try:
                    matrices = MATRICES if method == "cv_compiler" else [choices[method][0] if method in choices else MATRICES[0]]
                    probability, fitting = logistic_prediction(support, query, matrices, l1=method == "full_parity_l1")
                    path = predictions / f"{name}_{method}.npz"
                    np.savez_compressed(path, probability=probability)
                    selection = choices[method][1] if method in choices else {"index": fitting["matrix_index"]}
                    if method == "full_parity_l1":
                        selection = {"index": None, "full_parities": True}
                    matrix = MATRICES[selection["index"]] if selection["index"] is not None else None
                    code = np.array([int(j in task.active) for j in range(4)])
                    records.append({"task": name, "order": order, "gamma": gamma, "seed": seed, "method": method,
                                    "expected_nll": expected_nll(probability, targets.oracle),
                                    "empirical_nll": expected_nll(probability, targets.labels),
                                    "oracle_nll": expected_nll(targets.oracle, targets.oracle), "selection": selection,
                                    "fitting": fitting, "teacher_index": teacher,
                                    "teacher_match": selection["index"] == teacher if matrix is not None else None,
                                    "active_parity_axis_present": bool(np.any(np.all(matrix == code, axis=1))) if matrix is not None else True,
                                    "prediction_file": str(path.relative_to(out)), "prediction_sha256": sha(path)})
                except Exception as error:
                    errors.append({"task": name, "method": method, "error": repr(error)})
                    json_file(out / "errors.json", errors)
            if len(records) % (64 * 7) == 0:
                json_file(out / "development_results.json", records)
                print(f"Scored {len(records)}/3584 development predictions", flush=True)
    summary = []
    for order, gamma, method in itertools.product(range(1, 5), (0., .9), (*models, "fixed_heuristic", "identity", "full_parity_l1", "cv_compiler")):
        cell = [record for record in records if (record["order"], record["gamma"], record["method"]) == (order, gamma, method)]
        if cell:
            summary.append({"order": order, "gamma": gamma, "method": method, "tasks": len(cell),
                            "mean_nll": float(np.mean([r["expected_nll"] for r in cell])),
                            "empirical_nll": float(np.mean([r["empirical_nll"] for r in cell])),
                            "teacher_accuracy": float(np.mean([r["teacher_match"] for r in cell])) if cell[0]["teacher_match"] is not None else None,
                            "active_parity_axis_fraction": float(np.mean([r["active_parity_axis_present"] for r in cell])),
                            "nonidentity_fraction": float(np.mean([r["selection"]["index"] != 0 for r in cell])) if cell[0]["selection"]["index"] is not None else None})
    json_file(out / "development_results.json", records)
    json_file(out / "errors.json", errors)
    json_file(out / "summary.json", summary)
    json_file(out / "development_manifest.json", {"status": "complete" if not errors else "incomplete", "predictions": len(records),
               "tasks": 512, "development_seeds": [90000, 90063], "support": 128, "queries": 256, "beta": .8,
               "orders": [1, 2, 3, 4], "gammas": [0., .9], "namespace": 902, "solver_seed": SOLVER_SEED,
               "C_grid": list(CS), "confidence": .8, "heuristic_threshold": .3, "seconds": time.monotonic() - started,
               "paid_usd": 0, "source_sha256": sha(__file__), "scope": "New used development tasks, CPU U+encoded-mask logistic only. No TFM, blind confirmation, causal-prior or novelty claim. Full-parity L1 has15 mask features versus4 and CV compiler searches12 bases."})
    lines = ["# Matched compiler development", "", "Fixed final epochs/thresholds; every new seed is used development. Same synthetic teacher/tasks/optimizer/minibatches, different selector architectures. No TFM inference.", "",
             "| Order | Gamma | Prototype | Linear | Raw DeepSets | Heuristic | Identity | Full parity L1 | CV compiler |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for order, gamma in itertools.product(range(1, 5), (0., .9)):
        cell = {r["method"]: r["mean_nll"] for r in summary if (r["order"], r["gamma"]) == (order, gamma)}
        lines.append(f"| {order} | {gamma:g} | " + " | ".join(f"{cell.get(method, float('nan')):.5f}" for method in (*models, "fixed_heuristic", "identity", "full_parity_l1", "cv_compiler")) + " |")
    lines += ["", "Matrix teacher accuracy and active-parity-axis presence are evaluator diagnostics, not inference inputs. Selection accuracy is not downstream performance. All query targets are used only after the issued probability is saved. Failures remain in errors.json."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=MODEL_DIRECTORY)
    parser.add_argument("--prototype", type=Path, default=ROOT / "artifacts/reports/trained_compiler_v1/trained_parameters.npz")
    args = parser.parse_args()
    try:
        train(args.out, args.prototype)
        evaluate_development(args.out)
    except Exception as error:
        if args.out.exists():
            json_file(args.out / "stage_failure.json", {"error": repr(error), "source_sha256": sha(__file__)})
        raise
