"""Frozen, isolated backbone study. Selection sees labeled support only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mira.data import Mechanism, Observations, Support, QueryInputs
from mira.models import ModelSettings, create_model, predict_binary, cold_start_probability
from mira.representations import PreparedInputs
from mira.artifacts import atomic_json, atomic_npz, capture_environment, checkpoint_hashes
from mira.metrics import expected_nll
from feasibility import library, transform

MODELS = ("tabpfn:v2", "tabicl:v2")
VIEWS = ("native", "identity", "trained", "heuristic", "matched_linear", "cv_basis", "random_basis", "deepsets", "matched_deepsets", "shuffled")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Frozen paths must stay inside their declared root")
    return path


def load_protocol(path, root=ROOT):
    path = Path(path)
    digest = sha(path)
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise ValueError("Frozen protocol SHA256 differs")
    plan = json.loads(path.read_text())
    if not plan.get("frozen") or not isinstance(plan.get("protocol_version"), str) or not plan["protocol_version"]:
        raise ValueError("An explicitly frozen, versioned protocol is required")
    if plan.get("models") != list(MODELS) or plan.get("ensembles") != 4:
        raise ValueError("Require the two exact v2 backbones and four ensembles")
    synthetic = plan.get("synthetic")
    expected = {"orders": [2, 4], "gammas": [0, .9], "seeds": [74000, 74001, 74002], "context": 128, "queries": 256}
    if synthetic != expected:
        raise ValueError("Synthetic development matrix differs from the bounded design")
    views = plan.get("views", [])
    required = {"native", "identity", "trained", "heuristic", "matched_linear", "cv_basis", "random_basis"}
    if not required <= set(views) or len(set(views)) != len(views) or any(v not in VIEWS for v in views):
        raise ValueError("Missing/unknown/duplicate frozen representation views")
    real = plan.get("real", [])
    if not isinstance(real, list) or sum(len(r.get("folds", [])) for r in real) > 6:
        raise ValueError("At most six natural episodes fit this bounded study")
    if not isinstance(plan.get("checkpoint_sha256"), dict) or not plan["checkpoint_sha256"]:
        raise ValueError("Expected backbone checkpoint hashes are required")
    files = plan.get("file_sha256", {})
    required_files = {"experiments/mask_compiler/run_backbone.py", "infra/modal_compiler.py",
                      "infra/modal_mechanism.py", "pyproject.toml",
                      "experiments/mask_compiler/feasibility.py", "experiments/mask_compiler/train_selector.py",
                      "experiments/mask_compiler/matched_training.py", plan.get("compiler_checkpoint"), plan.get("linear_checkpoint")}
    required_files.update(str(p.relative_to(root)).replace("\\", "/") for p in (Path(root)/"src"/"mira").glob("*.py"))
    if {"deepsets", "matched_deepsets"} <= set(views):
        raise ValueError("Use one spelling for the optional DeepSets view")
    if {"deepsets", "matched_deepsets"} & set(views):
        required_files.add(plan.get("deepsets_checkpoint"))
    for record in real:
        directory = record.get("directory", "")
        if not record.get("folds") or len(set(record["folds"])) != len(record["folds"]):
            raise ValueError("Natural folds must be explicitly frozen without duplicates")
        required_files.update(f"{directory}/{name}" for name in ("dataset.npz", "splits.json", "manifest.json"))
    if None in required_files or not required_files <= set(files):
        raise ValueError("Protocol omits mandatory source/checkpoint/data hashes")
    for relative, expected_sha in files.items():
        if sha(contained(root, relative)) != expected_sha:
            raise ValueError(f"Frozen source/data/checkpoint changed: {relative}")
    return plan, digest


def verify_checkpoints(cache_root, expected):
    for relative, expected_sha in expected.items():
        # Existing runs record /cache-prefixed names; accept that spelling only.
        if str(relative).startswith("/cache/"):
            relative = str(relative)[7:]
        path = contained(cache_root, relative)
        if not path.is_file() or sha(path) != expected_sha:
            raise ValueError(f"Frozen backbone checkpoint missing/changed: {relative}")


def checkpoint_stats(cache_root, expected):
    """Fast between-cell guard; full SHA verification brackets the run."""
    result = {}
    for relative in expected:
        name = str(relative)[7:] if str(relative).startswith("/cache/") else relative
        stat = contained(cache_root, name).stat()
        result[relative] = (stat.st_size, stat.st_mtime_ns)
    return result


def load_selectors(plan, root=ROOT):
    from matched_training import load_selector, fixed_heuristic
    result = {"trained": load_selector("prototype", contained(root, plan["compiler_checkpoint"])),
              "matched_linear": load_selector("linear", contained(root, plan["linear_checkpoint"]))}
    for view in ("deepsets", "matched_deepsets"):
        if view in plan["views"]:
            result[view] = load_selector("deepsets", contained(root, plan["deepsets_checkpoint"]))
    result["heuristic"] = fixed_heuristic
    return result


def select_cv_basis(support, groups=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    labels = support.labels
    if groups is None:
        folds = list(StratifiedKFold(3, shuffle=True, random_state=91).split(np.zeros(len(labels)), labels))
    else:
        folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=91).split(np.zeros(len(labels)), labels, groups))
        if any(set(np.asarray(groups)[t]) & set(np.asarray(groups)[v]) for t,v in folds):
            raise ValueError("Natural support groups leaked within selector CV")
    best = None
    for index, matrix in enumerate(library(4)):
        features = np.column_stack([support.observations.u, 2*transform(support.observations.mask, matrix).astype(float)-1])
        for c in (.1, 1., 10.):
            losses = []
            for train, valid in folds:
                if len(np.unique(labels[train])) != 2:
                    raise ValueError("Selector CV fold lacks both support classes")
                estimator = LogisticRegression(C=c, max_iter=1000, random_state=91).fit(features[train], labels[train])
                losses.append(expected_nll(estimator.predict_proba(features[valid])[:, 1], labels[valid]))
            candidate = (float(np.mean(losses)), index, c)
            if best is None or candidate < best:
                best = candidate
    loss, index, c = best
    return library(4)[index], {"index": index, "C": c, "cv_nll": loss, "cv_fits": 108,
                             "group_cv": groups is not None, "objective": "support-only mask-logistic surrogate"}


def select_views(support, seed, views, selectors, groups=None):
    result = {}
    identity = np.eye(4, dtype=np.uint8)
    for view in views:
        if view in ("native", "identity", "shuffled"):
            matrix, info = identity, {"selection": "fixed identity/no labels"}
        elif view == "random_basis":
            index = int(np.random.default_rng([seed, 919]).integers(12))
            matrix, info = library(4)[index], {"index": index, "selection": "label-free seeded random", "identity_probability": 1/12}
        elif view == "cv_basis":
            matrix, info = select_cv_basis(support, groups)
        elif view == "heuristic":
            matrix, info = selectors[view](support, confidence=.3)
        else:
            matrix, info = selectors[view].select(support, confidence=.8)
        matrix = np.asarray(matrix, dtype=np.uint8)
        transform(support.observations.mask, matrix)  # Check invertibility before any backbone call.
        result[view] = {"matrix": matrix.tolist(), "selection": info}
    return result


def prepare_inputs(support, query, selected_columns, matrix, mode, seed, native_context, native_query, real=False):
    """Native X is copied unchanged; only appended mask coordinates are compiled."""
    if not isinstance(support, Support) or not isinstance(query, QueryInputs):
        raise TypeError("Restricted selector inputs required")
    xc, xq = np.array(native_context, copy=True), np.array(native_query, copy=True)
    if not real:
        xc, xq = np.column_stack([support.observations.u, xc]), np.column_stack([query.observations.u, xq])
    if mode != "native":
        mc, mq = np.isnan(native_context).astype(np.uint8), np.isnan(native_query).astype(np.uint8)
        if mode == "shuffled":
            rc, rq = np.random.default_rng([seed, 921, 1]), np.random.default_rng([seed, 921, 2])
            mc = np.column_stack([rc.permutation(v) for v in mc.T])
            mq = np.column_stack([rq.permutation(v) for v in mq.T])
        else:
            mc[:, selected_columns] = transform(support.observations.mask, matrix)
            mq[:, selected_columns] = transform(query.observations.mask, matrix)
        xc, xq = np.column_stack([xc, mc]), np.column_stack([xq, mq])
    return PreparedInputs(xc.astype(np.float32), support.labels.copy(), xq.astype(np.float32), mode)


def synthetic_episodes(plan):
    from matched_training import private_sample
    spec = plan["synthetic"]
    for order, gamma, seed in itertools.product(spec["orders"], spec["gammas"], spec["seeds"]):
        active = tuple(sorted(np.random.default_rng([seed, order, 911]).choice(4, order, replace=False)))
        task = Mechanism("compiler_backbone_development", 4, active, 1, gamma)
        obs, st = private_sample(task, spec["context"], np.random.default_rng([seed, order, 1]), np.random.default_rng([seed, order, 3]))
        qo, qt = private_sample(task, spec["queries"], np.random.default_rng([seed, order, 2]), np.random.default_rng([seed, order, 4]))
        yield {"episode_id": f"synthetic_k{order}_g{gamma:g}_s{seed}", "kind": "synthetic", "dataset": None, "fold": None,
               "order": order, "gamma": gamma, "seed": seed, "support": Support(obs, st.labels), "query": QueryInputs(qo),
               "native_context": obs.values, "native_query": qo.values, "selected_columns": np.arange(4),
               "targets": qt.labels, "oracle": qt.oracle, "groups": None, "active_evaluator_only": np.array(active),
               "support_ids": np.array([f"s{seed}_k{order}_g{gamma:g}_support_{i}" for i in range(spec["context"])]),
               "query_ids": np.array([f"s{seed}_k{order}_g{gamma:g}_query_{i}" for i in range(spec["queries"])])}


def natural_episodes(plan, root=ROOT):
    from mira.panel_data import load_panel
    for record in plan.get("real", []):
        directory = contained(root, record["directory"])
        panel = load_panel(directory)
        splits = json.loads((directory/"splits.json").read_text())
        manifest = json.loads((directory/"manifest.json").read_text())
        for fold in record["folds"]:
            split = next(s for s in splits["folds"] if s["fold"] == fold)
            si, qi = np.asarray(split["support_indices"], dtype=int), np.asarray(split["query_indices"], dtype=int)
            if ((si < 0) | (si >= len(panel.y))).any() or ((qi < 0) | (qi >= len(panel.y))).any():
                raise ValueError("Natural row indices are out of bounds")
            if len(si) > 128 or len(qi) > 256 or len(si) < 8 or not len(qi):
                raise ValueError("Natural episode exceeds frozen size bounds")
            if len(np.unique(si)) != len(si) or len(np.unique(qi)) != len(qi) or set(si)&set(qi):
                raise ValueError("Duplicate/leaking natural row indices")
            if set(panel.group_ids[si]) & set(panel.group_ids[qi]):
                raise ValueError("Natural source groups leaked between support and query")
            for key, actual in (("support_group_ids", panel.group_ids[si]), ("query_group_ids", panel.group_ids[qi]),
                                ("support_row_ids", panel.row_ids[si]), ("query_row_ids", panel.row_ids[qi])):
                if split.get(key) != actual.tolist():
                    raise ValueError("Natural saved row/group IDs differ from indices")
            rates = panel.native_mask[si].mean(axis=0)
            columns = np.lexsort((np.arange(len(rates)), -rates))[:4]
            if len(columns) != 4 or split.get("selected_columns") != columns.tolist():
                raise ValueError("Natural four-column support-only ranking differs")
            u, uq = np.zeros(len(si)), np.zeros(len(qi))
            support = Support(Observations(u, panel.X[si][:, columns], panel.native_mask[si][:, columns]), panel.y[si])
            query = QueryInputs(Observations(uq, panel.X[qi][:, columns], panel.native_mask[qi][:, columns]))
            yield {"episode_id": f"natural_{directory.name}_f{fold}", "kind": "natural", "dataset": directory.name, "fold": fold,
                   "order": None, "gamma": None, "seed": int(split.get("support_seed", 75000+fold)), "support": support,
                   "query": query, "native_context": panel.X[si], "native_query": panel.X[qi], "selected_columns": columns,
                   "targets": panel.y[qi], "oracle": None, "groups": panel.group_ids[si],
                   "query_groups": panel.group_ids[qi],
                   "support_ids": panel.row_ids[si], "query_ids": panel.row_ids[qi], "native_manifest": manifest,
                   "selector_ood": "U proxy is identically zero; no U column is added to natural backbone X"}


def run(plan, protocol_sha, model, out, root=ROOT, cache_root=Path("/cache"), factory=create_model,
        selectors=None, episodes=None, capture_lock=True, check_cache=True, max_seconds=740):
    if model not in MODELS:
        raise ValueError("Explicit v2 backbone required")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preserve old outputs; a fresh invocation directory is required")
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("episodes", "selections", "predictions"):
        (out/directory).mkdir()
    started = time.monotonic()
    rows, failures = [], []
    manifest = {"status": "running", "protocol_version": plan["protocol_version"], "protocol_sha256": protocol_sha,
                "model": model, "views": plan["views"], "source_sha256": plan["file_sha256"], "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(out/"manifest.json", manifest)
    environment = capture_environment(out, lock=capture_lock)
    atomic_json(out/"environment.json", environment)
    try:
        if check_cache:
            verify_checkpoints(cache_root, plan["checkpoint_sha256"])
            frozen_checkpoint_stats = checkpoint_stats(cache_root, plan["checkpoint_sha256"])
        selectors = load_selectors(plan, root) if selectors is None else selectors
        episodes = list(itertools.chain(synthetic_episodes(plan), natural_episodes(plan, root))) if episodes is None else list(episodes)
        manifest["expected_cells"] = len(episodes)*len(plan["views"])
        # Persist every episode before model execution, including evaluator-only arrays.
        for episode in episodes:
            path = out/"episodes"/(episode["episode_id"]+".npz")
            support, query = episode["support"], episode["query"]
            arrays = {"X_support": episode["native_context"], "X_query": episode["native_query"],
                      "u_support": support.observations.u, "u_query": query.observations.u,
                      "native_mask_support": np.isnan(episode["native_context"]).astype(np.uint8),
                      "native_mask_query": np.isnan(episode["native_query"]).astype(np.uint8),
                      "y_support": support.labels, "y_query_evaluator_only": episode["targets"],
                      "selected_columns": episode["selected_columns"]}
            for key in ("oracle", "active_evaluator_only", "support_ids", "query_ids", "groups", "query_groups"):
                if episode.get(key) is not None:
                    arrays[key] = episode[key]
            atomic_npz(path, **arrays)
            episode["data_file"], episode["data_sha256"] = str(path.relative_to(out)).replace("\\", "/"), sha(path)
            atomic_json(path.with_suffix(".json"), {k: episode[k] for k in ("episode_id", "kind", "dataset", "fold", "order", "gamma", "seed")} | {
                "selector_ood": episode.get("selector_ood"), "selected_columns": episode["selected_columns"].tolist(), "data_sha256": episode["data_sha256"]})
        for episode in episodes:
            support, query = episode["support"], episode["query"]
            selection_start = time.monotonic()
            selected = select_views(support, episode["seed"], plan["views"], selectors, episode.get("groups"))
            selection_seconds = time.monotonic()-selection_start
            atomic_json(out/"selections"/(episode["episode_id"]+".json"), selected)
            for mode in plan["views"]:
                if time.monotonic()-started > max_seconds:
                    manifest["status"] = "budget_truncated"
                    return manifest
                inputs = prepare_inputs(support, query, episode["selected_columns"], np.array(selected[mode]["matrix"]), mode,
                                        episode["seed"], episode["native_context"], episode["native_query"], episode["kind"] == "natural")
                estimator = None
                begun = time.monotonic()
                try:
                    p = cold_start_probability(inputs.labels, len(inputs.query))
                    cold_start = p is not None
                    parameters = None
                    if not cold_start:
                        estimator = factory(model, ModelSettings("cuda", 4), episode["seed"])
                        parameters = {k: repr(v) for k,v in estimator.get_params(deep=False).items()}
                        p = predict_binary(estimator, inputs)
                    prediction = out/"predictions"/(episode["episode_id"]+"_"+mode+".npz")
                    atomic_npz(prediction, probability=p)
                    if check_cache and checkpoint_stats(cache_root, plan["checkpoint_sha256"]) != frozen_checkpoint_stats:
                        raise ValueError("Frozen backbone checkpoint changed during execution")
                    row = {k: episode[k] for k in ("episode_id", "kind", "dataset", "fold", "order", "gamma", "seed")}
                    row.update(model=model, mode=mode, expected_nll=expected_nll(p, episode["oracle"]) if episode["oracle"] is not None else None,
                               empirical_nll=expected_nll(p, episode["targets"]), empirical_brier=float(np.mean((p-episode["targets"])**2)),
                               queries=len(p), input_columns=inputs.context.shape[1], cold_start=cold_start,
                               seconds=time.monotonic()-begun, data_file=episode["data_file"], data_sha256=episode["data_sha256"],
                               selection_seconds=selection_seconds,
                               prediction_file=str(prediction.relative_to(out)).replace("\\", "/"), prediction_sha256=sha(prediction),
                               selection=selected[mode], model_parameters=parameters)
                    rows.append(row)
                    atomic_json(out/"results.json", rows)
                    print(json.dumps({k: row[k] for k in ("episode_id", "mode", "expected_nll", "empirical_nll", "seconds")}), flush=True)
                except Exception as error:
                    failures.append({"episode_id": episode["episode_id"], "mode": mode, "model": model, "error": repr(error)})
                    manifest["status"] = "failed; remaining backbone calls skipped without retry"
                    return manifest
                finally:
                    del estimator
                    import gc
                    gc.collect()
                    torch = sys.modules.get("torch")
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if check_cache:
            verify_checkpoints(cache_root, plan["checkpoint_sha256"])
        manifest["status"] = "complete"
        return manifest
    except Exception as error:
        failures.append({"stage": "preparation_or_selection", "error": repr(error)})
        manifest["status"] = "failed before or between backbone calls"
        return manifest
    finally:
        manifest.update(completed_cells=len(rows), failures=len(failures), seconds=time.monotonic()-started)
        atomic_json(out/"errors.json", failures)
        atomic_json(out/"results.json", rows)
        atomic_json(out/"checkpoint_sha256.json", checkpoint_hashes([cache_root]) if check_cache else {})
        atomic_json(out/"manifest.json", manifest)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/cache"))
    args = parser.parse_args()
    plan, digest = load_protocol(args.protocol)
    result = run(plan, digest, args.model, args.out, cache_root=args.cache_root)
    if result["status"] != "complete":
        raise SystemExit("Incomplete frozen experiment; partial artifacts preserved and no model substituted")


if __name__ == "__main__":
    cli()
