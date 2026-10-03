"""Engineering boundary checks; no research evaluation or training outputs."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
import torch

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
try:
    spec = importlib.util.spec_from_file_location('mixture_boundary_checks', DIRECTORY/'mixture.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
finally:
    sys.path.pop(0)

PLAN = dict(namespace=1601, source_labels=128, target_labels=64, queries=16, cv_seed=1702)


def episode():
    # Dedicated engineering seed, outside registered research seed ranges.
    return m.p.world(129990, 'development', PLAN)[1]


def scorer():
    torch.manual_seed(13)
    result = m.Mixture()
    torch.nn.init.normal_(result.score[-1].weight, std=.1)
    return result


def test_query_labels_oracle_and_policy_excluded_from_inference():
    e = episode()
    first = m.prepare(e, PLAN)
    model = scorer()
    prediction = m.predict(model, first['context'], first['probability'])
    e['query_y'] = 1-e['query_y']
    e['oracle'] = 1-e['oracle']
    e['target_policy'] = dict(fake=999)
    second = m.prepare(e, PLAN)
    np.testing.assert_array_equal(first['context'], second['context'])
    np.testing.assert_array_equal(first['probability'], second['probability'])
    assert first['cv_choice'] == second['cv_choice']
    assert first['moment_choice'] == second['moment_choice']
    np.testing.assert_array_equal(prediction, m.predict(model, second['context'], second['probability']))


def test_oof_expert_fits_exclude_heldout_support_rows():
    e = episode()
    original = m.experts
    calls = []
    support_labels = dict(zip(e['support_f'][:, 0].tolist(), e['support_y'].tolist()))

    def recording(sf, sy, qf):
        train_ids, heldout_ids = set(sf[:, 0].tolist()), set(qf[:, 0].tolist())
        calls.append((train_ids, heldout_ids))
        assert sy.tolist() == [support_labels[u] for u in sf[:, 0].tolist()]
        return original(sf, sy, qf)

    m.experts = recording
    try:
        m.prepare(e, PLAN)
    finally:
        m.experts = original
    assert len(calls) == 4
    all_support = set(e['support_f'][:, 0].tolist())
    seen = []
    for fit_ids, heldout_ids in calls[:3]:
        assert fit_ids.isdisjoint(heldout_ids)
        assert fit_ids | heldout_ids == all_support
        seen.extend(heldout_ids)
    assert len(seen) == len(set(seen)) == len(all_support)
    assert calls[-1][0] == all_support


def test_expert_fits_match_independent_regularized_optimizer():
    e = episode()
    sf, sy, qf = e['support_f'], e['support_y'], e['query_f']
    probability, weights = m.experts(sf, sy, qf)
    assert probability.shape == (len(qf), 16)
    np.testing.assert_allclose(probability[:, 0], expit(qf[:, -1]), atol=1e-8)
    for index in (0, 7, 14):
        design = np.column_stack((sf[:, 7+index].astype(float), np.ones(len(sf))))
        independent = m.p.offset_fit(design, sy, sf[:, -1].astype(float), .01)
        np.testing.assert_allclose(weights[index], independent, atol=4e-4, rtol=4e-4)
        query_design = np.column_stack((qf[:, 7+index], np.ones(len(qf))))
        expected = expit(qf[:, -1]+query_design@independent)
        np.testing.assert_allclose(probability[:, index+1], expected, atol=4e-5, rtol=4e-4)


def test_full_expert_fits_are_support_permutation_invariant():
    e = episode()
    first, w_first = m.experts(e['support_f'], e['support_y'], e['query_f'])
    second, w_second = m.experts(e['support_f'][::-1], e['support_y'][::-1], e['query_f'])
    np.testing.assert_allclose(w_first, w_second, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(first, second, atol=1e-10, rtol=1e-10)


def test_mixture_is_convex_and_candidate_permutation_invariant():
    task = m.prepare(episode(), PLAN)
    model = scorer()
    context, probability = task['context'], task['probability']
    first = m.predict(model, context, probability)
    assert np.all(first >= probability.min(1)-1e-7)
    assert np.all(first <= probability.max(1)+1e-7)
    order = np.random.default_rng(13).permutation(16)
    second = m.predict(model, context[order].copy(), probability[:, order].copy())
    np.testing.assert_allclose(first, second, atol=2e-7, rtol=2e-6)


def test_query_order_and_batch_membership_do_not_affect_predictions():
    task = m.prepare(episode(), PLAN)
    model = scorer()
    context, probability = task['context'], task['probability']
    first = m.predict(model, context, probability)
    np.testing.assert_allclose(first[::-1], m.predict(model, context, probability[::-1].copy()), atol=1e-7)
    np.testing.assert_allclose(first[:1], m.predict(model, context, probability[:1]), atol=1e-7)


def test_gradient_updates_leave_fitted_experts_and_context_fixed():
    task = m.prepare(episode(), PLAN)
    context, probability = task['context'].copy(), task['probability'].copy()
    model = scorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    prediction = model(torch.from_numpy(task['context'])[None], torch.from_numpy(task['probability'])[None])
    loss = torch.nn.functional.binary_cross_entropy(prediction, torch.from_numpy(task['labels'])[None])
    loss.backward()
    optimizer.step()
    np.testing.assert_array_equal(context, task['context'])
    np.testing.assert_array_equal(probability, task['probability'])


if __name__ == '__main__':
    tests = sorted((name, value) for name, value in list(globals().items()) if name.startswith('test_'))
    for name, test in tests:
        test()
        print(name+' passed')
    print(str(len(tests))+' mixture boundary checks passed')
