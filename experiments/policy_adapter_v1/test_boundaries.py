import importlib.util
from pathlib import Path
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('policy_pilot', Path(__file__).with_name('pilot.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
PLAN = dict(namespace=1601, source_labels=128, target_labels=64, queries=128)


def sample():
    return p.world(119000, 'development', PLAN)[1]


def model():
    torch.manual_seed(13)
    result = p.ContextAdapter()
    # Exercise a non-identity model; the initialized final layer is deliberately zero.
    torch.nn.init.normal_(result.hyper[-1].weight, std=.02)
    return result


def run(m, e):
    return p.predict(m, e['source_f'], e['source_y'], e['support_f'], e['support_y'], e['query_f'])


def test_query_labels_oracles_and_true_policies_do_not_enter_prediction():
    e, m = sample(), model()
    first = run(m, e)
    e['query_y'] = 1-e['query_y']
    e['oracle'] = 1-e['oracle']
    e['target_policy'] = {'fake':999}
    np.testing.assert_array_equal(first, run(m, e))


def test_hidden_values_are_ignored():
    e = sample()
    first = p.features(e['query'], e['frozen'])
    e['query']['x'][e['query']['mask'].astype(bool)] = 999999.
    np.testing.assert_array_equal(first, p.features(e['query'], e['frozen']))


def test_support_permutations_preserve_predictions():
    e, m = sample(), model()
    first = run(m, e)
    for domain in ('source', 'support'):
        e[domain+'_f'] = e[domain+'_f'][::-1].copy()
        e[domain+'_y'] = e[domain+'_y'][::-1].copy()
    np.testing.assert_allclose(first, run(m, e), atol=2e-7, rtol=2e-6)


def test_queries_are_independent_and_order_equivariant():
    e, m = sample(), model()
    first = run(m, e)
    e['query_f'] = e['query_f'][::-1].copy()
    np.testing.assert_allclose(first[::-1], run(m, e), atol=1e-7, rtol=1e-6)
    e['query_f'] = e['query_f'][:1]
    np.testing.assert_allclose(first[-1:], run(m, e), atol=1e-7, rtol=1e-6)


def test_paired_policies_share_outcomes_not_observations_and_backbone_stays_fixed():
    episodes = p.world(119001, 'development', PLAN)
    for e in episodes:
        np.testing.assert_array_equal(episodes[0]['support_y'], e['support_y'])
        np.testing.assert_array_equal(episodes[0]['query_y'], e['query_y'])
        np.testing.assert_array_equal(episodes[0]['frozen'], e['frozen'])
    e, m = episodes[0], model()
    frozen = e['frozen'].copy()
    inputs = p.tensor_batch([e])
    torch.nn.functional.binary_cross_entropy_with_logits(m(*inputs[:5]), inputs[5]).backward()
    np.testing.assert_array_equal(frozen, e['frozen'])


def test_exact_policy_probabilities_and_posterior():
    policy = dict(active=7, signal=2., rate=0., block=0., value=0.)
    probs = p.policy_probs(np.zeros(16), np.ones(16), policy)
    np.testing.assert_allclose(probs.sum(1), 1., atol=1e-14)
    likelihood = probs[0]/p.policy_probs(np.zeros(16), np.zeros(16), policy)[0]
    # At uniform rates, normalizers cancel and the log likelihood ratio is signal*parity.
    np.testing.assert_allclose(np.log(likelihood), 2.*p.SIGNS[:, 7], atol=1e-14)
