import pytest


@pytest.fixture
def probe(session):
    session.train(2)
    return session.representation_probe()


def test_probe_reports_both_groups(probe):
    for group in ('warm', 'cold'):
        assert set(probe[group]) == {'norm_v', 'norm_t', 'distance', 'logit_gap', 'w_t'}


def test_ratio_is_cold_over_warm_distance(probe):
    assert probe['distance_ratio'] == pytest.approx(probe['cold']['distance'] / probe['warm']['distance'])


def test_probe_gate_weight_matches_the_gate_probe(probe, session):
    cold = session._gate_mean(None, False, session.dataset.cold_item_index)
    assert probe['cold']['w_t'] == pytest.approx(cold['w_t'], abs=1e-5)


def test_probe_leaves_the_model_clean(probe, session):
    assert session.model._eval_text is None
