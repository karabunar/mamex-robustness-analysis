import pytest


@pytest.fixture
def cause(session):
    session.train(2)
    return session.gate_cause()


def test_reports_every_intervention(cause):
    assert {'zero_gap', 'warm_norm', 'cold_norm', 'w_t_warm', 'w_t_cold', 'w_t_cold_scaled_both', 'w_t_cold_scaled_image',
            'w_t_cold_scaled_text', 'w_t_cold_replaced_by_warm', 'w_t_cold_image_replaced_by_warm',
            'w_t_cold_text_replaced_by_warm'} <= set(cause)


def test_weights_are_probabilities(cause):
    for key, value in cause.items():
        if key.startswith('w_t'):
            assert 0.0 <= value <= 1.0


def test_unchanged_cold_weight_matches_the_gate_probe(cause, session):
    probe = session._gate_mean(None, False, session.dataset.cold_item_index)
    assert cause['w_t_cold'] == pytest.approx(probe['w_t'], abs=1e-5)


def test_replacement_is_seeded(session):
    session.train(1)
    assert session.gate_cause(seed=3) == session.gate_cause(seed=3)


def test_leaves_the_model_clean(cause, session):
    assert session.model._eval_text is None
