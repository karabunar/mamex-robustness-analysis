import pytest


@pytest.fixture
def probe(session):
    session.env.args.noise_type = 'hashtag'
    session.env.args.noise_seed = 3
    session.train(2)
    return session.gate_probe(0.5, 'hashtag', 3)


def test_probe_splits_warm_items_by_the_corruption_mask(probe, session):
    n_warm = len(session.dataset.warm_item_index)
    assert probe['n_warm_corrupted'] == round(0.5 * n_warm)
    assert probe['n_warm_corrupted'] + probe['n_warm_uncorrupted'] == n_warm


def test_probe_weights_are_a_distribution(probe):
    for name in ('warm_clean', 'warm_corrupted', 'warm_uncorrupted', 'cold_clean'):
        assert probe[name]['w_v'] + probe[name]['w_t'] == pytest.approx(1.0, abs=1e-5)


def test_uncorrupted_warm_items_keep_their_clean_gate(probe, session):
    # The gate looks at one item at a time, so an item whose text is not corrupted gets the same weights as without any noise.
    clean_all = session._gate_mean(None, False, session.dataset.warm_item_index)
    assert probe['warm_clean']['w_t'] == pytest.approx(clean_all['w_t'], abs=1e-6)


def test_probe_leaves_the_model_clean(probe, session):
    assert session.model._eval_text is None


def test_probe_uses_a_mask_no_training_epoch_used(session):
    session.env.args.noise_seed = 3
    epochs = {session.TRAIN_NOISE_SEED_OFFSET + 3 + e for e in range(1000)}
    assert session.PROBE_NOISE_SEED_OFFSET + 3 not in epochs
