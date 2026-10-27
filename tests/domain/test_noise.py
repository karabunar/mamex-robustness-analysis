import numpy as np
import pytest

from mamex_robustness.domain.noise import CorruptionSpec, TextNoiseSimulator

TYPES = ['hashtag', 'shuffle', 'gaussian']


@pytest.fixture
def feats():
    rng = np.random.RandomState(1)
    feat = rng.randn(200, 32).astype(np.float32)
    return feat, np.arange(150, 200), np.arange(0, 150)  # feat, cold, warm


def run(feats, rate, noise_type='hashtag', seed=2025):
    feat, cold, warm = feats
    return TextNoiseSimulator(CorruptionSpec(rate, noise_type, seed))(feat, cold, warm)


def changed_rows(feat, out):
    return set(np.where((feat != out).any(axis=1))[0].tolist())


@pytest.mark.parametrize('noise_type', TYPES)
def test_rate_zero_is_identity(feats, noise_type):
    assert np.array_equal(run(feats, 0.0, noise_type), feats[0])


@pytest.mark.parametrize('noise_type', TYPES)
def test_rate_one_corrupts_all_cold_and_no_warm(feats, noise_type):
    feat, cold, warm = feats
    out = run(feats, 1.0, noise_type)
    assert changed_rows(feat, out) == set(cold.tolist())


@pytest.mark.parametrize('noise_type', TYPES)
def test_rate_fraction_of_cold_items_changed(feats, noise_type):
    assert len(changed_rows(feats[0], run(feats, 0.4, noise_type))) == 20


def test_mask_is_deterministic_and_independent_of_noise_type(feats):
    sets = [changed_rows(feats[0], run(feats, 0.4, t)) for t in TYPES]
    assert sets[0] == sets[1] == sets[2]
    assert sets[0] == changed_rows(feats[0], run(feats, 0.4, 'hashtag'))


def test_different_seed_changes_mask(feats):
    a = changed_rows(feats[0], run(feats, 0.4, seed=1))
    b = changed_rows(feats[0], run(feats, 0.4, seed=2))
    assert a != b


def test_input_not_mutated(feats):
    before = feats[0].copy()
    run(feats, 0.5)
    assert np.array_equal(feats[0], before)


def test_hashtag_rows_are_closer_to_a_shared_prototype_than_to_their_original(feats):
    feat, cold, warm = feats
    out = run(feats, 1.0, 'hashtag')

    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    to_original = (unit(out[cold]) * unit(feat[cold])).sum(1)
    to_nearest_warm = (unit(out[cold]) @ unit(feat[warm]).T).max(1)
    assert (to_nearest_warm > to_original).all()


@pytest.mark.parametrize('noise_type', TYPES)
def test_zero_vector_row_gives_finite_output(feats, noise_type):
    feat, cold, warm = feats
    feat = feat.copy()
    feat[cold[0]] = 0.0
    out = TextNoiseSimulator(CorruptionSpec(1.0, noise_type, 0))(feat, cold, warm)
    assert np.isfinite(out).all()


def test_unknown_noise_type_rejected():
    with pytest.raises(ValueError):
        CorruptionSpec(0.4, 'bogus')


def test_rate_out_of_range_rejected():
    with pytest.raises(ValueError):
        CorruptionSpec(1.5, 'hashtag')
