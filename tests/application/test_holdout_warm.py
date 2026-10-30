import numpy as np
import pytest

from mamex_robustness.data.loader import Loader4MM


def test_off_by_default_keeps_every_training_interaction(env):
    base = Loader4MM(env)
    assert len(base.heldout_warm_index) == 0
    assert sum(len(v) for v in base.train_data.values()) == 30 * 3


def test_heldout_items_lose_their_training_interactions_only(env):
    base = Loader4MM(env)
    env.args.holdout_warm = 0.2
    env.args.holdout_seed = 1
    held = Loader4MM(env)
    dropped = set(held.heldout_warm_index.tolist())
    assert len(dropped) == 6 and dropped <= set(held.warm_item_index.tolist())
    assert all(i not in dropped for items in held.train_data.values() for i in items)
    kept_before = sum(1 for items in base.train_data.values() for i in items if i not in dropped)
    assert sum(len(v) for v in held.train_data.values()) == kept_before
    assert held.val_data == base.val_data and held.test_data == base.test_data


def test_selection_is_seeded(env):
    env.args.holdout_warm = 0.2
    env.args.holdout_seed = 3
    assert np.array_equal(Loader4MM(env).heldout_warm_index, Loader4MM(env).heldout_warm_index)


def test_probe_reports_three_groups(session):
    session.env.args.holdout_warm = 0.0
    session.dataset.heldout_warm_index = np.asarray([0, 1, 2])
    session.train(1)
    out = session.holdout_probe()
    assert out['n_heldout_warm'] == 3 and out['n_trained_warm'] == 27
    for group in ('trained_warm', 'heldout_warm', 'cold'):
        assert 0.0 <= out[group]['w_t'] <= 1.0
