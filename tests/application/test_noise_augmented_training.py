import numpy as np
import pytest
import torch


@pytest.fixture
def seen(session):
    """Run the training loop and record the text the model used inside each train_epoch and each validation call."""
    session.env.args.train_noise_rate = 0.5
    session.env.args.noise_type = 'hashtag'
    session.env.args.noise_seed = 7
    record = {'train': [], 'val': []}
    original_epoch, original_test = session.train_epoch, session.test

    def train_epoch():
        text = session.model._eval_text
        record['train'].append(None if text is None else text.clone())
        return original_epoch()

    def test(*args, **kwargs):
        record['val'].append(session.model._eval_text)
        return original_test(*args, **kwargs)

    session.train_epoch, session.test = train_epoch, test
    session.train(3)
    return record


def test_rate_zero_trains_on_clean_text(session):
    session.env.args.train_noise_rate = 0.0
    texts = []
    original = session.train_epoch
    session.train_epoch = lambda: (texts.append(session.model._eval_text), original())[1]
    session.train(2)
    assert texts == [None, None]


def test_unset_rate_trains_on_clean_text(session):
    # Existing runs and tests do not define train_noise_rate at all.
    assert not hasattr(session.env.args, 'train_noise_rate')
    texts = []
    original = session.train_epoch
    session.train_epoch = lambda: (texts.append(session.model._eval_text), original())[1]
    session.train(1)
    assert texts == [None]


def test_training_text_corrupts_only_the_requested_share_of_warm_items(seen, session):
    clean = session.model.text_feat
    n_warm = len(session.dataset.warm_item_index)
    for text in seen['train']:
        changed = ~torch.isclose(text, clean).all(dim=1)
        assert int(changed.sum()) == round(0.5 * n_warm)
        assert not changed[torch.as_tensor(session.dataset.cold_item_index)].any()  # cold items are never seen in training


def test_mask_is_redrawn_every_epoch(seen):
    a, b = seen['train'][0], seen['train'][1]
    assert not torch.equal(a, b)


def test_validation_and_model_state_stay_clean(seen, session):
    assert all(text is None for text in seen['val'])
    assert session.model._eval_text is None


def test_training_mask_differs_from_the_test_mask(session):
    session.env.args.noise_seed = 7
    session.env.args.train_noise_rate = 0.5
    cold_noisy = session.dataset.noisy_text_feat(0.5, 'hashtag', 7)
    warm_noisy = session.dataset.train_noisy_text_feat(0.5, 'hashtag', 7 + session.TRAIN_NOISE_SEED_OFFSET)
    cold = np.asarray(session.dataset.cold_item_index)
    warm = np.asarray(session.dataset.warm_item_index)
    clean = session.dataset.text_feat
    assert (cold_noisy[warm] == clean[warm]).all()   # test noise never touches warm rows
    assert (warm_noisy[cold] == clean[cold]).all()   # training noise never touches cold rows
