import pytest
import torch


def test_saved_weights_are_the_best_state(session, tmp_path):
    session.train(2)
    path = tmp_path / 'sub' / 'best.pt'
    session.save_best(str(path))
    saved = torch.load(path)
    assert saved.keys() == session.best_state.keys()
    for key, value in session.best_state.items():
        assert torch.equal(saved[key], value)


def test_saved_weights_load_into_a_fresh_model(session, tmp_path):
    session.train(2)
    path = tmp_path / 'best.pt'
    session.save_best(str(path))
    session.model.load_state_dict(torch.load(path))


def test_nothing_to_save_before_training(session, tmp_path):
    with pytest.raises(RuntimeError):
        session.save_best(str(tmp_path / 'best.pt'))
    assert not (tmp_path / 'best.pt').exists()
