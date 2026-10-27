import numpy as np
import pytest

from mamex_robustness.application.diagnostics import metrics_from_ranks

BLOCKS = {'clean', 'clean_vgr', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant'}


def test_metrics_from_ranks_match_session_test(session):
    d = session.diagnostics(noise_rate=0.0, noise_type='hashtag', noise_seed=1)
    _, recall, ndcg, _ = session.test(mode='test', top_list=[3, 5])
    for k in (3, 5):
        _, r, n = metrics_from_ranks(d['users'], d['ranks']['clean'], k)
        assert r.mean() == pytest.approx(recall[k], abs=1e-6)
        assert n.mean() == pytest.approx(ndcg[k], abs=1e-6)


def test_noisy_blocks_match_noisy_session_test(session):
    noisy = session.dataset.noisy_text_feat(0.5, 'hashtag', seed=3)
    d = session.diagnostics(noise_rate=0.5, noise_type='hashtag', noise_seed=3)
    _, recall, _, _ = session.test(mode='test', top_list=[5], noisy_text=noisy, use_vgr=True)
    assert metrics_from_ranks(d['users'], d['ranks']['noisy_vgr'], 5)[1].mean() == pytest.approx(recall[5], abs=1e-6)


def test_diagnostics_layout(session):
    d = session.diagnostics(noise_rate=0.5, noise_type='hashtag', noise_seed=3)
    assert set(d['ranks']) == BLOCKS
    n_pairs = len(d['users'])
    assert len(d['items']) == n_pairs and all(len(r) == n_pairs for r in d['ranks'].values())
    assert set(d['items']) <= set(session.dataset.cold_item_index.tolist())
    assert set(d['corrupted_items']) <= set(session.dataset.cold_item_index.tolist())
    assert len(d['corrupted_items']) == 5  # round(0.5 * 10 cold items)
    assert d['ranks']['clean'].min() >= 1


def test_ranks_do_not_change_model_state(session):
    before = {k: v.clone() for k, v in session.model.state_dict().items()}
    use_vgr, gamma = session.model.use_vgr, session.model.vgr_gamma
    session.diagnostics(noise_rate=0.5, noise_type='hashtag', noise_seed=3)
    assert session.model.use_vgr == use_vgr and session.model.vgr_gamma == gamma
    assert all((before[k] == v).all() for k, v in session.model.state_dict().items())
