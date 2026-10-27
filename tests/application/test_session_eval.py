GOLDEN_VAL_NDCG3 = 0.13333333333333333
import pytest
import torch


def test_default_val_call_is_unchanged(session):
    hr, recall, ndcg, _ = session.test(mode='val', top_list=[3, 5])
    assert set(hr) == {3, 5}
    assert ndcg[3] == pytest.approx(GOLDEN_VAL_NDCG3, abs=1e-6)


def test_noisy_eval_restores_model_state(session):
    noisy = session.dataset.noisy_text_feat(1.0, 'hashtag', seed=1)
    original = session.model.text_feat.clone()
    session.model.use_vgr = False
    session.test(mode='test', top_list=[3, 5], noisy_text=noisy, use_vgr=True)
    assert session.model.use_vgr is False
    assert torch.equal(session.model.text_feat, original)
    clean_after = session.test(mode='test', top_list=[3, 5])[2]
    clean_again = session.test(mode='test', top_list=[3, 5])[2]
    assert clean_after == clean_again


def test_noisy_eval_runs_in_eval_mode_and_is_deterministic(session):
    noisy = session.dataset.noisy_text_feat(0.5, 'hashtag', seed=1)
    session.model.train()
    a = session.test(mode='test', top_list=[3, 5], noisy_text=noisy)[2]
    session.model.train()
    b = session.test(mode='test', top_list=[3, 5], noisy_text=noisy)[2]
    assert a == b
    assert not session.model.training


def test_best_state_is_a_cpu_copy_of_the_best_epoch(session):
    assert session.best_state is None
    session.train(3)
    assert set(session.best_state) == set(session.model.state_dict())
    key = next(iter(session.best_state))
    snapshot = session.best_state[key].clone()
    with torch.no_grad():
        session.model.state_dict()[key].add_(1.0)
    assert torch.equal(session.best_state[key], snapshot)
    assert session.best_state[key].device.type == 'cpu'


def test_final_report_structure_and_gate_rows(session):
    session.train(2)
    report = session.final_report(noise_rate=0.5, noise_type='hashtag', noise_seed=1)
    for block in ('clean', 'noisy', 'noisy_vgr', 'clean_vgr'):
        assert set(report[block]) == {'hr', 'recall', 'ndcg'}
        assert set(report[block]['ndcg']) == {'3', '5'}
    for name in ('clean', 'noisy', 'noisy_vgr'):
        g = report['gate'][name]
        assert g['w_v'] + g['w_t'] == pytest.approx(1.0, abs=1e-5)
    assert report['gate']['noisy_vgr']['w_t'] < report['gate']['noisy']['w_t']
    assert report['cost']['forward_ms_vgr'] > 0 and report['cost']['forward_ms_plain'] > 0
    assert report['n_corrupted'] == 5
    assert session.model.use_vgr is False


def test_final_report_with_zero_noise_uses_all_cold_items(session):
    session.train(1)
    report = session.final_report(noise_rate=0.0, noise_type='hashtag', noise_seed=1)
    assert report['n_corrupted'] == 0
    assert report['gate']['clean']['w_v'] + report['gate']['clean']['w_t'] == pytest.approx(1.0, abs=1e-5)


def test_final_report_includes_control_blocks_and_restores_the_control(session):
    session.train(1)
    session.model.vgr_control = 'cosine'
    report = session.final_report(noise_rate=0.5, noise_type='hashtag', noise_seed=1)
    for block in ('noisy_vgr_shuffled', 'noisy_vgr_constant'):
        assert set(report[block]) == {'hr', 'recall', 'ndcg'}
    assert session.model.vgr_control == 'cosine'


def _stub_val_scores(session, score_by_gamma):
    calls = []

    def fake_test(mode='val', top_list=[50], noisy_text=None, use_vgr=None, vgr_control=None):
        calls.append({'mode': mode, 'noisy': noisy_text is not None, 'vgr': use_vgr,
                      'gamma': session.model.vgr_gamma})
        score = score_by_gamma[session.model.vgr_gamma]
        return {}, {}, {k: score for k in top_list}, 0.0

    session.test = fake_test
    return calls


def test_select_gamma_uses_only_noisy_validation_and_picks_the_best(session):
    session.model.vgr_gamma = 8.0
    calls = _stub_val_scores(session, {1.0: 0.1, 2.0: 0.3, 4.0: 0.5, 8.0: 0.2})
    result = session.select_gamma([1.0, 2.0, 4.0, 8.0], 0.4, 'hashtag', noise_seed=1)
    assert result['selected'] == 4.0
    assert result['val_ndcg'] == {'1.0': 0.1, '2.0': 0.3, '4.0': 0.5, '8.0': 0.2}
    assert all(c['mode'] == 'val' and c['noisy'] and c['vgr'] is True for c in calls)
    assert session.model.vgr_gamma == 8.0  # restored


def test_select_gamma_breaks_ties_toward_the_smaller_gamma(session):
    _stub_val_scores(session, {1.0: 0.2, 4.0: 0.2})
    assert session.select_gamma([4.0, 1.0], 0.4, 'hashtag', noise_seed=1)['selected'] == 1.0


def test_validation_noise_uses_a_different_mask_than_test_noise(session):
    cold = session.dataset.cold_item_index
    test_noise = session.dataset.noisy_text_feat(0.4, 'hashtag', 1)
    val_noise = session.dataset.noisy_text_feat(0.4, 'hashtag', session.VAL_NOISE_SEED_OFFSET + 1)
    changed = lambda a: set(((a != session.dataset.text_feat).any(axis=1)).nonzero()[0].tolist())
    assert changed(test_noise) != changed(val_noise)
    assert changed(val_noise) <= set(cold.tolist())


def test_final_report_with_grid_reports_selection_and_uses_the_selected_gamma(session):
    session.train(1)
    report = session.final_report(noise_rate=0.5, noise_type='hashtag', noise_seed=1,
                                  gamma_grid=[1.0, 8.0])
    selection = report['gamma_selection']
    assert selection['selected'] in (1.0, 8.0) and set(selection['val_ndcg']) == {'1.0', '8.0'}
    assert session.model.vgr_gamma == selection['selected']
    noisy = session.dataset.noisy_text_feat(0.5, 'hashtag', 1)
    assert report['noisy_vgr'] == session._metrics(noisy_text=noisy, use_vgr=True)


def test_final_report_without_grid_has_no_selection(session):
    session.train(1)
    assert 'gamma_selection' not in session.final_report(0.5, 'hashtag', 1)


def _use_vgr_seen_in_training(session, monkeypatch):
    seen = []
    original = session.train_epoch

    def spy():
        seen.append(session.model.use_vgr)
        return original()

    monkeypatch.setattr(session, 'train_epoch', spy)
    session.train(2)
    return seen


def test_training_runs_without_vgr_by_default_and_restores_eval_setting(session, monkeypatch):
    session.model.use_vgr = True  # what --mode improved sets for evaluation
    assert _use_vgr_seen_in_training(session, monkeypatch) == [False, False]
    assert session.model.use_vgr is True


def test_train_with_vgr_flag_turns_vgr_on_during_training(session, monkeypatch):
    session.env.args.train_with_vgr = 1
    session.model.use_vgr = False
    assert _use_vgr_seen_in_training(session, monkeypatch) == [True, True]
    assert session.model.use_vgr is False
