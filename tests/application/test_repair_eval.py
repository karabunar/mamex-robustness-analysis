import numpy as np
import pytest

from mamex_robustness.application import repair_eval


@pytest.fixture
def report(session):
    session.train(2)
    return repair_eval.evaluate_repairs(session, 0.5, 'hashtag', 3, k=5, ks=(3, 5))


def test_every_method_is_reported_in_every_condition(report):
    methods = {'unrepaired', 'vgr', 'm1', 'm1_random', 'm2', 'm2_random', 'm2_shuffled'}
    assert methods <= set(report['noisy']) and methods <= set(report['clean'])
    assert set(report['noisy_plus_vgr']) == {'m1', 'm2'} == set(report['clean_plus_vgr'])


def test_selected_parameters_come_from_the_grids(report):
    sel = report['selection']
    assert sel['m1']['param'] in repair_eval.LAM_GRID and sel['m1_random']['param'] in repair_eval.LAM_GRID
    assert tuple(sel['m2']['param']) in repair_eval.GATED_GRID
    assert sel['vgr']['gamma'] in repair_eval.GAMMA_GRID


def test_unrepaired_scores_equal_a_direct_evaluation(report, session):
    noisy = session.dataset.noisy_text_feat(0.5, 'hashtag', 3)
    _, recall, ndcg, _ = session.test(mode='test', top_list=[3, 5], noisy_text=noisy, use_vgr=False)
    assert report['noisy']['unrepaired']['recall'] == pytest.approx(float(recall[5]))
    _, recall, _, _ = session.test(mode='test', top_list=[3, 5], use_vgr=False)
    assert report['clean']['unrepaired']['recall'] == pytest.approx(float(recall[5]))


def test_detection_and_cost_are_reported(report, session):
    assert 0.0 <= report['auroc']['agreement'] <= 1.0 and 0.0 <= report['auroc']['vgr_cosine'] <= 1.0
    assert report['n_corrupted_cold'] == 5
    assert report['cost']['extra_mb'] > 0 and report['cost']['forward_ms'] > 0 and report['cost']['forward_ms_repaired'] > 0


def test_evaluation_leaves_the_model_as_it_was(report, session):
    assert session.model._eval_text is None and not session.model.use_vgr
