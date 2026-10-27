import json

import pytest

from mamex_robustness.reporting.plot_results import aggregate, main


def result(mode, seed, recall, ndcg, rate=0.4, gamma=8.0, gate=None, suffix='default', selected=False):
    block = {'hr': {'20': recall}, 'recall': {'20': recall}, 'ndcg': {'20': ndcg}}
    g = gate or {'w_v': 0.4, 'w_t': 0.6}
    report = {name: block for name in ('clean', 'clean_vgr', 'noisy', 'noisy_vgr')}
    report['gate'] = {name: g for name in ('clean', 'noisy', 'noisy_vgr')}
    if selected:
        report['gamma_selection'] = {'grid': [gamma], 'val_ndcg': {str(gamma): 0.1}, 'selected': gamma}
    args = {'mode': mode, 'seed': seed, 'noise_rate': rate, 'noise_type': 'hashtag',
            'vgr_gamma': gamma, 'suffix': suffix}
    return {'args': args, 'report': report}


def write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


@pytest.fixture
def results_dir(tmp_path):
    for seed, (b, s, i) in enumerate([(0.10, 0.04, 0.07), (0.12, 0.06, 0.09), (0.14, 0.05, 0.08)]):
        write(tmp_path, f'b{seed}.json', result('baseline', seed, b, b, rate=0.0))
        write(tmp_path, f's{seed}.json', result('stress_test', seed, s, s,
                                                gate={'w_v': 0.2, 'w_t': 0.8}))
        write(tmp_path, f'i{seed}.json', result('improved', seed, i, i, selected=True,
                                                gate={'w_v': 0.7, 'w_t': 0.3}))
    return tmp_path


def test_aggregate_mean_and_std_over_seeds(results_dir):
    acc = aggregate(results_dir, k=20)['accuracy']
    assert acc['Clean baseline']['recall'][0] == pytest.approx(0.12)
    assert acc['Clean baseline']['recall'][1] == pytest.approx(0.02)  # sample std, ddof=1
    assert acc['Stress test']['n'] == 3
    assert acc['MAMEX-VGR']['ndcg'][0] == pytest.approx(0.08)


def test_aggregate_gate_uses_matching_blocks(results_dir):
    gate = aggregate(results_dir, k=20)['gate']
    assert gate['Noisy text']['w_t'][0] == pytest.approx(0.8)
    assert gate['Noisy text + VGR']['w_t'][0] == pytest.approx(0.3)
    assert gate['Clean text']['w_t'][0] == pytest.approx(0.8)  # taken from the stress_test files


def test_aggregate_ignores_other_gamma_rate_and_suffix(results_dir):
    write(results_dir, 'sweep.json', result('improved', 9, 0.9, 0.9, gamma=2.0))
    write(results_dir, 'rate.json', result('stress_test', 9, 0.9, 0.9, rate=0.8))
    write(results_dir, 'abl.json', result('stress_test', 9, 0.9, 0.9, suffix='noise_aug'))
    acc = aggregate(results_dir, k=20)['accuracy']
    assert acc['MAMEX-VGR']['n'] == 3 and acc['Stress test']['n'] == 3


def test_default_uses_validation_selected_gamma_runs_only(results_dir):
    write(results_dir, 'manual8.json', result('improved', 9, 0.9, 0.9, gamma=8.0))
    acc = aggregate(results_dir, k=20)['accuracy']
    assert acc['MAMEX-VGR']['n'] == 3 and acc['MAMEX-VGR']['recall'][0] == pytest.approx(0.08)


def test_explicit_gamma_selects_the_manual_runs(results_dir):
    write(results_dir, 'manual2.json', result('improved', 9, 0.5, 0.5, gamma=2.0))
    acc = aggregate(results_dir, k=20, vgr_gamma=2.0)['accuracy']
    assert acc['MAMEX-VGR']['n'] == 1 and acc['MAMEX-VGR']['recall'][0] == pytest.approx(0.5)


def test_missing_mode_is_omitted_not_invented(tmp_path):
    write(tmp_path, 'b.json', result('baseline', 0, 0.1, 0.1, rate=0.0))
    acc = aggregate(tmp_path, k=20)['accuracy']
    assert list(acc) == ['Clean baseline']


def test_main_writes_both_figures(results_dir):
    main(['--results_dir', str(results_dir), '--k', '20'])
    for name in ('accuracy_comparison.png', 'gating_weight_shift.png'):
        assert (results_dir / name).stat().st_size > 1000


def test_main_without_results_exits_with_message(tmp_path):
    with pytest.raises(SystemExit):
        main(['--results_dir', str(tmp_path)])
