import json

import pytest

from mamex_robustness.reporting.make_tables import build_tables, main

BLOCKS = ('clean', 'clean_vgr', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')


def metrics(r20, n20, r10=None, n10=None):
    return {'hr': {'10': r10 or r20, '20': r20}, 'recall': {'10': r10 or r20, '20': r20},
            'ndcg': {'10': n10 or n20, '20': n20}}


def run(mode, seed, values, rate=0.4, selected=None, train_s=100.0):
    """values: {block: (recall20, ndcg20)}"""
    report = {b: metrics(*values.get(b, (0.0, 0.0))) for b in BLOCKS}
    report['gate'] = {'clean': {'w_v': 0.45, 'w_t': 0.55}, 'noisy': {'w_v': 0.40, 'w_t': 0.60},
                      'noisy_vgr': {'w_v': 0.52, 'w_t': 0.48}}
    report['cost'] = {'forward_ms_plain': 6.0, 'forward_ms_vgr': 7.0, 'gpu_mem_mb': 470.0}
    if selected:
        report['gamma_selection'] = {'grid': [1.0, 4.0], 'val_ndcg': {'1.0': 0.1, '4.0': 0.2}, 'selected': selected}
    args = {'mode': mode, 'seed': seed, 'noise_rate': rate, 'noise_type': 'hashtag', 'vgr_gamma': selected or 8.0,
            'suffix': 'default', 'train_with_vgr': 0}
    return {'args': args, 'report': report, 'best_epoch': 100, 'train_seconds': train_s}


def write(tmp_path, runs):
    for i, r in enumerate(runs):
        (tmp_path / f'r{i}.json').write_text(json.dumps(r))


@pytest.fixture
def full(tmp_path):
    runs = []
    for seed, (c, n, v) in enumerate([(0.10, 0.06, 0.07), (0.12, 0.08, 0.10), (0.14, 0.10, 0.12)]):
        vals = {'clean': (c, c / 2), 'clean_vgr': (c - 0.01, c / 2), 'noisy': (n, n / 2), 'noisy_vgr': (v, v / 2),
                'noisy_vgr_shuffled': (n, n / 2), 'noisy_vgr_constant': (n + 0.001, n / 2)}
        runs.append(run('baseline', seed, vals, rate=0.0))
        runs.append(run('stress_test', seed, vals))
        runs.append(run('improved', seed, vals, selected=4.0))
    write(tmp_path, runs)
    return tmp_path


def row(table, name):
    return next(r for r in table['rows'] if r['label'] == name)


def test_main_table_rows_come_from_the_right_files(full):
    t = build_tables(full)['accuracy']
    assert row(t, 'Clean baseline')['recall@20'] == pytest.approx((0.12, 0.02))
    assert row(t, 'Stress test (noisy text)')['recall@20'][0] == pytest.approx(0.08)
    assert row(t, 'MAMEX-VGR (noisy text)')['recall@20'][0] == pytest.approx(0.0966667, abs=1e-6)
    assert row(t, 'MAMEX-VGR (noisy text)')['n'] == 3


def test_controls_and_clean_cost_rows_exist(full):
    t = build_tables(full)['accuracy']
    labels = [r['label'] for r in t['rows']]
    assert 'VGR control: shuffled cosine (noisy text)' in labels
    assert 'VGR control: constant factor (noisy text)' in labels
    assert 'MAMEX-VGR (clean text)' in labels


def test_paired_differences_are_per_seed_and_signed(full):
    p = build_tables(full)['paired']
    d = p['noisy_vgr - noisy']['recall@20']
    assert d['per_seed'] == pytest.approx([0.07 - 0.06, 0.10 - 0.08, 0.12 - 0.10])
    assert d['positive'] == 3 and d['n'] == 3
    assert p['noisy_vgr - noisy_vgr_shuffled']['recall@20']['n'] == 3


def test_cost_and_selected_gamma_are_reported(full):
    t = build_tables(full)
    assert t['cost']['forward_ms_vgr'][0] == pytest.approx(7.0)
    assert t['cost']['train_seconds'][0] == pytest.approx(100.0)
    assert t['gamma']['selected'] == [4.0, 4.0, 4.0]


def test_missing_groups_are_pending_not_invented(tmp_path):
    write(tmp_path, [run('baseline', 0, {'clean': (0.1, 0.05)}, rate=0.0)])
    t = build_tables(tmp_path)
    assert [r['label'] for r in t['accuracy']['rows']] == ['Clean baseline']
    assert t['paired'] == {}


def test_main_writes_markdown_with_published_numbers_kept_apart(full, tmp_path):
    out = tmp_path / 'tables.md'
    main(['--results_dir', str(full), '--out', str(out)])
    text = out.read_text()
    published, measured = text.split('## Measured')[0], text.split('## Measured')[1]
    assert '## Published' in published and '0.0771' in published
    assert '0.120 ± 0.020' in measured and '0.0771' not in measured
    assert '| Clean baseline |' in measured and '## Paired differences' in text


def sweep_runs(seed=7):
    vals = lambda c, n, v: {'clean': (c, c / 2), 'clean_vgr': (c, c / 2), 'noisy': (n, n / 2), 'noisy_vgr': (v, v / 2),
                            'noisy_vgr_shuffled': (n, n / 2), 'noisy_vgr_constant': (n, n / 2)}
    runs = [run('stress_test', seed, vals(0.10, 0.05, 0.05), rate=0.8),
            run('improved', seed, vals(0.10, 0.05, 0.07), rate=0.8, selected=2.0)]
    for g, v in [(1.0, 0.08), (4.0, 0.06)]:
        r = run('improved', seed, vals(0.10, 0.05, v))
        r['args']['vgr_gamma'] = g
        r['args']['gamma_grid'] = None
        runs.append(r)
    return runs


def test_sensitivity_rows_come_from_the_single_seed_sweeps(tmp_path):
    from mamex_robustness.reporting.make_tables import build_sensitivity
    write(tmp_path, sweep_runs())
    s = build_sensitivity(tmp_path, seed=7)
    assert s['rates'] == [{'rate': 0.8, 'clean': 0.10, 'noisy': 0.05, 'noisy_vgr': 0.07, 'selected': 2.0}]
    assert [(g['gamma'], g['noisy_vgr']) for g in s['gammas']] == [(1.0, 0.08), (4.0, 0.06)]
    assert [(g['shuffled'], g['constant']) for g in s['gammas']] == [(0.05, 0.05), (0.05, 0.05)]


def test_sensitivity_is_empty_when_no_sweep_was_run(full):
    from mamex_robustness.reporting.make_tables import build_sensitivity
    s = build_sensitivity(full, seed=0)
    assert s['gammas'] == []


def test_rendered_sensitivity_is_labelled_single_seed(tmp_path):
    out = tmp_path / 'tables.md'
    write(tmp_path, sweep_runs())
    main(['--results_dir', str(tmp_path), '--out', str(out), '--sweep_seed', '7'])
    text = out.read_text()
    assert 'single seed' in text and '| 0.8 |' in text and '| 4.0 |' in text and 'shuffled cosine' in text


def aug_run(seed, values, selected=1.0):
    r = run('improved', seed, values, selected=selected)
    r['args'].update(suffix='noise_aug', train_noise_rate=0.4)
    return r


@pytest.fixture
def with_ablation(tmp_path):
    runs = []
    for seed in (1, 2, 3):
        base = {'clean': (0.10, 0.05), 'clean_vgr': (0.10, 0.05), 'noisy': (0.08, 0.04), 'noisy_vgr': (0.09, 0.045),
                'noisy_vgr_shuffled': (0.08, 0.04), 'noisy_vgr_constant': (0.08, 0.04)}
        runs += [run('baseline', seed, base, rate=0.0), run('stress_test', seed, base), run('improved', seed, base, selected=1.0)]
        aug = {'clean': (0.095, 0.05), 'clean_vgr': (0.095, 0.05), 'noisy': (0.09, 0.045), 'noisy_vgr': (0.0915, 0.046),
               'noisy_vgr_shuffled': (0.09, 0.045), 'noisy_vgr_constant': (0.09, 0.045)}
        runs.append(aug_run(seed, aug))
    write(tmp_path, runs)
    return tmp_path


def test_ablation_is_excluded_from_the_main_tables(with_ablation):
    rows = {r['label']: r for r in build_tables(str(with_ablation))['accuracy']['rows']}
    assert rows['Stress test (noisy text)']['n'] == 3 and rows['MAMEX-VGR (noisy text)']['n'] == 3


def test_ablation_pairs_each_seed_with_the_clean_trained_runs(with_ablation):
    from mamex_robustness.reporting.make_tables import build_ablation
    ab = build_ablation(str(with_ablation))
    assert ab['n'] == 3
    gain = ab['paired']['noisy: noise-augmented - stress test']['recall@20']
    assert gain['mean'] == pytest.approx(0.01) and gain['positive'] == 3
    cost = ab['paired']['clean: noise-augmented - clean baseline']['recall@20']
    assert cost['mean'] == pytest.approx(-0.005) and cost['positive'] == 0
    vgr = ab['paired']['VGR gain on noisy text: noise-augmented - clean-trained']['recall@20']
    assert vgr['mean'] == pytest.approx(0.0015 - 0.01)


def test_ablation_is_empty_without_augmented_runs(tmp_path):
    from mamex_robustness.reporting.make_tables import build_ablation
    assert build_ablation(str(tmp_path)) is None


def test_main_writes_the_ablation_section_only_when_runs_exist(with_ablation, tmp_path):
    out = tmp_path / 'tables.md'
    main(['--results_dir', str(with_ablation), '--out', str(out)])
    assert '## Noise-augmented training' in out.read_text()
