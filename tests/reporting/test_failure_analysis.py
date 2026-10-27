import json

import numpy as np
import pytest

from mamex_robustness.application.diagnostics import metrics_from_ranks
from mamex_robustness.reporting.failure_analysis import analyze_run, main

# 4 users, 2 test items each. Items 10, 11 are corrupted; 12, 13 are not.
USERS = np.repeat([0, 1, 2, 3], 2)
ITEMS = np.array([10, 12, 12, 13, 10, 13, 12, 11])
RANKS = {
    'clean': np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    'clean_vgr': np.array([1, 1, 1, 1, 1, 1, 1, 30]),           # hurts user 3's item 11 on clean text
    'noisy': np.array([30, 1, 1, 1, 30, 1, 1, 30]),
    'noisy_vgr': np.array([5, 1, 1, 1, 30, 3, 1, 1]),            # helps u0/i10 and u3/i11; u2/i13 slips 1 -> 3; u1 unchanged
    'noisy_vgr_shuffled': np.array([30, 1, 1, 1, 30, 1, 1, 30]),
    'noisy_vgr_constant': np.array([30, 1, 1, 1, 30, 1, 1, 30]),
}


def make_run(tmp_path, ranks=RANKS, k=20):
    np.savez(tmp_path / 'run.npz', users=USERS, items=ITEMS, corrupted_items=np.array([10, 11]),
             **{f'ranks_{n}': r for n, r in ranks.items()})
    report = {}
    for name, r in ranks.items():
        _, rec, nd = metrics_from_ranks(USERS, r, k)
        report[name] = {'recall': {str(k): float(rec.mean())}, 'ndcg': {str(k): float(nd.mean())}}
    (tmp_path / 'run.json').write_text(json.dumps({'args': {'seed': 7, 'mode': 'improved', 'gamma_grid': [1.0, 2.0], 'noise_rate': 0.4}, 'report': report}))
    return tmp_path / 'run.npz', tmp_path / 'run.json'


def test_user_level_help_and_harm_counts(tmp_path):
    out = analyze_run(*make_run(tmp_path), k=20)['users']['noisy_vgr vs noisy']
    assert out['n_users'] == 4
    # helped/hurt are judged by NDCG@k, which is rank-sensitive: u0 and u3 gain, u2's item slipping 1 -> 3 loses
    # (its recall is unchanged because rank 3 is still inside the top 20), u1 is unchanged
    assert (out['helped'], out['hurt'], out['unchanged']) == (2, 1, 1)
    assert out['mean_recall_diff'] == pytest.approx((0.5 + 0.0 + 0.0 + 0.5) / 4)


def test_users_split_by_exposure_to_corrupted_items(tmp_path):
    groups = analyze_run(*make_run(tmp_path), k=20)['users']['noisy_vgr vs noisy']['by_exposure']
    assert groups['exposed']['n_users'] == 3 and groups['not_exposed']['n_users'] == 1  # only user 1 has no corrupted positive


def test_item_level_rank_changes_and_extremes(tmp_path):
    items = analyze_run(*make_run(tmp_path), k=20)['items']['noisy_vgr vs noisy']
    assert items['corrupted']['mean_rank_gain'] > 0                 # grounding moves corrupted items up
    assert items['not_corrupted']['mean_rank_gain'] < 0             # and costs a little elsewhere (13: rank 1 -> 3)
    assert items['most_helped'][0][0] in (10, 11) and items['most_hurt'][0][0] == 13


def test_clean_text_harm_is_reported_separately(tmp_path):
    out = analyze_run(*make_run(tmp_path), k=20)['users']['clean_vgr vs clean']
    assert out['hurt'] == 1 and out['helped'] == 0


def test_inconsistent_diagnostics_are_rejected(tmp_path):
    npz, js = make_run(tmp_path)
    data = json.loads(js.read_text())
    data['report']['noisy']['recall']['20'] += 0.01
    js.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='do not match'):
        analyze_run(npz, js, k=20)


def test_main_writes_markdown(tmp_path):
    make_run(tmp_path)
    (tmp_path / 'diagnostics').mkdir()
    (tmp_path / 'run.npz').rename(tmp_path / 'diagnostics' / 'run.npz')
    out = tmp_path / 'fa.md'
    main(['--results_dir', str(tmp_path), '--out', str(out)])
    text = out.read_text()
    assert '## Users' in text and '## Items' in text and 'helped' in text


def test_float32_level_ndcg_differences_are_tolerated_but_real_ones_are_not(tmp_path):
    npz, js = make_run(tmp_path)
    data = json.loads(js.read_text())
    data['report']['noisy']['ndcg']['20'] += 3e-6   # near-tie ordering noise between faiss and numpy
    js.write_text(json.dumps(data))
    analyze_run(npz, js, k=20)
    data['report']['noisy']['ndcg']['20'] += 1e-3
    js.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='do not match'):
        analyze_run(npz, js, k=20)


def test_recall_must_match_exactly(tmp_path):
    npz, js = make_run(tmp_path)
    data = json.loads(js.read_text())
    data['report']['noisy']['recall']['20'] += 3e-6
    js.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='do not match'):
        analyze_run(npz, js, k=20)
