import json

import numpy as np
import pytest

from mamex_robustness.application.diagnostics import metrics_from_ranks
from mamex_robustness.reporting.ranking_metrics import bootstrap_means, main, user_metrics

# 3 users: u0 has test items at ranks 1 and 30, u1 at rank 4, u2 at rank 60.
USERS = np.array([0, 0, 1, 2])
RANKS = np.array([1, 30, 4, 60])


def test_success_counts_users_with_any_positive_in_top_k():
    m = user_metrics(USERS, RANKS, ks=(1, 5, 50), k_main=20)
    assert m['success@1'].tolist() == [1.0, 0.0, 0.0]
    assert m['success@5'].tolist() == [1.0, 1.0, 0.0]
    assert m['success@50'].tolist() == [1.0, 1.0, 0.0]   # u2's only item is at rank 60


def test_mrr_uses_each_users_best_rank():
    m = user_metrics(USERS, RANKS, ks=(1,), k_main=20)
    assert m['mrr'] == pytest.approx([1.0, 0.25, 1 / 60])


def test_recall_and_ndcg_agree_with_the_reported_definition():
    m = user_metrics(USERS, RANKS, ks=(1,), k_main=20)
    _, rec, nd = metrics_from_ranks(USERS, RANKS, 20)
    assert m['recall@20'] == pytest.approx(rec) and m['ndcg@20'] == pytest.approx(nd)


def test_bootstrap_interval_contains_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    values = {'a': rng.random((3, 200))}            # 3 seeds x 200 users
    out1 = bootstrap_means(values, n_boot=300, seed=1)
    out2 = bootstrap_means(values, n_boot=300, seed=1)
    lo, hi = out1['a']['ci']
    assert lo <= out1['a']['mean'] <= hi
    assert out1 == out2
    assert hi - lo < 0.2                             # 200 users: a narrow interval, not the whole [0, 1]


def test_paired_difference_uses_the_same_resampled_users():
    rng = np.random.default_rng(0)
    base = rng.random((2, 300))
    values = {'a': base, 'b': base + 0.05}           # b beats a for every user by exactly 0.05
    out = bootstrap_means(values, n_boot=200, seed=3, diffs=[('b', 'a')])
    d = out['b - a']
    assert d['mean'] == pytest.approx(0.05)
    assert d['ci'][0] == pytest.approx(0.05) and d['ci'][1] == pytest.approx(0.05)   # no spread: pairing cancels the sampling noise


def test_main_writes_a_table_per_condition(tmp_path):
    users = np.repeat(np.arange(50), 2)
    rng = np.random.default_rng(0)
    ranks = {n: rng.integers(1, 80, len(users)) for n in
             ('clean', 'clean_vgr', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant')}
    (tmp_path / 'diagnostics').mkdir()
    for seed in (1, 2):
        stem = f'improved_hashtag_r0.40_g1.0_seed{seed}'
        np.savez(tmp_path / 'diagnostics' / f'{stem}.npz', users=users, items=np.zeros(len(users), int),
                 corrupted_items=np.array([0]), **{f'ranks_{n}': r for n, r in ranks.items()})
        report = {}
        for n, r in ranks.items():
            _, rec, nd = metrics_from_ranks(users, r, 20)
            report[n] = {'recall': {'20': float(rec.mean())}, 'ndcg': {'20': float(nd.mean())}}
        (tmp_path / f'{stem}.json').write_text(json.dumps(
            {'args': {'seed': seed, 'mode': 'improved', 'gamma_grid': [1.0], 'noise_rate': 0.4}, 'report': report}))
    out = tmp_path / 'ranking.md'
    main(['--results_dir', str(tmp_path), '--out', str(out), '--n_boot', '100'])
    text = out.read_text()
    assert 'Success@5' in text and 'MRR' in text and '95% CI' in text and 'noisy_vgr - noisy' in text
