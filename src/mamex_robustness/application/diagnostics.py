"""Per-(user, test item) ranks among the cold candidates, for failure analysis.

metrics_from_ranks reproduces the Recall/NDCG of evaluation.py from these ranks, so any per-user or
per-item breakdown derived from them is consistent with the reported tables.
"""
import numpy as np


def positive_ranks(user_emb, item_emb, cold_index, test_data, train_data):
    """Return (users, items, ranks): one row per test pair; rank is 1-based among cold items, train items masked."""
    cold = np.asarray(cold_index)
    col_of = {int(i): c for c, i in enumerate(cold)}
    users, items = [], []
    for u, positives in test_data.items():
        for i in positives:
            users.append(int(u))
            items.append(int(i))
    users, items = np.asarray(users), np.asarray(items)
    unique_users, row = np.unique(users, return_inverse=True)
    scores = user_emb[unique_users] @ item_emb[cold].T
    for r, u in enumerate(unique_users):
        for i in train_data.get(int(u), ()):
            if int(i) in col_of:
                scores[r, col_of[int(i)]] = -np.inf
    cols = np.asarray([col_of[int(i)] for i in items])
    pos_score = scores[row, cols]
    ranks = (scores[row] > pos_score[:, None]).sum(axis=1) + 1
    return users, items, ranks


def metrics_from_ranks(users, ranks, k):
    """Per-user (user_ids, recall@k, ndcg@k) from the ranks of each user's test items."""
    unique_users, row = np.unique(users, return_inverse=True)
    n_pos = np.bincount(row)
    in_top = ranks <= k
    hits = np.bincount(row, weights=in_top.astype(float))
    dcg = np.bincount(row, weights=np.where(in_top, 1.0 / np.log2(ranks + 1.0), 0.0))
    idcg = np.asarray([(1.0 / np.log2(np.arange(1, min(k, n) + 1) + 1.0)).sum() for n in n_pos])
    return unique_users, hits / n_pos, dcg / idcg
