"""Training-free repair of suspect text features from the text of visually similar warm items."""
import numpy as np
from scipy.stats import rankdata

_EPS = 1e-8


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + _EPS)


def neighbor_text(image_feat, text_feat, warm_index, query_index, k=10, random_seed=None):
    """Unit mean of the unit text features of each query item's k nearest warm items by image cosine.

    With random_seed, k warm items drawn at random per query item replace the image neighbours (control).
    """
    warm, query = np.asarray(warm_index), np.asarray(query_index)
    if random_seed is None:
        sims = _unit(image_feat[query].astype(np.float32)) @ _unit(image_feat[warm].astype(np.float32)).T
        chosen = warm[np.argpartition(-sims, k - 1, axis=1)[:, :k]]
    else:
        rng = np.random.default_rng(random_seed)
        chosen = np.stack([rng.choice(warm, size=k, replace=False) for _ in query])
    return _unit(_unit(text_feat[chosen]).mean(axis=1))


def agreement(text, neighbor):
    """Row-wise cosine between an item's text and the text built from its image neighbours."""
    return (_unit(text) * _unit(neighbor)).sum(axis=1)


def gated_lambda(agree, lam_max, steepness):
    """Per-item repair weight: lam_max for items far below the median agreement, about 0 for items far above it."""
    return lam_max / (1.0 + np.exp(steepness * (agree - np.median(agree))))


def quantile_lambda(score, lam_max, steepness, q):
    """Repair weight that is high for items whose score lies below the q-quantile of the scores and about 0 above it."""
    return lam_max / (1.0 + np.exp(-steepness * (np.quantile(score, q) - score)))


def blend(text, neighbor, lam):
    """Unit rows of (1 - lam) * unit(text) + lam * neighbor; lam is a scalar or one weight per row."""
    lam = np.asarray(lam, dtype=np.float64)
    lam = lam[:, None] if lam.ndim else lam
    return _unit((1.0 - lam) * _unit(text) + lam * neighbor)


def with_cold_rows(text_feat, cold_index, rows):
    """Copy of text_feat in which the cold items' rows are replaced."""
    out = text_feat.copy()
    out[np.asarray(cold_index)] = rows
    return out


def auroc(scores, positive):
    """Probability that a random positive item scores higher than a random negative one (ties count half)."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
