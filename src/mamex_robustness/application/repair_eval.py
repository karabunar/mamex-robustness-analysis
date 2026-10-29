import time

import numpy as np
import torch

from mamex_robustness.domain import repair

LAM_GRID = (0.25, 0.5, 0.75, 1.0)
GATED_GRID = tuple((lam_max, g) for lam_max in (0.5, 1.0) for g in (5.0, 10.0, 20.0))
COSINE_GRID = tuple((lam_max, q, g) for lam_max in (0.75, 1.0) for q in (0.3, 0.4, 0.5) for g in (20.0, 50.0))
GAMMA_GRID = (1.0, 2.0, 4.0, 8.0, 16.0)
CONTROL_SEED = 0


def _scores(session, text, mode, use_vgr=False, gamma=None, ks=(10, 20)):
    """Recall and NDCG at the report cutoff and NDCG at the selection cutoff, on cold items, for one text matrix.

    text None means the model's own clean text; ks is (selection cutoff, report cutoff).
    """
    previous = session.model.vgr_gamma
    if gamma is not None:
        session.model.vgr_gamma = gamma
    try:
        _, recall, ndcg, _ = session.test(mode=mode, top_list=list(ks), noisy_text=text, use_vgr=use_vgr)
    finally:
        session.model.vgr_gamma = previous
    return {'recall': float(recall[ks[1]]), 'ndcg': float(ndcg[ks[1]]), 'ndcg_select': float(ndcg[ks[0]])}


@torch.no_grad()
def _vgr_cosine(session, text):
    session.model.eval()
    session.model.set_eval_text(text)
    try:
        session.model()
        return session.model.last_cos.cpu().numpy()
    finally:
        session.model.set_eval_text(None)


def _select(session, candidates, ks):
    """candidates: {param: validation text}; the best validation score wins, ties go to the earlier candidate."""
    best, best_score = None, -1.0
    for param, text in candidates.items():
        score = _scores(session, text, 'val', ks=ks)['ndcg_select']
        if score > best_score:
            best, best_score = param, score
    return best, best_score


def evaluate_repairs(session, noise_rate, noise_type, noise_seed, k=10, ks=(10, 20)):
    """Compare the unrepaired model, VGR and two image-guided text repairs, all on the model's current (best) weights.

    Parameters are chosen on noisy validation data only (NDCG at ks[0]); the test set is read once per reported condition
    and reports Recall and NDCG at ks[1].
    """
    data = session.dataset
    cold = np.asarray(data.cold_item_index)
    noisy_test = data.noisy_text_feat(noise_rate, noise_type, noise_seed)
    noisy_val = data.noisy_text_feat(noise_rate, noise_type, session.VAL_NOISE_SEED_OFFSET + noise_seed)
    clean = data.text_feat
    corrupted = (noisy_test[cold] != clean[cold]).any(axis=1)

    started = time.perf_counter()
    knn = repair.neighbor_text(data.image_feat, clean, data.warm_item_index, cold, k)
    knn_seconds = time.perf_counter() - started
    rand = repair.neighbor_text(data.image_feat, clean, data.warm_item_index, cold, k, random_seed=CONTROL_SEED)
    permutation = np.random.default_rng(CONTROL_SEED).permutation(len(cold))

    def build(method, text, param):
        """Cold rows of `text` repaired by `method` with its parameter; returns the full text matrix."""
        rows = text[cold]
        if method == 'm1':
            new = repair.blend(rows, knn, param)
        elif method == 'm1_random':
            new = repair.blend(rows, rand, param)
        elif method == 'm2':
            new = repair.blend(rows, knn, repair.gated_lambda(repair.agreement(rows, knn), *param))
        elif method == 'm2_random':  # the same rule with random warm items in place of the image neighbours
            new = repair.blend(rows, rand, repair.gated_lambda(repair.agreement(rows, rand), *param))
        elif method in ('m3', 'm3_random', 'm3_shuffled'):
            score = _vgr_cosine(session, text)[cold]
            if method == 'm3_shuffled':
                score = score[permutation]
            lam = repair.quantile_lambda(score, param[0], param[2], param[1])
            new = repair.blend(rows, rand if method == 'm3_random' else knn, lam)
        elif method == 'oracle':  # upper bound: the true corruption mask, not available to a real method
            is_corrupted = (rows != clean[cold]).any(axis=1)
            new = repair.blend(rows, knn, np.where(is_corrupted, param, 0.0))
        else:  # m2_shuffled: the agreement scores are handed to the wrong items
            lam = repair.gated_lambda(repair.agreement(rows, knn), *param)[permutation]
            new = repair.blend(rows, knn, lam)
        return repair.with_cold_rows(text, cold, new)

    grids = {'m1': LAM_GRID, 'm1_random': LAM_GRID, 'm2': GATED_GRID, 'm2_random': GATED_GRID, 'm2_shuffled': GATED_GRID,
             'm3': COSINE_GRID, 'm3_random': COSINE_GRID, 'm3_shuffled': COSINE_GRID, 'oracle': LAM_GRID}
    out = {'selection': {}, 'noisy': {}, 'clean': {}, 'noisy_plus_vgr': {}, 'clean_plus_vgr': {}}

    out['noisy']['unrepaired'] = _scores(session, noisy_test, 'test', ks=ks)
    out['clean']['unrepaired'] = _scores(session, None, 'test', ks=ks)
    gamma, val = _select_gamma(session, {g: noisy_val for g in GAMMA_GRID}, ks)
    out['selection']['vgr'] = {'gamma': gamma, 'val_ndcg': val}
    out['noisy']['vgr'] = _scores(session, noisy_test, 'test', True, gamma, ks)
    out['clean']['vgr'] = _scores(session, None, 'test', True, gamma, ks)

    for method, grid in grids.items():
        param, val = _select(session, {p: build(method, noisy_val, p) for p in grid}, ks)
        out['selection'][method] = {'param': list(param) if isinstance(param, tuple) else param, 'val_ndcg': val}
        out['noisy'][method] = _scores(session, build(method, noisy_test, param), 'test', ks=ks)
        out['clean'][method] = _scores(session, build(method, clean, param), 'test', ks=ks)
        if method in ('m1', 'm2', 'm3'):
            g, gval = _select_gamma(session, {g: build(method, noisy_val, param) for g in GAMMA_GRID}, ks)
            out['selection'][method + '_plus_vgr'] = {'gamma': g, 'val_ndcg': gval}
            out['noisy_plus_vgr'][method] = _scores(session, build(method, noisy_test, param), 'test', True, g, ks)
            out['clean_plus_vgr'][method] = _scores(session, build(method, clean, param), 'test', True, g, ks)

    started = time.perf_counter()
    build('m2', noisy_test, GATED_GRID[0])
    build_seconds = time.perf_counter() - started
    out['auroc'] = {'agreement': repair.auroc(-repair.agreement(noisy_test[cold], knn), corrupted),
                    'vgr_cosine': repair.auroc(-_vgr_cosine(session, noisy_test)[cold], corrupted)}
    started = time.perf_counter()
    cosine = _vgr_cosine(session, noisy_test)[cold]
    cosine_seconds = time.perf_counter() - started
    repaired = build('m1', noisy_test, 0.5)
    forward_plain = session._forward_ms(False)
    session.model.set_eval_text(repaired)
    try:
        forward_repaired = session._forward_ms(False)
    finally:
        session.model.set_eval_text(None)
    out['cost'] = {'neighbor_seconds': knn_seconds, 'repair_seconds': build_seconds,
                   'extra_mb': (len(cold) * len(data.warm_item_index) * 4 + knn.nbytes + rand.nbytes) / 2 ** 20,
                   'forward_ms': forward_plain, 'forward_ms_repaired': forward_repaired,
                   'cosine_seconds': cosine_seconds, 'cosine_mb': cosine.nbytes / 2 ** 20}
    out['n_corrupted_cold'] = int(corrupted.sum())
    out['k'] = k
    out['ks'] = list(ks)
    return out


def _select_gamma(session, candidates, ks):
    """VGR gamma with the best noisy-validation NDCG@10 (ties go to the smaller gamma)."""
    best, best_score = None, -1.0
    for gamma, text in sorted(candidates.items()):
        score = _scores(session, text, 'val', True, gamma, ks)['ndcg_select']
        if score > best_score:
            best, best_score = gamma, score
    return best, best_score
