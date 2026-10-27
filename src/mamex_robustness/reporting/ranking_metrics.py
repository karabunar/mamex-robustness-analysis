"""Success@K, MRR and bootstrap confidence intervals, built from the per-pair ranks each run saves in results/diagnostics/.

Users are resampled with replacement (the same users in every seed and every condition of a draw), so the
intervals reflect which users happened to be in the test set; they do not include training-seed variance,
which the per-seed spread in the main tables covers.
"""
import argparse
import json

import numpy as np

from mamex_robustness.application.diagnostics import metrics_from_ranks
from mamex_robustness.reporting.failure_analysis import _check_against_report, _find_runs

CONDITIONS = ['clean', 'noisy', 'noisy_vgr', 'noisy_vgr_shuffled', 'noisy_vgr_constant', 'clean_vgr']
DIFFS = [('noisy', 'clean'), ('noisy_vgr', 'noisy'), ('noisy_vgr', 'noisy_vgr_shuffled'), ('clean_vgr', 'clean')]
SUCCESS_KS = (1, 5, 10, 20, 50)


def user_metrics(users, ranks, ks=SUCCESS_KS, k_main=20):
    """Per-user arrays (sorted by user id): success@k = any test item in the top k (not the repository's HR, which divides hits by min(k, positives)); mrr = 1 / best rank; recall and ndcg at k_main."""
    unique_users, row = np.unique(users, return_inverse=True)
    best = np.full(len(unique_users), np.inf)
    np.minimum.at(best, row, ranks)
    out = {f'success@{k}': (best <= k).astype(float) for k in ks}
    out['mrr'] = 1.0 / best
    _, rec, nd = metrics_from_ranks(users, ranks, k_main)
    out[f'recall@{k_main}'], out[f'ndcg@{k_main}'] = rec, nd
    return out


def bootstrap_means(values, n_boot=1000, seed=0, diffs=(), alpha=0.05):
    """values: name -> array (seeds, users). Mean over seeds and users with a percentile interval over user resamples.

    Each draw resamples users once and applies it to every array, so a difference between two arrays is paired."""
    rng = np.random.default_rng(seed)
    n_users = next(iter(values.values())).shape[1]
    keys = list(values) + [f'{a} - {b}' for a, b in diffs]
    draws = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n_users, n_users)
        res = {k: values[k][:, idx].mean() for k in values}
        for k in values:
            draws[k][i] = res[k]
        for a, b in diffs:
            draws[f'{a} - {b}'][i] = res[a] - res[b]
    point = {k: float(values[k].mean()) for k in values}
    for a, b in diffs:
        point[f'{a} - {b}'] = point[a] - point[b]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {'mean': point[k], 'ci': (float(np.percentile(draws[k], lo)), float(np.percentile(draws[k], hi)))} for k in keys}


def collect(runs, k=20):
    """Per-condition, per-metric arrays (seeds, users) from the saved ranks; rejects ranks that do not match the reports."""
    per_seed, seeds = [], []
    for npz, js in runs:
        with open(js) as f:
            run = json.load(f)
        d = np.load(npz)
        ranks = {name[len('ranks_'):]: d[name] for name in d.files if name.startswith('ranks_')}
        _check_against_report(d['users'], ranks, run['report'], k)
        per_seed.append({c: user_metrics(d['users'], ranks[c], k_main=k) for c in CONDITIONS if c in ranks})
        seeds.append(run['args']['seed'])
    return seeds, per_seed


def analyze(runs, k=20, n_boot=1000, seed=0):
    seeds, per_seed = collect(runs, k)
    conditions = [c for c in CONDITIONS if all(c in s for s in per_seed)]
    metrics = list(per_seed[0][conditions[0]])
    out = {'seeds': seeds, 'conditions': conditions, 'metrics': metrics, 'k': k, 'n_boot': n_boot, 'stats': {}}
    diffs = [(a, b) for a, b in DIFFS if a in conditions and b in conditions]
    out['diffs'] = diffs
    for m in metrics:
        values = {c: np.stack([s[c][m] for s in per_seed]) for c in conditions}
        out['stats'][m] = bootstrap_means(values, n_boot=n_boot, seed=seed, diffs=diffs)
    return out


def _fmt(s, digits=4):
    return f"{s['mean']:.{digits}f} [{s['ci'][0]:.{digits}f}, {s['ci'][1]:.{digits}f}]"


def render(res):
    k = res['k']
    names = {'mrr': 'MRR', f'recall@{k}': f'Recall@{k}', f'ndcg@{k}': f'NDCG@{k}'}
    cols = [f'success@{x}' for x in SUCCESS_KS] + ['mrr', f'recall@{k}', f'ndcg@{k}']
    head = lambda m: names.get(m, m.capitalize())
    out = [f'# Ranking metrics with bootstrap intervals (cold-item test set, mean over seeds {res["seeds"]})', '',
           f'Each cell is the mean with a 95% CI ({res["n_boot"]} resamples of the test users; the same users in every seed and condition of a draw). '
           'Success@K = share of users with at least one test item in the top K; MRR = mean of 1 / best rank. '
           'Ranks are among the cold candidates. The intervals reflect test-user sampling only, not training-seed variance.', '',
           '## Conditions', '', '| condition | ' + ' | '.join(head(m) for m in cols) + ' |', '|---|' + '---|' * len(cols)]
    for c in res['conditions']:
        out.append(f'| {c} | ' + ' | '.join(_fmt(res['stats'][m][c]) for m in cols) + ' |')
    out += ['', '## Paired differences (first minus second, same resampled users)', '',
            '| difference | ' + ' | '.join(head(m) for m in cols) + ' |', '|---|' + '---|' * len(cols)]
    for a, b in res['diffs']:
        key = f'{a} - {b}'
        cells = []
        for m in cols:
            s = res['stats'][m][key]
            mark = '*' if s['ci'][0] > 0 or s['ci'][1] < 0 else ''
            cells.append(_fmt(s) + mark)
        out.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    out += ['', '`*` = the interval excludes 0.', '']
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description='Success@K, MRR and bootstrap CIs from saved per-pair ranks')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out', default='results/ranking_metrics.md')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--noise_rate', type=float, default=0.4)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    runs = _find_runs(a.results_dir, a.noise_rate)
    if not runs:
        raise SystemExit(f'no diagnostics found in {a.results_dir}/diagnostics')
    res = analyze(runs, a.k, a.n_boot, a.seed)
    with open(a.out, 'w') as f:
        f.write(render(res))
    return res


if __name__ == '__main__':
    main()
