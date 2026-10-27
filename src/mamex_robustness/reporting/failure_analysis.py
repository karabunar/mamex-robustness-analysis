"""Who does VGR help and who does it hurt? Built from the per-pair ranks each run saves in results/diagnostics/."""
import argparse
import glob
import json
import os

import numpy as np

from mamex_robustness.application.diagnostics import metrics_from_ranks

COMPARISONS = [('noisy_vgr', 'noisy'), ('clean_vgr', 'clean'), ('noisy_vgr', 'noisy_vgr_shuffled')]
# Recall is discrete, so it must match exactly. NDCG is rank-sensitive and faiss and numpy can order
# near-tied float32 scores differently, which moves it by ~1e-6 on the real data.
_RECALL_TOL = 1e-6
_NDCG_TOL = 1e-5
_TOL = 1e-6  # threshold for calling a per-user difference non-zero


def _label(a, b):
    return f'{a} vs {b}'


def _check_against_report(users, ranks, report, k):
    """The ranks must reproduce the metrics that the run reported, otherwise the breakdown would not match the tables."""
    for name, r in ranks.items():
        _, recall, ndcg = metrics_from_ranks(users, r, k)
        expected = report[name]
        if (abs(recall.mean() - expected['recall'][str(k)]) > _RECALL_TOL
                or abs(ndcg.mean() - expected['ndcg'][str(k)]) > _NDCG_TOL):
            raise ValueError(f'per-pair ranks for {name!r} do not match the reported metrics')


def _user_effect(users, ranks_a, ranks_b, exposed, k):
    ids, rec_a, nd_a = metrics_from_ranks(users, ranks_a, k)
    _, rec_b, nd_b = metrics_from_ranks(users, ranks_b, k)
    d_rec, d_nd = rec_a - rec_b, nd_a - nd_b

    def summarize(mask):
        n = int(mask.sum())
        if n == 0:
            return {'n_users': 0}
        return {'n_users': n, 'helped': int((d_nd[mask] > _TOL).sum()), 'hurt': int((d_nd[mask] < -_TOL).sum()),
                'unchanged': int((np.abs(d_nd[mask]) <= _TOL).sum()),
                'mean_recall_diff': float(d_rec[mask].mean()), 'mean_ndcg_diff': float(d_nd[mask].mean())}

    everyone = np.ones(len(ids), dtype=bool)
    out = summarize(everyone)
    is_exposed = np.isin(ids, exposed)
    out['by_exposure'] = {'exposed': summarize(is_exposed), 'not_exposed': summarize(~is_exposed)}
    return out


def _item_effect(items, ranks_a, ranks_b, corrupted, top=5):
    """Rank gain = rank_b - rank_a: positive means A places the test item higher (better)."""
    gain = (ranks_b - ranks_a).astype(float)
    unique_items = np.unique(items)
    per_item = {int(i): float(gain[items == i].mean()) for i in unique_items}
    is_corrupted = np.isin(items, corrupted)

    def summarize(mask):
        return {'n_pairs': int(mask.sum()), 'mean_rank_gain': float(gain[mask].mean()) if mask.any() else None,
                'share_improved': float((gain[mask] > 0).mean()) if mask.any() else None,
                'share_worse': float((gain[mask] < 0).mean()) if mask.any() else None}

    ordered = sorted(per_item.items(), key=lambda kv: kv[1])
    return {'corrupted': summarize(is_corrupted), 'not_corrupted': summarize(~is_corrupted),
            'most_hurt': ordered[:top], 'most_helped': ordered[::-1][:top]}


def analyze_run(npz_path, json_path, k=20):
    with open(json_path) as f:
        run = json.load(f)
    d = np.load(npz_path)
    ranks = {name[len('ranks_'):]: d[name] for name in d.files if name.startswith('ranks_')}
    _check_against_report(d['users'], ranks, run['report'], k)
    exposed = np.unique(d['users'][np.isin(d['items'], d['corrupted_items'])])
    users, items = {}, {}
    for a, b in COMPARISONS:
        if a in ranks and b in ranks:
            users[_label(a, b)] = _user_effect(d['users'], ranks[a], ranks[b], exposed, k)
            items[_label(a, b)] = _item_effect(d['items'], ranks[a], ranks[b], d['corrupted_items'])
    return {'seed': run['args']['seed'], 'k': k, 'users': users, 'items': items}


def _find_runs(results_dir, noise_rate):
    pairs = []
    for npz in sorted(glob.glob(os.path.join(results_dir, 'diagnostics', '*.npz'))):
        js = os.path.join(results_dir, os.path.basename(npz)[:-4] + '.json')
        if os.path.exists(js):
            with open(js) as f:
                a = json.load(f)['args']
            if (a['mode'] == 'improved' and a.get('suffix', 'default') == 'default' and a.get('gamma_grid')
                    and abs(a['noise_rate'] - noise_rate) < 1e-9):
                pairs.append((npz, js))
    return pairs


def render(results, k):
    out = [f'# Failure analysis (cold-item test set, top-{k}; ranks are among the cold candidates)', '',
           'Only item ids are available (the processed data has no titles), so items are listed by id.',
           'helped / hurt = a user whose NDCG@k is higher / lower with the first-named condition; the mean differences are first minus second.', '']
    for title, key in (('Users', 'users'), ('Items', 'items')):
        out += [f'## {title}', '']
        for label in results[0][key]:
            out += [f'### {label}', '']
            if key == 'users':
                out += ['| seed | group | users | helped | hurt | unchanged | mean Δrecall | mean Δndcg |', '|---|---|---|---|---|---|---|---|']
                for r in results:
                    u = r['users'][label]
                    for name, s in (('all', u), ('exposed to corrupted item', u['by_exposure']['exposed']),
                                    ('not exposed', u['by_exposure']['not_exposed'])):
                        if s['n_users']:
                            out.append(f"| {r['seed']} | {name} | {s['n_users']} | {s['helped']} | {s['hurt']} | {s['unchanged']} | "
                                       f"{s['mean_recall_diff']:+.4f} | {s['mean_ndcg_diff']:+.4f} |")
            else:
                out += ['| seed | items | pairs | mean rank gain | share improved | share worse |', '|---|---|---|---|---|---|']
                for r in results:
                    for name in ('corrupted', 'not_corrupted'):
                        s = r['items'][label][name]
                        if s['n_pairs']:
                            out.append(f"| {r['seed']} | {name} | {s['n_pairs']} | {s['mean_rank_gain']:+.1f} | "
                                       f"{s['share_improved']:.2f} | {s['share_worse']:.2f} |")
                out += ['', 'Most hurt / most helped test items by id (mean rank gain; positive = first-named condition better):', '']
                for r in results:
                    s = r['items'][label]
                    out.append(f"- seed {r['seed']}: most hurt {[(i, round(g, 1)) for i, g in s['most_hurt']]}; "
                               f"most helped {[(i, round(g, 1)) for i, g in s['most_helped']]}")
            out.append('')
    return '\n'.join(out) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description='Failure analysis of VGR from saved per-pair ranks')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out', default='results/failure_analysis.md')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--noise_rate', type=float, default=0.4)
    a = p.parse_args(argv)
    runs = _find_runs(a.results_dir, a.noise_rate)
    if not runs:
        raise SystemExit(f'no diagnostics found in {a.results_dir}/diagnostics')
    results = [analyze_run(npz, js, a.k) for npz, js in runs]
    with open(a.out, 'w') as f:
        f.write(render(results, a.k))
    return results


if __name__ == '__main__':
    main()
