"""Build the report tables from results/*.json (and nothing else). Missing groups are left out, never filled in."""
import argparse
import glob
import json
import os

import numpy as np

# MAMEX paper, Amazon Baby (published numbers; kept apart from measured ones).
PUBLISHED = {'Rec@10': 0.0512, 'Rec@20': 0.0771, 'NDCG@10': 0.0292, 'NDCG@20': 0.0363}
METRICS = {'recall@10': ('recall', '10'), 'recall@20': ('recall', '20'),
           'ndcg@10': ('ndcg', '10'), 'ndcg@20': ('ndcg', '20')}
PAIRS = [('noisy_vgr', 'noisy'), ('noisy_vgr', 'noisy_vgr_shuffled'), ('noisy_vgr', 'noisy_vgr_constant'),
         ('clean_vgr', 'clean')]


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _load_all(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _load(results_dir):
    return [r for r in _load_all(results_dir) if r['args'].get('suffix', 'default') == 'default']


def _rows(runs, rate, noise_type):
    def pick(mode, any_rate=False):
        return sorted((r for r in runs if r['args']['mode'] == mode and r['args']['noise_type'] == noise_type
                       and (any_rate or abs(r['args']['noise_rate'] - rate) < 1e-9)
                       and (mode != 'improved' or 'gamma_selection' in r['report'])
                       and not r['args'].get('train_with_vgr')), key=lambda r: r['args']['seed'])
    return {'baseline': pick('baseline', any_rate=True), 'stress_test': pick('stress_test'), 'improved': pick('improved')}


def build_tables(results_dir, noise_rate=0.4, noise_type='hashtag'):
    g = _rows(_load(results_dir), noise_rate, noise_type)
    spec = [('Clean baseline', 'baseline', 'clean'), ('Stress test (noisy text)', 'stress_test', 'noisy'),
            ('MAMEX-VGR (noisy text)', 'improved', 'noisy_vgr'), ('MAMEX-VGR (clean text)', 'improved', 'clean_vgr'),
            ('VGR control: shuffled cosine (noisy text)', 'improved', 'noisy_vgr_shuffled'),
            ('VGR control: constant factor (noisy text)', 'improved', 'noisy_vgr_constant')]
    rows = []
    for label, group, block in spec:
        runs = g[group]
        if runs:
            row = {'label': label, 'n': len(runs)}
            for name, (kind, k) in METRICS.items():
                row[name] = _mean_std([r['report'][block][kind][k] for r in runs])
            rows.append(row)

    paired = {}
    for a, b in PAIRS:
        by_seed = {r['args']['seed']: r['report'] for r in g['improved']}
        if not by_seed:
            continue
        entry = {}
        for name, (kind, k) in METRICS.items():
            diffs = [rep[a][kind][k] - rep[b][kind][k] for _, rep in sorted(by_seed.items())]
            entry[name] = {'per_seed': diffs, 'mean': float(np.mean(diffs)), 'positive': sum(d > 0 for d in diffs),
                           'n': len(diffs)}
        paired[f'{a} - {b}'] = entry

    cost, gamma, gate = {}, {}, {}
    imp = g['improved']
    if imp:
        for key in ('forward_ms_plain', 'forward_ms_vgr', 'gpu_mem_mb'):
            cost[key] = _mean_std([r['report']['cost'][key] for r in imp])
        cost['train_seconds'] = _mean_std([r['train_seconds'] for r in imp])
        gamma = {'selected': [r['report']['gamma_selection']['selected'] for r in imp]}
        for block in ('clean', 'noisy', 'noisy_vgr'):
            gate[block] = {w: _mean_std([r['report']['gate'][block][w] for r in imp]) for w in ('w_v', 'w_t')}
    return {'accuracy': {'rows': rows}, 'paired': paired, 'cost': cost, 'gamma': gamma, 'gate': gate}


def build_ablation(results_dir, noise_rate=0.4, noise_type='hashtag'):
    """Noise-augmented training (suffix noise_aug) against the clean-trained runs of the same seeds; None if none exist."""
    aug = {r['args']['seed']: r['report'] for r in _load_all(results_dir)
           if r['args'].get('suffix') == 'noise_aug' and r['args']['noise_type'] == noise_type
           and abs(r['args']['noise_rate'] - noise_rate) < 1e-9 and r['args'].get('train_noise_rate', 0) > 0}
    if not aug:
        return None
    g = _rows(_load(results_dir), noise_rate, noise_type)
    ref = {name: {r['args']['seed']: r['report'] for r in g[name]} for name in g}
    seeds = sorted(s for s in aug if all(s in ref[n] for n in ref))
    if not seeds:
        return None
    gain = lambda rep, kind, k: rep['noisy_vgr'][kind][k] - rep['noisy'][kind][k]
    comparisons = {
        'noisy: noise-augmented - stress test':
            lambda s, kind, k: aug[s]['noisy'][kind][k] - ref['stress_test'][s]['noisy'][kind][k],
        'clean: noise-augmented - clean baseline':
            lambda s, kind, k: aug[s]['clean'][kind][k] - ref['baseline'][s]['clean'][kind][k],
        'noisy: clean baseline - noise-augmented (loss left)':
            lambda s, kind, k: ref['baseline'][s]['clean'][kind][k] - aug[s]['noisy'][kind][k],
        'VGR gain on noisy text: noise-augmented - clean-trained':
            lambda s, kind, k: gain(aug[s], kind, k) - gain(ref['improved'][s], kind, k)}
    paired = {}
    for name, fn in comparisons.items():
        paired[name] = {}
        for metric, (kind, k) in METRICS.items():
            d = [fn(s, kind, k) for s in seeds]
            paired[name][metric] = {'per_seed': d, 'mean': float(np.mean(d)), 'positive': sum(x > 0 for x in d), 'n': len(d)}
    blocks = [('Noise-augmented model, clean text', 'clean'), ('Noise-augmented model, noisy text', 'noisy'),
              ('Noise-augmented model, noisy text + VGR', 'noisy_vgr'), ('Noise-augmented model, clean text + VGR', 'clean_vgr')]
    rows = [{'label': label, **{m: _mean_std([aug[s][block][kind][k] for s in seeds]) for m, (kind, k) in METRICS.items()}}
            for label, block in blocks]
    gamma = [aug[s]['gamma_selection']['selected'] for s in seeds if 'gamma_selection' in aug[s]]
    gate = {b: {w: _mean_std([aug[s]['gate'][b][w] for s in seeds]) for w in ('w_v', 'w_t')}
            for b in ('clean', 'noisy', 'noisy_vgr')}
    return {'n': len(seeds), 'seeds': seeds, 'rows': rows, 'paired': paired, 'gamma': gamma, 'gate': gate}


def render_ablation(ab, noise_rate, noise_type):
    out = [f'## Noise-augmented training ({noise_type} noise on warm items\' text in every training epoch, rate {noise_rate}; '
           f'cold-item test; seeds {ab["seeds"]}; mean ± sample std)', '',
           '| Condition | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |', '|---|---|---|---|---|']
    out += [f"| {r['label']} | " + ' | '.join(_pm(r[m]) for m in METRICS) + ' |' for r in ab['rows']]
    out += ['', '### Paired differences per seed', '',
            '| Comparison | metric | per-seed differences | mean | seeds > 0 |', '|---|---|---|---|---|']
    for name, entry in ab['paired'].items():
        for metric in ('recall@20', 'ndcg@20'):
            d = entry[metric]
            out.append(f"| {name} | {metric} | {', '.join(f'{x:+.4f}' for x in d['per_seed'])} | {d['mean']:+.4f} | {d['positive']}/{d['n']} |")
    out += ['', f"Selected gamma per seed: {ab['gamma']}", '', '| Gate (corrupted cold items) | w_v | w_t |', '|---|---|---|']
    out += [f'| {b} | {_pm(v["w_v"])} | {_pm(v["w_t"])} |' for b, v in ab['gate'].items()]
    return out


def build_sensitivity(results_dir, seed, noise_rate=0.4, noise_type='hashtag'):
    """Single-seed noise-rate and gamma sweeps. Only runs that exist are listed."""
    runs = [r for r in _load(results_dir) if r['args']['seed'] == seed and r['args']['noise_type'] == noise_type
            and r['args']['mode'] == 'improved' and not r['args'].get('train_with_vgr')]
    selected = sorted((r for r in runs if 'gamma_selection' in r['report']), key=lambda r: r['args']['noise_rate'])
    manual = sorted((r for r in runs if 'gamma_selection' not in r['report'] and not r['args'].get('gamma_grid')
                     and abs(r['args']['noise_rate'] - noise_rate) < 1e-9), key=lambda r: r['args']['vgr_gamma'])
    at = lambda r, b: r['report'][b]['recall']['20']
    return {'rates': [{'rate': r['args']['noise_rate'], 'clean': at(r, 'clean'), 'noisy': at(r, 'noisy'),
                       'noisy_vgr': at(r, 'noisy_vgr'), 'selected': r['report']['gamma_selection']['selected']}
                      for r in selected],
            'gammas': [{'gamma': r['args']['vgr_gamma'], 'noisy': at(r, 'noisy'), 'noisy_vgr': at(r, 'noisy_vgr'),
                        'clean': at(r, 'clean'), 'clean_vgr': at(r, 'clean_vgr'),
                        'shuffled': at(r, 'noisy_vgr_shuffled'), 'constant': at(r, 'noisy_vgr_constant')} for r in manual]}


def render_sensitivity(s, seed):
    out = [f'## Sensitivity (single seed {seed}; Recall@20; not an average)', '']
    if s['rates']:
        out += ['| noise rate | clean | noisy | noisy + VGR | gamma selected on validation |', '|---|---|---|---|---|']
        out += [f"| {r['rate']} | {r['clean']:.4f} | {r['noisy']:.4f} | {r['noisy_vgr']:.4f} | {r['selected']} |"
                for r in s['rates']]
    else:
        out.append('pending')
    out += ['', '| gamma (set by hand, test sensitivity only) | noisy | noisy + VGR | shuffled cosine | constant factor '
            '| clean | clean + VGR |', '|---|---|---|---|---|---|---|']
    out += [f"| {g['gamma']} | {g['noisy']:.4f} | {g['noisy_vgr']:.4f} | {g['shuffled']:.4f} | {g['constant']:.4f} | "
            f"{g['clean']:.4f} | {g['clean_vgr']:.4f} |" for g in s['gammas']] or ['| pending | | | | | | |']
    return out


def _pm(v):
    return f'{v[0]:.3f} ± {v[1]:.3f}'


def render(tables, noise_rate, noise_type):
    out = ['## Published (MAMEX paper, Amazon Baby; not measured here)', '',
           '| ' + ' | '.join(PUBLISHED) + ' |', '|' + '---|' * len(PUBLISHED),
           '| ' + ' | '.join(f'{v:.4f}' for v in PUBLISHED.values()) + ' |', '',
           f'## Measured (cold items, test set, noise={noise_type} rate {noise_rate}; mean ± sample std over seeds)', '']
    rows = tables['accuracy']['rows']
    if rows:
        out += ['| Condition | n | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |', '|---|---|---|---|---|---|']
        out += [f"| {r['label']} | {r['n']} | " + ' | '.join(_pm(r[m]) for m in METRICS) + ' |' for r in rows]
    else:
        out.append('pending')
    out += ['', '## Paired differences per seed (A - B)', '']
    if tables['paired']:
        out += ['| A - B | metric | per-seed differences | mean | seeds with A > B |', '|---|---|---|---|---|']
        for pair, entry in tables['paired'].items():
            for metric in ('recall@20', 'ndcg@20'):
                d = entry[metric]
                out.append(f"| {pair} | {metric} | {', '.join(f'{x:+.4f}' for x in d['per_seed'])} | "
                           f"{d['mean']:+.4f} | {d['positive']}/{d['n']} |")
    else:
        out.append('pending')
    out += ['', '## Gate weights (improved runs, corrupted cold items)', '']
    if tables['gate']:
        out += ['| Setting | w_v | w_t |', '|---|---|---|']
        out += [f'| {b} | {_pm(v["w_v"])} | {_pm(v["w_t"])} |' for b, v in tables['gate'].items()]
    else:
        out.append('pending')
    out += ['', '## Cost and selected gamma', '']
    if tables['cost']:
        c = tables['cost']
        out += ['| forward ms (plain) | forward ms (VGR) | GPU memory MB | train seconds | selected gamma per seed |',
                '|---|---|---|---|---|',
                f"| {c['forward_ms_plain'][0]:.2f} | {c['forward_ms_vgr'][0]:.2f} | {c['gpu_mem_mb'][0]:.0f} | "
                f"{c['train_seconds'][0]:.0f} | {tables['gamma']['selected']} |"]
    else:
        out.append('pending')
    return '\n'.join(out) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description='Build report tables from result JSON files')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out', default='results/tables.md')
    p.add_argument('--noise_rate', type=float, default=0.4)
    p.add_argument('--noise_type', default='hashtag')
    p.add_argument('--sweep_seed', type=int, default=2025)
    a = p.parse_args(argv)
    tables = build_tables(a.results_dir, a.noise_rate, a.noise_type)
    text = render(tables, a.noise_rate, a.noise_type)
    sweep = build_sensitivity(a.results_dir, a.sweep_seed, a.noise_rate, a.noise_type)
    text += '\n' + '\n'.join(render_sensitivity(sweep, a.sweep_seed)) + '\n'
    ablation = build_ablation(a.results_dir, a.noise_rate, a.noise_type)
    if ablation:
        text += '\n' + '\n'.join(render_ablation(ablation, a.noise_rate, a.noise_type)) + '\n'
    with open(a.out, 'w') as f:
        f.write(text)
    return tables


if __name__ == '__main__':
    main()
