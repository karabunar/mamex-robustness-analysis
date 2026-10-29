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


def build_gate_probe(results_dir, noise_rate=0.4, noise_type='hashtag'):
    """Gate weights on warm items and same-environment metrics for the augmented model against its clean-trained control.

    Both come from runs made together (suffix gateprobe_aug / gateprobe_clean); None unless both kinds exist for some seed.
    """
    by_kind = {'aug': {}, 'clean': {}}
    for r in _load_all(results_dir):
        suffix = r['args'].get('suffix', '')
        if suffix in ('gateprobe_aug', 'gateprobe_clean') and r['args']['noise_type'] == noise_type \
                and abs(r['args']['noise_rate'] - noise_rate) < 1e-9 and 'gate_probe' in r['report']:
            by_kind[suffix.split('_')[1]][r['args']['seed']] = r['report']
    seeds = sorted(set(by_kind['aug']) & set(by_kind['clean']))
    if not seeds:
        return None
    names = ('warm_corrupted', 'warm_uncorrupted', 'warm_clean', 'cold_clean')
    gate = {k: {n: _mean_std([by_kind[k][s]['gate_probe'][n]['w_t'] for s in seeds]) for n in names} for k in by_kind}
    gap = {k: [by_kind[k][s]['gate_probe']['warm_uncorrupted']['w_t'] - by_kind[k][s]['gate_probe']['warm_corrupted']['w_t']
               for s in seeds] for k in by_kind}
    paired = {}
    for label, block in (('clean text', 'clean'), ('noisy text', 'noisy')):
        paired[f'{label}: noise-augmented - clean-trained'] = {}
        for metric, (kind, k) in METRICS.items():
            d = [by_kind['aug'][s][block][kind][k] - by_kind['clean'][s][block][kind][k] for s in seeds]
            paired[f'{label}: noise-augmented - clean-trained'][metric] = {
                'per_seed': d, 'mean': float(np.mean(d)), 'positive': sum(x > 0 for x in d), 'n': len(d)}
    return {'seeds': seeds, 'gate': gate, 'gap': gap, 'paired': paired}


def render_gate_probe(gp):
    out = [f'## Gate probe: mean text weight w_t on warm items (seeds {gp["seeds"]}; mean ± sample std)', '',
           '| Model | warm, corrupted | warm, not corrupted | warm, no noise | cold, no noise | gap (not corrupted - corrupted) per seed |',
           '|---|---|---|---|---|---|']
    for k, label in (('aug', 'Noise-augmented training'), ('clean', 'Clean training (control)')):
        g = gp['gate'][k]
        out.append(f"| {label} | " + ' | '.join(_pm(g[n]) for n in ('warm_corrupted', 'warm_uncorrupted', 'warm_clean', 'cold_clean'))
                   + f" | {', '.join(f'{x:+.3f}' for x in gp['gap'][k])} |")
    out += ['', 'Same-environment comparison (both models from the gate-probe runs; paired per seed):', '',
            '| Comparison | metric | per-seed differences | mean | seeds > 0 |', '|---|---|---|---|---|']
    for name, entry in gp['paired'].items():
        for metric in ('recall@20', 'ndcg@20'):
            d = entry[metric]
            out.append(f"| {name} | {metric} | {', '.join(f'{x:+.4f}' for x in d['per_seed'])} | {d['mean']:+.4f} | {d['positive']}/{d['n']} |")
    return out


def build_repr_probe(results_dir, noise_rate=0.4, noise_type='hashtag'):
    """Where cold items sit relative to warm items in the space the gate reads (suffix repr_aug / repr_clean); None if no such runs."""
    by_kind = {'aug': {}, 'clean': {}}
    for r in _load_all(results_dir):
        suffix = r['args'].get('suffix', '')
        if suffix in ('repr_aug', 'repr_clean') and r['args']['noise_type'] == noise_type \
                and abs(r['args']['noise_rate'] - noise_rate) < 1e-9 and 'repr_probe' in r['report']:
            by_kind[suffix.split('_')[1]][r['args']['seed']] = r['report']['repr_probe']
    seeds = sorted(set(by_kind['aug']) & set(by_kind['clean']))
    if not seeds:
        return None
    group = lambda k, g, name: _mean_std([by_kind[k][s][g][name] for s in seeds])
    return {'seeds': seeds,
            'stats': {k: {g: {n: group(k, g, n) for n in ('norm_v', 'norm_t', 'distance', 'logit_gap', 'w_t')}
                          for g in ('warm', 'cold')} for k in by_kind},
            'ratio': {k: [by_kind[k][s]['distance_ratio'] for s in seeds] for k in by_kind},
            'spearman': {k: [by_kind[k][s]['spearman_distance_wt_cold'] for s in seeds] for k in by_kind}}


def render_repr_probe(rp):
    out = [f'## Representation probe: warm and cold items in the space the gate reads (clean text; seeds {rp["seeds"]}; mean ± sample std)', '',
           '| Model | group | norm (image) | norm (text) | distance to warm centroid | gate logit (text - image) | w_t |',
           '|---|---|---|---|---|---|---|']
    for k, label in (('aug', 'Noise-augmented training'), ('clean', 'Clean training (control)')):
        for g in ('warm', 'cold'):
            st = rp['stats'][k][g]
            out.append(f"| {label} | {g} | " + ' | '.join(_pm(st[n]) for n in ('norm_v', 'norm_t', 'distance', 'logit_gap', 'w_t')) + ' |')
    out += ['', '| Model | cold / warm distance ratio per seed | Spearman(distance, w_t) over cold items per seed |', '|---|---|---|']
    for k, label in (('aug', 'Noise-augmented training'), ('clean', 'Clean training (control)')):
        out.append(f"| {label} | {', '.join(f'{x:.3f}' for x in rp['ratio'][k])} | "
                   f"{', '.join('n/a' if x is None else f'{x:+.3f}' for x in rp['spearman'][k])} |")
    return out


REPAIR_METHODS = (('unrepaired', 'Unrepaired'), ('vgr', 'VGR'), ('m1', 'M1 blind blend'), ('m1_random', 'M1, random neighbours'),
                  ('m2', 'M2 gated repair'), ('m2_random', 'M2, random neighbours'), ('m2_shuffled', 'M2, permuted agreement'),
                  ('m3', 'M3 cosine-gated repair'), ('m3_random', 'M3, random neighbours'), ('m3_shuffled', 'M3, permuted cosine'),
                  ('oracle', 'Oracle (true mask; upper bound only)'))


def build_repair(results_dir, noise_rate=0.4, noise_type='hashtag'):
    """Training-free text repairs on the saved clean-trained checkpoints (results/repair/*.json); None if there are none."""
    runs = {}
    for r in _load_all(os.path.join(results_dir, 'repair')):
        if r['args']['noise_type'] == noise_type and abs(r['args']['noise_rate'] - noise_rate) < 1e-9 \
                and r.get('validation', {'matched': True})['matched']:
            runs[r['args']['seed']] = r['report']
    if not runs:
        return None
    seeds = sorted(runs)
    series = lambda cond, m: [runs[s][cond][m]['recall'] for s in seeds]
    rows = {}
    for cond in ('noisy', 'clean'):
        rows[cond] = {m: _mean_std(series(cond, m)) for m, _ in REPAIR_METHODS}
    for cond in ('noisy_plus_vgr', 'clean_plus_vgr'):
        rows[cond] = {m: _mean_std(series(cond, m)) for m in ('m1', 'm2', 'm3')}
    base = {c: series(c, 'unrepaired') for c in ('noisy', 'clean')}
    paired = {}
    for cond, group, methods in (('noisy', 'noisy', [m for m, _ in REPAIR_METHODS[1:]]), ('clean', 'clean', [m for m, _ in REPAIR_METHODS[1:]]),
                                 ('noisy', 'noisy_plus_vgr', ['m1', 'm2', 'm3']), ('clean', 'clean_plus_vgr', ['m1', 'm2', 'm3'])):
        for m in methods:
            d = [a - b for a, b in zip(series(group, m), base[cond])]
            paired[f'{group}: {m} - unrepaired'] = {'per_seed': d, 'mean': float(np.mean(d)), 'positive': sum(x > 0 for x in d), 'n': len(d)}
    for m, c in (('m1', 'm1_random'), ('m2', 'm2_random'), ('m2', 'm2_shuffled'), ('m3', 'm3_random'), ('m3', 'm3_shuffled'),
                 ('m3', 'm1'), ('m3', 'oracle'), ('m3', 'vgr')):
        d = [a - b for a, b in zip(series('noisy', m), series('noisy', c))]
        paired[f'noisy: {m} - {c}'] = {'per_seed': d, 'mean': float(np.mean(d)), 'positive': sum(x > 0 for x in d), 'n': len(d)}
    return {'seeds': seeds, 'rows': rows, 'paired': paired,
            'selection': {s: runs[s]['selection'] for s in seeds},
            'auroc': {k: [runs[s]['auroc'][k] for s in seeds] for k in ('agreement', 'vgr_cosine')},
            'cost': {k: _mean_std([runs[s]['cost'][k] for s in seeds]) for k in runs[seeds[0]]['cost']},
            'n_corrupted': runs[seeds[0]]['n_corrupted_cold']}


def render_repair(rp):
    out = [f'## Text repair on saved clean-trained checkpoints (seeds {rp["seeds"]}; Recall@20; mean ± sample std)', '',
           '| Method | noisy text | clean text |', '|---|---|---|']
    for m, label in REPAIR_METHODS:
        out.append(f"| {label} | {_pm(rp['rows']['noisy'][m])} | {_pm(rp['rows']['clean'][m])} |")
    for m, label in (('m1', 'M1 blind blend, then VGR'), ('m2', 'M2 gated repair, then VGR'), ('m3', 'M3 cosine-gated repair, then VGR')):
        out.append(f"| {label} | {_pm(rp['rows']['noisy_plus_vgr'][m])} | {_pm(rp['rows']['clean_plus_vgr'][m])} |")
    out += ['', 'Paired differences per seed (Recall@20):', '', '| Comparison | per-seed differences | mean | seeds > 0 |', '|---|---|---|---|']
    for name, d in rp['paired'].items():
        out.append(f"| {name} | {', '.join(f'{x:+.4f}' for x in d['per_seed'])} | {d['mean']:+.4f} | {d['positive']}/{d['n']} |")
    out += ['', f"Detection of the {rp['n_corrupted']} corrupted cold items (AUROC per seed): agreement with the image neighbours' text "
                f"{', '.join(f'{x:.3f}' for x in rp['auroc']['agreement'])}; VGR cosine of the expert outputs "
                f"{', '.join(f'{x:.3f}' for x in rp['auroc']['vgr_cosine'])}.", '',
            '| Selected on noisy validation data | ' + ' | '.join(f'seed {s}' for s in rp['seeds']) + ' |', '|---|' + '---|' * len(rp['seeds'])]
    for key in rp['selection'][rp['seeds'][0]]:
        vals = [rp['selection'][s][key] for s in rp['seeds']]
        out.append(f'| {key} | ' + ' | '.join(str(v.get("param", v.get("gamma"))) for v in vals) + ' |')
    c = rp['cost']
    out += ['', f"Cost: nearest-neighbour search {c['neighbor_seconds'][0]:.2f} s once, building the repaired features {c['repair_seconds'][0]:.3f} s, "
                f"extra memory {c['extra_mb'][0]:.0f} MB (computed from the array sizes), forward pass {c['forward_ms'][0]:.2f} ms without "
                f"and {c['forward_ms_repaired'][0]:.2f} ms with repaired text; the cosine for M3 takes {c['cosine_seconds'][0]:.3f} s and {c['cosine_mb'][0]:.3f} MB."]
    return out


def build_repair_grid(results_dir):
    """Per noise setting: gains of the repairs over the unrepaired model, per seed, with matched or mismatched validation noise."""
    cells = {}
    for r in _load_all(os.path.join(results_dir, 'repair')):
        v = r.get('validation', {'noise_type': r['args']['noise_type'], 'noise_rate': r['args']['noise_rate'], 'matched': True})
        key = (r['args']['noise_type'], round(r['args']['noise_rate'], 2), v['noise_type'], round(v['noise_rate'], 2), v['matched'])
        cells.setdefault(key, {})[r['args']['seed']] = r['report']
    if not cells:
        return None
    out = []
    for (t, rate, vt, vr, matched), runs in sorted(cells.items(), key=lambda kv: (not kv[0][4], kv[0][0], kv[0][1])):
        seeds = sorted(runs)
        gain = lambda m: [runs[s]['noisy'][m]['recall'] - runs[s]['noisy']['unrepaired']['recall'] for s in seeds]
        entry = {'noise_type': t, 'noise_rate': rate, 'val_type': vt, 'val_rate': vr, 'matched': matched, 'n': len(seeds),
                 'unrepaired': float(np.mean([runs[s]['noisy']['unrepaired']['recall'] for s in seeds])),
                 'clean_unrepaired': float(np.mean([runs[s]['clean']['unrepaired']['recall'] for s in seeds])),
                 'gain': {m: gain(m) for m in ('vgr', 'm1', 'm3', 'm3_random', 'm3_shuffled')},
                 'm3_plus_vgr': [runs[s]['noisy_plus_vgr']['m3']['recall'] - runs[s]['noisy']['unrepaired']['recall'] for s in seeds],
                 'm3_minus_m1': [runs[s]['noisy']['m3']['recall'] - runs[s]['noisy']['m1']['recall'] for s in seeds],
                 'm3_clean_change': [runs[s]['clean']['m3']['recall'] - runs[s]['clean']['unrepaired']['recall'] for s in seeds],
                 'auroc_cosine': float(np.mean([runs[s]['auroc']['vgr_cosine'] for s in seeds]))}
        out.append(entry)
    return out


def render_repair_grid(grid):
    pos = lambda d: f"{np.mean(d):+.4f} ({sum(x > 0 for x in d)}/{len(d)})"
    head = ['| Test noise | validation noise | unrepaired | VGR | M1 | M3 | M3, random | M3, permuted | M3 - M1 | M3 + VGR | M3 on clean | AUROC cosine |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    row = lambda e: (f"| {e['noise_type']} {e['noise_rate']} | {e['val_type']} {e['val_rate']} | {e['unrepaired']:.4f} | "
                     + ' | '.join(pos(e['gain'][m]) for m in ('vgr', 'm1', 'm3', 'm3_random', 'm3_shuffled'))
                     + f" | {pos(e['m3_minus_m1'])} | {pos(e['m3_plus_vgr'])} | {np.mean(e['m3_clean_change']):+.4f} | {e['auroc_cosine']:.3f} |")
    out = ['## Text repair under other noise (Recall@20; gains over the unrepaired model, mean over seeds, seeds above 0 in brackets)', '']
    matched = [e for e in grid if e['matched']]
    out += ['Validation noise equal to the test noise:', ''] + head + [row(e) for e in matched]
    mism = [e for e in grid if not e['matched']]
    if mism:
        ref = {(e['noise_type'], e['noise_rate']): e for e in matched}
        out += ['', 'Validation noise fixed at hashtag 0.4, test noise different (share of the matched M3 gain kept in the last column):', '',
                head[0] + ' M3 gain kept |', head[1] + '---|']
        for e in mism:
            m = ref.get((e['noise_type'], e['noise_rate']))
            share = f"{np.mean(e['gain']['m3']) / np.mean(m['gain']['m3']):.0%}" if m and np.mean(m['gain']['m3']) else 'n/a'
            out.append(row(e) + f' {share} |')
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
    probe = build_gate_probe(a.results_dir, a.noise_rate, a.noise_type)
    if probe:
        text += '\n' + '\n'.join(render_gate_probe(probe)) + '\n'
    repair = build_repair(a.results_dir, a.noise_rate, a.noise_type)
    if repair:
        text += '\n' + '\n'.join(render_repair(repair)) + '\n'
    grid = build_repair_grid(a.results_dir)
    if grid:
        text += '\n' + '\n'.join(render_repair_grid(grid)) + '\n'
    repr_probe = build_repr_probe(a.results_dir, a.noise_rate, a.noise_type)
    if repr_probe:
        text += '\n' + '\n'.join(render_repr_probe(repr_probe)) + '\n'
    with open(a.out, 'w') as f:
        f.write(text)
    return tables


if __name__ == '__main__':
    main()
