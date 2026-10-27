"""Aggregate result JSONs over seeds and draw the two report figures."""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette, categorical slots 1-3 (blue, orange, aqua).
COLORS = {'Clean baseline': '#2a78d6', 'Stress test': '#eb6834', 'MAMEX-VGR': '#1baf7a',
          'Clean text': '#2a78d6', 'Noisy text': '#eb6834', 'Noisy text + VGR': '#1baf7a'}
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _load(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def aggregate(results_dir, k=20, noise_rate=0.4, noise_type='hashtag', vgr_gamma=None):
    """Mean/std over seeds. Groups with no matching runs are omitted, never filled in.

    vgr_gamma=None: MAMEX-VGR uses the runs whose gamma was selected on validation data (the headline result);
    a number selects the manual-gamma runs with exactly that gamma (sensitivity analysis).
    """
    runs = [r for r in _load(results_dir) if r['args'].get('suffix', 'default') == 'default']

    def select(mode, filter_noise=True, filter_gamma=False):
        out = []
        for r in runs:
            a = r['args']
            if a['mode'] != mode:
                continue
            if filter_noise and not (abs(a['noise_rate'] - noise_rate) < 1e-9 and a['noise_type'] == noise_type):
                continue
            if filter_gamma:
                selected = 'gamma_selection' in r['report']
                if vgr_gamma is None and not selected:
                    continue
                if vgr_gamma is not None and (selected or abs(a['vgr_gamma'] - vgr_gamma) > 1e-9):
                    continue
            out.append(r)
        return out

    groups = {'Clean baseline': (select('baseline', filter_noise=False), 'clean'),
              'Stress test': (select('stress_test'), 'noisy'),
              'MAMEX-VGR': (select('improved', filter_gamma=True), 'noisy_vgr')}
    accuracy = {}
    for label, (rs, block) in groups.items():
        if rs:
            accuracy[label] = {m: _mean_std([r['report'][block][m][str(k)] for r in rs])
                               for m in ('hr', 'recall', 'ndcg')}
            accuracy[label]['n'] = len(rs)

    stress, improved = select('stress_test'), select('improved', filter_gamma=True)
    gate_groups = {'Clean text': (stress, 'clean'), 'Noisy text': (stress, 'noisy'),
                   'Noisy text + VGR': (improved, 'noisy_vgr')}
    gate = {}
    for label, (rs, block) in gate_groups.items():
        if rs:
            gate[label] = {w: _mean_std([r['report']['gate'][block][w] for r in rs]) for w in ('w_v', 'w_t')}
            gate[label]['n'] = len(rs)
    return {'accuracy': accuracy, 'gate': gate}


def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, positions, means, stds, color, label, width):
    ax.bar(positions, means, width, yerr=stds, color=color, label=label, error_kw={'ecolor': INK_MUTED, 'capsize': 3})
    for x, m, sd in zip(positions, means, stds):
        ax.text(x, m + sd, f'{m:.3f}', ha='center', va='bottom', fontsize=8, color=INK)


def plot_accuracy(agg, k, path):
    labels = list(agg['accuracy'])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        d = agg['accuracy'][label]
        means, stds = [d['recall'][0], d['ndcg'][0]], [d['recall'][1], d['ndcg'][1]]
        _bars(ax, x + (i - (len(labels) - 1) / 2) * width, means, stds, COLORS[label],
              f"{label} (n={d['n']})", width * 0.9)
    top = max(agg['accuracy'][l][m][0] + agg['accuracy'][l][m][1] for l in labels for m in ('recall', 'ndcg'))
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(x, [f'Recall@{k}', f'NDCG@{k}'])
    ax.set_ylabel('Cold-item test metric (mean ± std over seeds)', color=INK_MUTED)
    ax.set_title('Accuracy under text noise', color=INK, loc='left')
    ax.legend(frameon=False, ncol=len(labels), loc='upper center', fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gate(agg, path):
    labels = list(agg['gate'])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        d = agg['gate'][label]
        means, stds = [d['w_v'][0], d['w_t'][0]], [d['w_v'][1], d['w_t'][1]]
        _bars(ax, x + (i - (len(labels) - 1) / 2) * width, means, stds, COLORS[label],
              f"{label} (n={d['n']})", width * 0.9)
    ax.set_xticks(x, ['Visual weight $w_v$', 'Textual weight $w_t$'])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Mean gate weight on corrupted cold items', color=INK_MUTED)
    ax.set_title('Gating shift under text noise', color=INK, loc='left')
    ax.legend(frameon=False, ncol=len(labels), loc='upper center', fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description='Plot MAMEX robustness results')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--noise_rate', type=float, default=0.4)
    p.add_argument('--noise_type', default='hashtag')
    p.add_argument('--vgr_gamma', type=float, default=None)
    a = p.parse_args(argv)
    agg = aggregate(a.results_dir, a.k, a.noise_rate, a.noise_type, a.vgr_gamma)
    if not agg['accuracy']:
        raise SystemExit(f'no result JSON files found in {a.results_dir}')
    plot_accuracy(agg, a.k, os.path.join(a.results_dir, 'accuracy_comparison.png'))
    if agg['gate']:
        plot_gate(agg, os.path.join(a.results_dir, 'gating_weight_shift.png'))
    return agg


if __name__ == '__main__':
    main()
