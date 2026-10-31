"""H19: gate text weight of warm items against their number of training interactions (clean text, saved checkpoints)."""
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

RUNS = [('augmented', 2025, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2025.pt'),
        ('augmented', 2026, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2026.pt'),
        ('clean-trained', 2025, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2025.pt'),
        ('clean-trained', 2026, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2026.pt')]


def text_weight(model):
    model.eval()
    with torch.no_grad():
        image_emb, _ = model.image_linear(model.image_feat)
        text_emb, _ = model.text_linear(model.text_feat)
        return torch.softmax(model.gate(torch.cat([image_emb, text_emb], dim=1)), dim=1)[:, 1].cpu().numpy()


def main():
    rows = []
    for name, seed, path in RUNS:
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0', '--data_path', 'processed_data_baby/cold']))
        env = Env(args)
        loader = Loader4MM(env)
        counts = np.zeros(loader.m_item)
        for items in loader.train_data.values():
            for i in items:
                counts[i] += 1
        model = MOME_model(env, loader)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        w = text_weight(model)
        warm = np.asarray(loader.warm_item_index)
        n, w = counts[warm], w[warm]
        rho = float(spearmanr(w, np.log(n)).correlation)
        order = np.argsort(n, kind='stable')
        fifth = len(order) // 5
        low, high = w[order[:fifth]].mean(), w[order[-fifth:]].mean()
        rows.append({'model': name, 'seed': seed, 'n_warm': int(len(warm)), 'spearman': rho, 'w_t_lowest_fifth': float(low),
                     'w_t_highest_fifth': float(high), 'gap': float(low - high), 'median_count_lowest': float(np.median(n[order[:fifth]])),
                     'median_count_highest': float(np.median(n[order[-fifth:]]))})
        print(rows[-1], flush=True)
    with open('results/h19.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H19: text weight of warm items against their number of training interactions (clean text)', '',
           '| model | seed | Spearman(w_t, log n) | w_t lowest fifth | w_t highest fifth | gap | median n lowest / highest |', '|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['model']} | {r['seed']} | {r['spearman']:+.3f} | {r['w_t_lowest_fifth']:.3f} | {r['w_t_highest_fifth']:.3f} | "
                   f"{r['gap']:+.3f} | {r['median_count_lowest']:.0f} / {r['median_count_highest']:.0f} |")
    open('results/h19.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
