"""H20: which statistic of a warm item's image expert output tracks the gate's text weight (clean text, saved checkpoints)."""
import json

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

RUNS = [('augmented', 2025, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2025.pt'),
        ('augmented', 2026, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2026.pt'),
        ('clean-trained', 2025, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2025.pt'),
        ('clean-trained', 2026, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2026.pt')]
STATS = ['S1 user direction', 'S2 density (output)', 'S3 density (raw feature)', 'S4 norm']
K = 10


def knn_density(x, k=K):
    x = torch.nn.functional.normalize(x, dim=1)
    sims = x @ x.T
    sims.fill_diagonal_(-2)
    return sims.topk(k, dim=1).values.mean(dim=1).numpy()


def residual_rank(a, control):
    ra, rc = rankdata(a), rankdata(control)
    slope = np.cov(ra, rc)[0, 1] / np.var(rc, ddof=1)
    return ra - slope * rc


def main():
    rows = []
    for name, seed, path in RUNS:
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0',
                                                    '--data_path', 'processed_data_baby/cold']))
        env = Env(args)
        loader = Loader4MM(env)
        counts = np.zeros(loader.m_item)
        for items in loader.train_data.values():
            for i in items:
                counts[i] += 1
        model = MOME_model(env, loader)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        warm = np.asarray(loader.warm_item_index)
        with torch.no_grad():
            image_emb, _ = model.image_linear(model.image_feat)
            text_emb, _ = model.text_linear(model.text_feat)
            w = torch.softmax(model.gate(torch.cat([image_emb, text_emb], dim=1)), dim=1)[:, 1].numpy()
            user_mean = model.user_emb.weight.mean(dim=0, keepdim=True)
            s1 = torch.nn.functional.cosine_similarity(image_emb, user_mean, dim=1).numpy()
            stats = {STATS[0]: s1[warm], STATS[1]: knn_density(image_emb[warm]),
                     STATS[2]: knn_density(model.image_feat[warm]), STATS[3]: image_emb[warm].norm(dim=1).numpy()}
        w, logn = w[warm], np.log(counts[warm])
        row = {'model': name, 'seed': seed, 'log_n': float(spearmanr(w, logn).correlation)}
        for key, v in stats.items():
            row[key] = {'raw': float(spearmanr(w, v).correlation),
                        'partial': float(np.corrcoef(residual_rank(w, logn), residual_rank(v, logn))[0, 1])}
        rows.append(row)
        print(row, flush=True)
    with open('results/h20.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H20: Spearman correlation of the gate text weight with statistics of the image expert output (warm items, clean text)', '',
           '| model | seed | log n | ' + ' | '.join(f'{s}: raw / without log n' for s in STATS) + ' |', '|---|---|---|' + '---|' * len(STATS)]
    for r in rows:
        out.append(f"| {r['model']} | {r['seed']} | {r['log_n']:+.3f} | " +
                   ' | '.join(f"{r[s]['raw']:+.3f} / {r[s]['partial']:+.3f}" for s in STATS) + ' |')
    open('results/h20.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
