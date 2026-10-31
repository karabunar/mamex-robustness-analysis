"""H23: do M3's gains add up over a partition of the cold items? Corrupted/uncorrupted against 5 random halves (seeded).

usage: PYTHONPATH=src python scripts/h23_additivity.py   (writes results/h23.md and results/h23.json)
"""
import json
import os

import numpy as np
import torch

from mamex_robustness.application.environment import Env
from mamex_robustness.application.repair_eval import _scores, _vgr_cosine
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.domain import repair
from mamex_robustness.models.mamex import MOME_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    rows = {}
    for seed in (2025, 2026, 2027, 2028, 2029):
        args = parse_args(['--mode', 'baseline', '--seed', str(seed), '--noise_rate', '0.4', '--use_gpu', '0', '--data_path', 'processed_data_baby/cold'])
        args.noise_seed = seed
        env = Env(args)
        data = Loader4MM(env)
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(os.path.join(ROOT, 'results', 'ckpt291', 'checkpoints', f'baseline_hashtag_r0.00_g8.0_seed{seed}.pt'), map_location='cpu'))
        session = Session(env, model, data)
        session.model.eval()
        with open(os.path.join(ROOT, 'results', 'repair', f'repair_hashtag_r0.40_seed{seed}.json')) as f:
            param = json.load(f)['report']['selection']['m3']['param']
        cold = np.asarray(data.cold_item_index)
        noisy = data.noisy_text_feat(0.4, 'hashtag', seed)
        corrupted = (noisy[cold] != data.text_feat[cold]).any(axis=1)
        knn = repair.neighbor_text(data.image_feat, data.text_feat, data.warm_item_index, cold, 10)
        lam = repair.quantile_lambda(_vgr_cosine(session, noisy)[cold], param[0], param[2], param[1])
        base = _scores(session, noisy, 'test')['recall']

        def gain(weights):
            return _scores(session, repair.with_cold_rows(noisy, cold, repair.blend(noisy[cold], knn, weights)), 'test')['recall'] - base

        def interaction(mask):
            full = gain(lam)
            return full - gain(np.where(mask, lam, 0.0)) - gain(np.where(mask, 0.0, lam))
        rng = np.random.default_rng(seed)
        size = int(corrupted.sum())
        randoms = []
        for _ in range(5):
            mask = np.zeros(len(cold), dtype=bool)
            mask[rng.choice(len(cold), size=size, replace=False)] = True
            randoms.append(interaction(mask))
        rows[seed] = {'corrupted_partition': interaction(corrupted), 'random_halves': randoms,
                      'random_mean_abs': float(np.mean(np.abs(randoms))), 'random_mean': float(np.mean(randoms))}
        print(seed, rows[seed], flush=True)
    with open(os.path.join(ROOT, 'results', 'h23.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H23: interaction (full gain minus the sum of the two parts) of M3 over two kinds of partition, Recall@20', '',
           '| seed | corrupted / uncorrupted | random partitions of the same size: mean | mean absolute | largest absolute |', '|---|---|---|---|---|']
    for seed, r in rows.items():
        out.append(f"| {seed} | {r['corrupted_partition']:+.4f} | {r['random_mean']:+.4f} | {r['random_mean_abs']:.4f} | {max(abs(x) for x in r['random_halves']):.4f} |")
    open(os.path.join(ROOT, 'results', 'h23.md'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
