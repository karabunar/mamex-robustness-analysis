"""H15: where does the gain of the cosine-gated repair M3 come from? M3 applied to all cold items, only to the truly
corrupted ones, and only to the uncorrupted ones (hashtag 0.4, noisy test text, M3 parameters as selected on validation data).

usage: PYTHONPATH=src python scripts/h15_decomposition.py --data_path processed_data_baby/cold   (writes results/h15.md and results/h15.json)
"""
import argparse
import json
import os
import statistics as st

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
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027, 2028, 2029])
    own.add_argument('--ckpt_dir', default=os.path.join(ROOT, 'results', 'ckpt291', 'checkpoints'))
    own_args, rest = own.parse_known_args()
    rows = {}
    for seed in own_args.seeds:
        args = parse_args(['--mode', 'baseline', '--seed', str(seed), '--noise_rate', '0.4'] + rest)
        args.noise_seed = seed
        env = Env(args)
        data = Loader4MM(env)
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(os.path.join(own_args.ckpt_dir, f'baseline_hashtag_r0.00_g8.0_seed{seed}.pt'), map_location='cpu'))
        session = Session(env, model, data)
        session.model.eval()
        with open(os.path.join(ROOT, 'results', 'repair', f'repair_hashtag_r0.40_seed{seed}.json')) as f:
            param = json.load(f)['report']['selection']['m3']['param']
        cold = np.asarray(data.cold_item_index)
        noisy = data.noisy_text_feat(0.4, 'hashtag', seed)
        corrupted = (noisy[cold] != data.text_feat[cold]).any(axis=1)
        knn = repair.neighbor_text(data.image_feat, data.text_feat, data.warm_item_index, cold, 10)
        lam = repair.quantile_lambda(_vgr_cosine(session, noisy)[cold], param[0], param[2], param[1])

        def recall(weights):
            text = repair.with_cold_rows(noisy, cold, repair.blend(noisy[cold], knn, weights))
            return _scores(session, text, 'test')['recall']
        rows[seed] = {'param': param, 'unrepaired': _scores(session, noisy, 'test')['recall'], 'clean': _scores(session, None, 'test')['recall'],
                      'all': recall(lam), 'corrupted_only': recall(np.where(corrupted, lam, 0.0)), 'clean_only': recall(np.where(corrupted, 0.0, lam)),
                      'mean_lambda_corrupted': float(lam[corrupted].mean()), 'mean_lambda_clean': float(lam[~corrupted].mean())}
    seeds = list(rows)
    out = ['## H15: where the gain of M3 comes from (hashtag 0.4, Recall@20, gains over the unrepaired model on noisy text)', '',
           '| seed | unrepaired | M3 on all cold items | M3 on corrupted items only | M3 on uncorrupted items only | share from corrupted only | mean lambda corrupted / clean |',
           '|---|---|---|---|---|---|---|']
    shares = []
    for s in seeds:
        r = rows[s]
        full, c_only, k_only = r['all'] - r['unrepaired'], r['corrupted_only'] - r['unrepaired'], r['clean_only'] - r['unrepaired']
        shares.append(c_only / full)
        out.append(f"| {s} | {r['unrepaired']:.4f} | {full:+.4f} | {c_only:+.4f} | {k_only:+.4f} | {c_only / full:.0%} | {r['mean_lambda_corrupted']:.2f} / {r['mean_lambda_clean']:.2f} |")
    mean = {k: st.mean(rows[s][k] - rows[s]['unrepaired'] for s in seeds) for k in ('all', 'corrupted_only', 'clean_only')}
    out.append(f"| mean | {st.mean(rows[s]['unrepaired'] for s in seeds):.4f} | {mean['all']:+.4f} | {mean['corrupted_only']:+.4f} | {mean['clean_only']:+.4f} | {mean['corrupted_only'] / mean['all']:.0%} | |")
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', 'h15.md'), 'w') as f:
        f.write(text)
    with open(os.path.join(ROOT, 'results', 'h15.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(text)


if __name__ == '__main__':
    main()
