"""H14: (a) test-set metrics when the candidates are the test items only instead of all cold items;
(b) the per-metric maxima over the evaluation epochs of each seed, against the published vector.

usage: PYTHONPATH=src python scripts/h14_candidates.py --data_path processed_data_baby/cold   (writes results/h14.md and results/h14.json)
"""
import argparse
import json
import os
import statistics as st

import numpy as np
import torch

from mamex_robustness.application.environment import Env
from mamex_robustness.application.evaluation import num_faiss_evaluate
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SEEDS = (2025, 2026, 2027)
PUBLISHED = {'Rec@10': 0.0512, 'Rec@20': 0.0771, 'NDCG@10': 0.0292, 'NDCG@20': 0.0363}


def candidate_metrics(session, candidates):
    user_emb, item_emb = session._embeddings()
    data = session.dataset.test_data
    _, recall, ndcg = num_faiss_evaluate(data, list(data.keys()), list(candidates), session.dataset.train_data, [10, 20], user_emb, item_emb)
    return {'Rec@10': float(recall[10]), 'Rec@20': float(recall[20]), 'NDCG@10': float(ndcg[10]), 'NDCG@20': float(ndcg[20])}


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--ckpt_dir', default=os.path.join(ROOT, 'results', 'ckpt291', 'checkpoints'))
    own_args, rest = own.parse_known_args()
    rows = {}
    for seed in SEEDS:
        args = parse_args(['--mode', 'baseline', '--seed', str(seed)] + rest)
        env = Env(args)
        loader = Loader4MM(env)
        model = MOME_model(env, loader)
        model.load_state_dict(torch.load(os.path.join(own_args.ckpt_dir, f'baseline_hashtag_r0.00_g8.0_seed{seed}.pt'), map_location='cpu'))
        session = Session(env, model, loader)
        session.model.eval()
        test_items = sorted({int(i) for items in loader.test_data.values() for i in items})
        rows[seed] = {'all cold items': candidate_metrics(session, loader.cold_item_index),
                      'test items only': candidate_metrics(session, test_items),
                      'n_candidates': {'all cold items': len(loader.cold_item_index), 'test items only': len(test_items)}}
    curves = {}
    for seed in SEEDS:
        with open(os.path.join(ROOT, 'results', 'curves291', 'curves', f'baseline_hashtag_r0.00_g8.0_seed{seed}.json')) as f:
            c = json.load(f)['curve']
        curves[seed] = {'Rec@10': max(p['test_recall']['10'] for p in c), 'Rec@20': max(p['test_recall']['20'] for p in c),
                        'NDCG@10': max(p['test_ndcg']['10'] for p in c), 'NDCG@20': max(p['test_ndcg']['20'] for p in c)}
    keys = list(PUBLISHED)
    out = ['## H14: two further candidate explanations of the Rec@10 gap (clean baselines, seeds 2025 to 2027)', '',
           f'Published: ' + ', '.join(f'{k} {v}' for k, v in PUBLISHED.items()), '',
           '### (a) Candidate set for the test ranking', '', '| candidates | n | ' + ' | '.join(keys) + ' |', '|---|---|' + '---|' * len(keys)]
    for name in ('all cold items', 'test items only'):
        mean = {k: st.mean(rows[s][name][k] for s in SEEDS) for k in keys}
        out.append(f"| {name} (mean of 3 seeds) | {rows[SEEDS[0]]['n_candidates'][name]} | " + ' | '.join(f'{mean[k]:.4f}' for k in keys) + ' |')
    all_, only = ({k: st.mean(rows[s][n][k] for s in SEEDS) for k in keys} for n in ('all cold items', 'test items only'))
    out.append('| relative change | | ' + ' | '.join(f'{only[k] / all_[k] - 1:+.1%}' for k in keys) + ' |')
    out += ['', '### (b) Best test value over the evaluation epochs, per metric', '', '| seed | ' + ' | '.join(keys) + ' | within tolerance of all four |', '|---|' + '---|' * (len(keys) + 1)]
    tol = {'Rec@10': 0.001, 'Rec@20': 0.001, 'NDCG@10': 0.0005, 'NDCG@20': 0.0005}
    for seed in SEEDS:
        ok = all(abs(curves[seed][k] - PUBLISHED[k]) <= tol[k] for k in keys)
        out.append(f'| {seed} | ' + ' | '.join(f'{curves[seed][k]:.4f}' for k in keys) + f' | {"yes" if ok else "no"} |')
    mean = {k: st.mean(curves[s][k] for s in SEEDS) for k in keys}
    ok = all(abs(mean[k] - PUBLISHED[k]) <= tol[k] for k in keys)
    out.append('| mean of maxima | ' + ' | '.join(f'{mean[k]:.4f}' for k in keys) + f' | {"yes" if ok else "no"} |')
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', 'h14.md'), 'w') as f:
        f.write(text)
    with open(os.path.join(ROOT, 'results', 'h14.json'), 'w') as f:
        json.dump({'candidates': rows, 'epoch_maxima': curves}, f, indent=2)
    print(text)


if __name__ == '__main__':
    main()
