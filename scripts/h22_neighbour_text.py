"""H22: how informative is the neighbour text (mean text of the 10 image-nearest warm items) on Baby and on Sports?

usage: PYTHONPATH=src python scripts/h22_neighbour_text.py   (writes results/h22.md and results/h22.json)
"""
import json
import statistics as st

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.repair_eval import _scores
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.domain import repair
from mamex_robustness.models.mamex import MOME_model

DATASETS = [('Baby', 'processed_data_baby/cold', 'results/ckpt291/checkpoints'),
            ('Sports', '/media/baykar/disk27/mamex_data/sports/cold', 'results/sports/checkpoints')]


def main():
    result = {}
    for name, path, ckpt_dir in DATASETS:
        shares, cosines = [], []
        for seed in (2025, 2026, 2027):
            args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0', '--data_path', path]))
            env = Env(args)
            data = Loader4MM(env)
            model = MOME_model(env, data)
            model.load_state_dict(torch.load(f'{ckpt_dir}/baseline_hashtag_r0.00_g8.0_seed{seed}.pt', map_location='cpu'))
            session = Session(env, model, data)
            session.model.eval()
            cold = np.asarray(data.cold_item_index)
            knn = repair.neighbor_text(data.image_feat, data.text_feat, data.warm_item_index, cold, 10)
            cosines.append(float(repair.agreement(data.text_feat[cold], knn).mean()))
            clean = _scores(session, None, 'test')['recall']
            replaced = _scores(session, repair.with_cold_rows(data.text_feat, cold, knn), 'test')['recall']
            shares.append({'seed': seed, 'clean': clean, 'replaced': replaced, 'share': replaced / clean})
            print(name, shares[-1], flush=True)
        result[name] = {'mean_cosine': cosines[0], 'cosine_is_seed_free': max(cosines) - min(cosines) < 1e-6, 'runs': shares,
                        'mean_share': st.mean(r['share'] for r in shares)}
    with open('results/h22.json', 'w') as f:
        json.dump(result, f, indent=2)
    out = ['## H22: neighbour text on Baby and on Sports (clean text; all cold items)', '',
           '| dataset | mean cosine(own text, neighbour text) | clean Recall@20 | Recall@20 with neighbour text only | share kept |', '|---|---|---|---|---|']
    for name, r in result.items():
        out.append(f"| {name} | {r['mean_cosine']:.3f} | {st.mean(x['clean'] for x in r['runs']):.4f} | {st.mean(x['replaced'] for x in r['runs']):.4f} | {r['mean_share']:.1%} |")
    open('results/h22.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
