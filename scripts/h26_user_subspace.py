"""H26: swap the projection of the image expert output onto the top-k user subspace between cold and warm items; read the gate's text weight.

usage: PYTHONPATH=src python scripts/h26_user_subspace.py   (writes results/h26.md and results/h26.json)
"""
import json

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

RUNS = [('augmented', 2025, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2025.pt'),
        ('augmented', 2026, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2026.pt')]
KS = (5, 20)


def swap_subspace(v, basis, target_mean):
    """v with its projection onto the orthonormal columns of `basis` replaced by target_mean (a vector of coordinates)."""
    coords = v @ basis
    return v + (target_mean - coords) @ basis.T


def main():
    rows = []
    for name, seed, path in RUNS:
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--use_gpu', '0', '--data_path', 'processed_data_baby/cold']))
        env = Env(args)
        data = Loader4MM(env)
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        warm, cold = torch.as_tensor(np.asarray(data.warm_item_index)), torch.as_tensor(np.asarray(data.cold_item_index))
        with torch.no_grad():
            img, _ = model.image_linear(model.image_feat)
            txt, _ = model.text_linear(model.text_feat)
            vw, vc, tc = img[warm], img[cold], txt[cold]

            def w_t(v):
                return float(torch.softmax(model.gate(torch.cat([v, tc], dim=1)), dim=1)[:, 1].mean())
            base = w_t(vc)
            _, _, vh = torch.linalg.svd(model.user_emb.weight, full_matrices=False)
            generator = torch.Generator().manual_seed(0)
            row = {'model': name, 'seed': seed, 'w_t_cold': base}
            for k in KS:
                basis = vh[:k].T
                row[f'k{k}_user_drop'] = base - w_t(swap_subspace(vc, basis, (vw @ basis).mean(dim=0)))
                drops = []
                for _ in range(5):
                    q, _ = torch.linalg.qr(torch.randn(vh.shape[1], k, generator=generator))
                    drops.append(base - w_t(swap_subspace(vc, q, (vw @ q).mean(dim=0))))
                row[f'k{k}_random_abs_drop'] = float(np.mean(np.abs(drops)))
            rows.append(row)
        print(row, flush=True)
    with open('results/h26.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H26: drop of the mean text weight on cold items when the projection onto a k-dimensional subspace is replaced by the warm mean (clean text)', '',
           '| model | seed | w_t cold | k=5 user subspace | k=5 random (mean abs.) | k=20 user subspace | k=20 random (mean abs.) |', '|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['model']} | {r['seed']} | {r['w_t_cold']:.3f} | {r['k5_user_drop']:+.3f} | {r['k5_random_abs_drop']:.3f} | {r['k20_user_drop']:+.3f} | {r['k20_random_abs_drop']:.3f} |")
    open('results/h26.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
