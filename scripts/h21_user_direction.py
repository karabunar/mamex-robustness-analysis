"""H21: swap the user-direction component of the image expert output between cold and warm items and read the gate's text weight.

usage: PYTHONPATH=src python scripts/h21_user_direction.py   (writes results/h21.md and results/h21.json)
"""
import json
import os

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

RUNS = [('augmented', 2025, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2025.pt'),
        ('augmented', 2026, 'results/ckpt_aug/checkpoints/stress_test_hashtag_r0.40_g8.0_seed2026.pt'),
        ('clean-trained', 2025, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2025.pt'),
        ('clean-trained', 2026, 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed2026.pt')]


def swap_component(v, direction, target_mean):
    """v with its component along the unit vector `direction` replaced by target_mean (a scalar)."""
    p = (v @ direction).unsqueeze(1)
    return v + (target_mean - p) * direction


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

            def w_t(v, t):
                return float(torch.softmax(model.gate(torch.cat([v, t], dim=1)), dim=1)[:, 1].mean())

            u = torch.nn.functional.normalize(model.user_emb.weight.mean(dim=0), dim=0)
            generator = torch.Generator().manual_seed(0)
            randoms = [torch.nn.functional.normalize(torch.randn(u.shape[0], generator=generator), dim=0) for _ in range(5)]
            vw, vc, tw, tc = img[warm], img[cold], txt[warm], txt[cold]

            def effect(direction):
                to_warm = swap_component(vc, direction, float((vw @ direction).mean()))
                to_cold = swap_component(vw, direction, float((vc @ direction).mean()))
                return {'cold_to_warm': w_t(to_warm, tc) - w_t(vc, tc), 'warm_to_cold': w_t(to_cold, tw) - w_t(vw, tw)}
            user = effect(u)
            rand = [effect(r) for r in randoms]
            rows.append({'model': name, 'seed': seed, 'w_t_cold': w_t(vc, tc), 'w_t_warm': w_t(vw, tw),
                         'mean_p_cold': float((vc @ u).mean()), 'mean_p_warm': float((vw @ u).mean()),
                         'user': user, 'random_mean': {k: float(np.mean([r[k] for r in rand])) for k in user},
                         'random_abs_mean': {k: float(np.mean([abs(r[k]) for r in rand])) for k in user}})
        print(rows[-1], flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/h21.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H21: change of the mean text weight when the user-direction component of the image output is swapped (clean text)', '',
           '| model | seed | w_t cold | w_t warm | cold gets the warm component (user dir.) | same, 5 random dirs (mean abs.) | warm gets the cold component (user dir.) | same, random (mean abs.) |',
           '|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['model']} | {r['seed']} | {r['w_t_cold']:.3f} | {r['w_t_warm']:.3f} | {r['user']['cold_to_warm']:+.3f} | "
                   f"{r['random_abs_mean']['cold_to_warm']:.3f} | {r['user']['warm_to_cold']:+.3f} | {r['random_abs_mean']['warm_to_cold']:.3f} |")
    open('results/h21.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
