import argparse
import json
import os

import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.repair_eval import evaluate_repairs
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model


def main(argv=None):
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--ckpt_dir', default='results/ckpt291/checkpoints')
    own.add_argument('--out_dir', default='results/repair')
    own.add_argument('--k', type=int, default=10)
    own.add_argument('--val_noise_type', default=None, help='validation noise type when it differs from --noise_type')
    own.add_argument('--val_noise_rate', type=float, default=None, help='validation noise rate when it differs from --noise_rate')
    own_args, rest = own.parse_known_args(argv)
    args = tool.apply_mode_defaults(parse_args(['--mode', 'stress_test'] + rest))
    env = Env(args)
    loader = Loader4MM(env)
    model = MOME_model(env, loader)
    path = os.path.join(own_args.ckpt_dir, f'baseline_hashtag_r0.00_g8.0_seed{args.seed}.pt')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    session = Session(env, model, loader)
    val_type = own_args.val_noise_type or args.noise_type
    val_rate = args.noise_rate if own_args.val_noise_rate is None else own_args.val_noise_rate
    matched = (val_type, val_rate) == (args.noise_type, args.noise_rate)
    report = evaluate_repairs(session, args.noise_rate, args.noise_type, args.noise_seed, own_args.k,
                              val_noise=None if matched else (val_type, val_rate))
    os.makedirs(own_args.out_dir, exist_ok=True)
    name = f'repair_{args.noise_type}_r{args.noise_rate:.2f}_seed{args.seed}'
    if not matched:
        name += f'_val{val_type}{val_rate:.2f}'
    out = os.path.join(own_args.out_dir, name + '.json')
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'checkpoint': path, 'validation': {'noise_type': val_type, 'noise_rate': val_rate, 'matched': matched},
                   'report': report}, f, indent=2)
    tool.cprint(f'results written to {out}')


if __name__ == '__main__':
    main()
