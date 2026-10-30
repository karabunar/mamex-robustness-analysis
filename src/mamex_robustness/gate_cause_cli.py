import argparse
import json
import os

import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model


def main(argv=None):
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--ckpt', required=True, help='saved best-validation weights (--save_final)')
    own.add_argument('--out', required=True, help='json file to write')
    own.add_argument('--probe_seed', type=int, default=0)
    own_args, rest = own.parse_known_args(argv)
    args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline'] + rest))
    env = Env(args)
    loader = Loader4MM(env)
    model = MOME_model(env, loader)
    model.load_state_dict(torch.load(own_args.ckpt, map_location='cpu'))
    report = Session(env, model, loader).gate_cause(own_args.probe_seed)
    os.makedirs(os.path.dirname(os.path.abspath(own_args.out)), exist_ok=True)
    with open(own_args.out, 'w') as f:
        json.dump({'checkpoint': own_args.ckpt, 'seed': args.seed, 'report': report}, f, indent=2)
    tool.cprint(f'results written to {own_args.out}')


if __name__ == '__main__':
    main()
