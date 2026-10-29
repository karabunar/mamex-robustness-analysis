import argparse
import json
import os
import time

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.session import Session
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MAMEX robustness study")

    # ----------------------- Experiment mode
    parser.add_argument('--mode', choices=['baseline', 'stress_test', 'improved'], default='baseline')
    parser.add_argument('--noise_rate', type=float, default=0.4)
    parser.add_argument('--noise_type', choices=['hashtag', 'shuffle', 'gaussian'], default='hashtag')
    parser.add_argument('--noise_seed', type=int, default=None)
    parser.add_argument('--use_vgr', type=int, default=None)
    parser.add_argument('--train_with_vgr', type=int, default=0,
                        help='1 = also apply VGR during training (ablation); default trains clean and applies VGR at evaluation')
    parser.add_argument('--train_noise_rate', type=float, default=0.0,
                        help='ablation: corrupt this share of warm items\' text (--noise_type) in every training epoch; '
                             'validation and model selection stay clean; 0 = train on clean text')
    parser.add_argument('--gate_probe', type=int, default=0,
                        help='1 = also report the mean gate weights on warm items split by a training-style corruption (rate --noise_rate)')
    parser.add_argument('--vgr_gamma', type=float, default=8.0)
    parser.add_argument('--gamma_grid', type=float, nargs='+', default=None,
                        help='select vgr_gamma on noisy validation data from these values')
    parser.add_argument('--vgr_control', choices=['cosine', 'shuffled', 'constant'], default='cosine')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--data_path', type=str, default='baby')

    # ----------------------- File Identification
    parser.add_argument('--suffix', type=str, default='default')

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    # ------------------------ Training Setting
    parser.add_argument('--free_emb_dimension', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eva_interval', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    parser.add_argument('--dataset', type=str, default='baby')

    # ----------------------- Regularizer coefficient
    parser.add_argument('--reg_coeff', type=float, default=1e-6)
    parser.add_argument('--penalty_coeff', type=float, default=50) 
    parser.add_argument('--align_coeff', type=float, default=1) 
    parser.add_argument('--moe_balance_coeff', type=float, default=1) 
    parser.add_argument('--fusion_balance_coeff', type=float, default=1) 


    # ----------------------- logger
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args(argv)



def main(argv=None):
    # ----------------------------------- Env Init -----------------------------------------------------------
    tool.cprint('Init Env')
    args = tool.apply_mode_defaults(parse_args(argv))
    my_env = Env(args)
    tool.cprint(f'---------- {my_env.args.suffix} ----------')
    print(f'{my_env.args}')

    # ----------------------------------- Dataset Init -----------------------------------------------------------

    my_loader = Loader4MM(my_env)

    tool.cprint('Init Dataset')

    # ----------------------------------- Model Init -----------------------------------------------------------

    my_model = MOME_model(my_env, my_loader)
    if args.ckpt != None:
        my_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    tool.cprint('Init Model')

    # ----------------------------------- Session Init -----------------------------------------------------------

    my_session = Session(my_env, my_model, my_loader)
    tool.cprint('Init Session')

    # ---------------------------------------- Main -----------------------------------------------------------

    t = time.time()
    my_session.train(my_env.args.epoch)
    # my_session.save_memory()
    # my_env.close_env()
    train_seconds = time.time() - t
    tool.cprint(f'training stage cost time: {train_seconds}')
    if my_env.args.log:
        my_env.test_logger.info(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')

    tool.cprint(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')
    for top_k in args.topk:
        tool.cprint(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
        if my_env.args.log:
            my_env.test_logger.info(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')

    report = my_session.final_report(args.noise_rate, args.noise_type, args.noise_seed, args.gamma_grid)
    if args.gate_probe:
        report['gate_probe'] = my_session.gate_probe(args.noise_rate, args.noise_type, args.noise_seed)
    if 'gamma_selection' in report:
        args.vgr_gamma = report['gamma_selection']['selected']
    gamma_tag = 'sel' if args.gamma_grid else str(args.vgr_gamma)
    result_name = f'{args.mode}_{args.noise_type}_r{args.noise_rate:.2f}_g{gamma_tag}_seed{args.seed}'
    if args.suffix != 'default':
        result_name += f'_{args.suffix}'
    os.makedirs(args.results_dir, exist_ok=True)
    result_path = os.path.join(args.results_dir, result_name + '.json')
    with open(result_path, 'w') as f:
        json.dump({'args': vars(args), 'report': report, 'best_epoch': my_session.best_epoch,
                   'train_seconds': train_seconds}, f, indent=2)
    diag = my_session.diagnostics(args.noise_rate, args.noise_type, args.noise_seed)
    diag_dir = os.path.join(args.results_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    np.savez_compressed(os.path.join(diag_dir, result_name + '.npz'), users=diag['users'], items=diag['items'],
                        corrupted_items=diag['corrupted_items'], **{f'ranks_{k}': v for k, v in diag['ranks'].items()})
    tool.cprint(f'results written to {result_path}')
