"""H24: choose VGR's gamma by validation NDCG@10 of VGR minus the shuffled-cosine control (rule B) and compare with gamma 1 on the test set.

usage: PYTHONPATH=src python scripts/h24_gamma_rule.py   (writes results/h24.md and results/h24.json)
"""
import json

import numpy as np
import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main():
    rows = []
    for seed in (2025, 2026, 2027):
        args = tool.apply_mode_defaults(parse_args(['--mode', 'baseline', '--seed', str(seed), '--noise_rate', '0.4', '--use_gpu', '0',
                                                    '--data_path', 'processed_data_baby/cold']))
        args.noise_seed = seed
        env = Env(args)
        data = Loader4MM(env)
        model = MOME_model(env, data)
        model.load_state_dict(torch.load(f'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed{seed}.pt', map_location='cpu'))
        session = Session(env, model, data)
        session.model.eval()
        val = data.noisy_text_feat(0.4, 'hashtag', Session.VAL_NOISE_SEED_OFFSET + seed)
        test = data.noisy_text_feat(0.4, 'hashtag', seed)

        def score(mode, text, gamma, control, use_vgr=True):
            model.vgr_gamma, model.vgr_control = gamma, control
            _, recall, ndcg, _ = session.test(mode=mode, top_list=[10, 20], noisy_text=text, use_vgr=use_vgr)
            return float(recall[20]), float(ndcg[10])
        base_test = score('test', test, 1.0, 'cosine', use_vgr=False)[0]
        per_gamma = {}
        for g in GRID:
            val_cos, val_shuf = score('val', val, g, 'cosine')[1], score('val', val, g, 'shuffled')[1]
            test_cos, test_shuf = score('test', test, g, 'cosine')[0], score('test', test, g, 'shuffled')[0]
            per_gamma[g] = {'val_vgr': val_cos, 'val_shuffled': val_shuf, 'val_diff': val_cos - val_shuf,
                            'test_vgr': test_cos, 'test_shuffled': test_shuf, 'test_diff': test_cos - test_shuf, 'test_gain': test_cos - base_test}
        released = max([g for g in GRID if g >= 1.0], key=lambda g: (per_gamma[g]['val_vgr'], -g))
        rule_b = max(GRID, key=lambda g: (per_gamma[g]['val_diff'], -g))
        rows.append({'seed': seed, 'unrepaired': base_test, 'released_gamma': released, 'rule_b_gamma': rule_b, 'per_gamma': per_gamma})
        print(seed, released, rule_b, flush=True)
    with open('results/h24.json', 'w') as f:
        json.dump(rows, f, indent=2)
    out = ['## H24: gamma chosen by the released rule and by rule B (validation VGR minus shuffled control), hashtag 0.4, test Recall@20', '',
           '| seed | released gamma | rule B gamma | total gain, released | total gain, B | VGR minus shuffled, released | VGR minus shuffled, B | VGR minus shuffled at gamma 1 |',
           '|---|---|---|---|---|---|---|---|']
    for r in rows:
        a, b, one = (r['per_gamma'][g] for g in (r['released_gamma'], r['rule_b_gamma'], 1.0))
        out.append(f"| {r['seed']} | {r['released_gamma']:g} | {r['rule_b_gamma']:g} | {a['test_gain']:+.4f} | {b['test_gain']:+.4f} | {a['test_diff']:+.4f} | {b['test_diff']:+.4f} | {one['test_diff']:+.4f} |")
    open('results/h24.md', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
