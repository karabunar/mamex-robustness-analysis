"""Tables for H12 (which epoch rule reproduces the published Rec@10) and H13 (what drives the gate on cold items).

Reads results/curves291/curves/*.json (retrained clean baselines run with --curve 1) and results/gate_cause/*.json,
writes results/h12_h13.md. usage: python scripts/h12_h13_tables.py
"""
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SEEDS = (2025, 2026, 2027)
PUBLISHED_REC10 = 0.0512
RULES = [('validation NDCG@10 (the released rule)', lambda p: p['val_ndcg']['10']),
         ('validation Recall@10', lambda p: p['val_recall']['10']),
         ('validation NDCG@20', lambda p: p['val_ndcg']['20']),
         ('validation Recall@20', lambda p: p['val_recall']['20']),
         ('test Recall@10 (oracle, uses the test set)', lambda p: p['test_recall']['10']),
         ('test NDCG@10 (oracle, uses the test set)', lambda p: p['test_ndcg']['10']),
         ('test Recall@20 (oracle, uses the test set)', lambda p: p['test_recall']['20'])]


def load(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def epoch_table():
    curves = {s: load('results', 'curves291', 'curves', f'baseline_hashtag_r0.00_g8.0_seed{s}.json')['curve'] for s in SEEDS}
    out = ['## H12: test Recall@10 of the clean baseline under different epoch rules (seeds 2025 to 2027)', '',
           f'Published Rec@10: {PUBLISHED_REC10}. Ties go to the earlier epoch. Evaluation epochs: '
           + ', '.join(f'{len(curves[s])} (seed {s})' for s in SEEDS) + '.', '',
           '| Epoch chosen by | epochs | test Rec@10 per seed | mean | gap to published |', '|---|---|---|---|---|']
    for name, key in RULES:
        picks = [max(curves[s], key=lambda p: (key(p), -p['epoch'])) for s in SEEDS]
        values = [p['test_recall']['10'] for p in picks]
        out.append(f"| {name} | {', '.join(str(p['epoch']) for p in picks)} | {', '.join(f'{v:.4f}' for v in values)} "
                   f"| {st.mean(values):.4f} | {PUBLISHED_REC10 - st.mean(values):+.4f} |")
    out += ['', 'Test Recall@10 across the evaluation epochs of each seed:', '', '| seed | max | median | last |', '|---|---|---|---|']
    for s in SEEDS:
        r = [p['test_recall']['10'] for p in curves[s]]
        out.append(f'| {s} | {max(r):.4f} | {st.median(r):.4f} | {r[-1]:.4f} |')
    return out


def gate_table():
    keys = [('zero_gap', 'text minus image logit, zero input'), ('w_t_warm', 'w_t warm'), ('w_t_cold', 'w_t cold'),
            ('w_t_cold_scaled_both', 'cold, both outputs scaled to warm norm'),
            ('w_t_cold_replaced_by_warm', 'cold, both outputs from a random warm item'),
            ('w_t_cold_image_replaced_by_warm', 'cold, image output from a warm item (exploratory)'),
            ('w_t_cold_text_replaced_by_warm', 'cold, text output from a warm item (exploratory)')]
    runs = [(f'{kind} {s}', load('results', 'gate_cause', f'{tag}_seed{s}.json')['report'])
            for kind, tag in (('clean-trained', 'clean'), ('augmented', 'aug')) for s in (2025, 2026)]
    out = ['## H13: interventions on the gate input of the cold items (clean text)', '',
           '| quantity | ' + ' | '.join(name for name, _ in runs) + ' |', '|---|' + '---|' * len(runs)]
    for key, label in keys:
        out.append(f'| {label} | ' + ' | '.join(f'{r[key]:.3f}' for _, r in runs) + ' |')
    out.append('| mean expert-output norm, cold / warm (image, text) | ' + ' | '.join(
        f"{r['cold_norm']['v'] / r['warm_norm']['v']:.2f}, {r['cold_norm']['t'] / r['warm_norm']['t']:.2f}" for _, r in runs) + ' |')
    return out


if __name__ == '__main__':
    text = '\n'.join(epoch_table() + [''] + gate_table()) + '\n'
    with open(os.path.join(ROOT, 'results', 'h12_h13.md'), 'w') as f:
        f.write(text)
    print(text)
