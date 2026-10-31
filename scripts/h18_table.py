"""H18: the repair evaluation on a second dataset, from <repair dir>/repair_hashtag_r0.40_seed*.json.

usage: python scripts/h18_table.py <repair dir> <dataset name>   (writes results/h18_<dataset name>.md)
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    directory, name = sys.argv[1], sys.argv[2]
    reports = {}
    for path in sorted(glob.glob(os.path.join(directory, 'repair_hashtag_r0.40_seed*.json'))):
        with open(path) as f:
            data = json.load(f)
        reports[data['args']['seed']] = data['report']
    seeds = sorted(reports)
    rec = lambda s, block, m: reports[s][block][m]['recall']
    gain = lambda s, m, base='unrepaired': rec(s, 'noisy', m) - rec(s, 'noisy', base)
    cols = [('VGR', lambda s: gain(s, 'vgr')), ('M1', lambda s: gain(s, 'm1')), ('M3', lambda s: gain(s, 'm3')),
            ('M3 then VGR', lambda s: rec(s, 'noisy_plus_vgr', 'm3') - rec(s, 'noisy', 'unrepaired')),
            ('M3 - random control', lambda s: gain(s, 'm3', 'm3_random')), ('M3 - permuted control', lambda s: gain(s, 'm3', 'm3_shuffled')),
            ('M3 - VGR', lambda s: gain(s, 'm3', 'vgr')), ('M3 - M1', lambda s: gain(s, 'm3', 'm1'))]
    out = [f'## H18: {name}, Recall@20, hashtag noise on 40% of the cold items ({len(seeds)} seeds)', '',
           '| seed | clean | noisy, unrepaired | ' + ' | '.join(c for c, _ in cols) + ' | AUROC of the cosine |', '|---|---|---|' + '---|' * (len(cols) + 1)]
    for s in seeds:
        out.append(f"| {s} | {rec(s, 'clean', 'unrepaired'):.4f} | {rec(s, 'noisy', 'unrepaired'):.4f} | " + ' | '.join(f'{f(s):+.4f}' for _, f in cols)
                   + f" | {reports[s]['auroc']['vgr_cosine']:.3f} |")
    out.append(f"| mean | {st.mean(rec(s, 'clean', 'unrepaired') for s in seeds):.4f} | {st.mean(rec(s, 'noisy', 'unrepaired') for s in seeds):.4f} | "
               + ' | '.join(f'{st.mean(f(s) for s in seeds):+.4f}' for _, f in cols) + ' | |')
    out += ['', 'Seeds above zero: ' + ', '.join(f'{c} {sum(f(s) > 0 for s in seeds)}/{len(seeds)}' for c, f in cols),
            'Loss from the corruption (clean minus noisy, unrepaired): ' + ', '.join(f"{rec(s, 'clean', 'unrepaired') - rec(s, 'noisy', 'unrepaired'):+.4f}" for s in seeds),
            'Selected on noisy validation data (M3): ' + '; '.join(f"seed {s} {reports[s]['selection']['m3']['param']}" for s in seeds)]
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', f'h18_{name}.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
