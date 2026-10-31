"""H16: M3 against M1 and the unrepaired model over ten seeds (hashtag 0.4, Recall@20), with a paired bootstrap over seeds.

usage: python scripts/h16_ten_seeds.py   (reads results/repair/repair_hashtag_r0.40_seed*.json; writes results/h16.md)
"""
import json
import os
import random
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIRST = (2025, 2026, 2027, 2028, 2029)


def main():
    reports = {}
    for seed in range(2025, 2035):
        path = os.path.join(ROOT, 'results', 'repair', f'repair_hashtag_r0.40_seed{seed}.json')
        if os.path.exists(path):
            with open(path) as f:
                reports[seed] = json.load(f)['report']
    seeds = sorted(reports)
    get = lambda s, block, m: reports[s][block][m]['recall']
    diff = lambda s, a, b: get(s, 'noisy', a) - get(s, 'noisy', b)
    out = [f'## H16: Recall@20 gains over the unrepaired model, hashtag 0.4, {len(seeds)} seeds', '',
           '| seed | unrepaired | VGR | M1 | M3 | M3 then VGR | M3 - M1 | M3 beats both controls |', '|---|---|---|---|---|---|---|---|']
    for s in seeds:
        base = get(s, 'noisy', 'unrepaired')
        beats = diff(s, 'm3', 'm3_random') > 0 and diff(s, 'm3', 'm3_shuffled') > 0
        out.append(f"| {s} | {base:.4f} | {diff(s, 'vgr', 'unrepaired'):+.4f} | {diff(s, 'm1', 'unrepaired'):+.4f} | {diff(s, 'm3', 'unrepaired'):+.4f} | "
                   f"{get(s, 'noisy_plus_vgr', 'm3') - base:+.4f} | {diff(s, 'm3', 'm1'):+.4f} | {'yes' if beats else 'no'} |")
    mean = lambda f: st.mean(f(s) for s in seeds)
    out.append(f"| mean | {mean(lambda s: get(s, 'noisy', 'unrepaired')):.4f} | {mean(lambda s: diff(s, 'vgr', 'unrepaired')):+.4f} | {mean(lambda s: diff(s, 'm1', 'unrepaired')):+.4f} | "
               f"{mean(lambda s: diff(s, 'm3', 'unrepaired')):+.4f} | {mean(lambda s: get(s, 'noisy_plus_vgr', 'm3') - get(s, 'noisy', 'unrepaired')):+.4f} | {mean(lambda s: diff(s, 'm3', 'm1')):+.4f} | |")
    d = [diff(s, 'm3', 'm1') for s in seeds]
    rng = random.Random(0)
    boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    new = [s for s in seeds if s not in FIRST]
    out += ['', f'M3 minus M1 over {len(seeds)} seeds: mean {st.mean(d):+.4f}, sample std {st.stdev(d):.4f}, paired bootstrap 95% interval (10000 resamples over seeds) '
            f'[{boots[249]:+.4f}, {boots[9749]:+.4f}]; M3 at least M1 in {sum(x >= 0 for x in d)} of {len(d)} seeds.']
    if new:
        out.append(f"New seeds {new}: M3 beats the unrepaired model in {sum(diff(s, 'm3', 'unrepaired') > 0 for s in new)} of {len(new)}; "
                   f"M3 at least M1 in {sum(diff(s, 'm3', 'm1') >= 0 for s in new)} of {len(new)}.")
    text = '\n'.join(out) + '\n'
    with open(os.path.join(ROOT, 'results', 'h16.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
