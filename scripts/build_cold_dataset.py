"""Build a MAMEX-format cold-item dataset from the LATTICE authors' published files (H18).

The Baby data of MAMEX is exactly the union of LATTICE's 5-core train, validation and test pairs, split by item into warm items
(training file) and cold items (validation and test files). This script applies the same construction to another category with a
fixed seed: --cold_share of the items are cold, half validation and half test. It is our construction, not the authors' split.

usage: python scripts/build_cold_dataset.py <LATTICE category folder, e.g. .../sports> <output folder> [--seed 2025] [--cold_share 0.2]
The feature files are linked, not copied. No download is executed; only the .json and .npy files are read.
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('out')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--cold_share', type=float, default=0.2)
args = parser.parse_args()

pairs = set()
for part in ('train', 'val', 'test'):
    with open(os.path.join(args.source, '5-core', f'{part}.json')) as f:
        for user, items in json.load(f).items():
            pairs.update((int(user), int(item)) for item in items)
image = np.load(os.path.join(args.source, 'image_feat.npy'), mmap_mode='r')
text = np.load(os.path.join(args.source, 'text_feat.npy'), mmap_mode='r')
n_items = image.shape[0]
assert text.shape[0] == n_items and max(i for _, i in pairs) + 1 == n_items, 'item ids must be 0..n-1 and match the feature rows'

rng = np.random.default_rng(args.seed)
n_cold = int(round(args.cold_share * n_items))
while True:  # the loader sizes the item set from the training items, so the largest id must stay warm
    cold = np.sort(rng.choice(n_items, size=n_cold, replace=False))
    if n_items - 1 not in set(cold.tolist()):
        break
shuffled = rng.permutation(cold)
val_items, test_items = set(shuffled[: n_cold // 2].tolist()), set(shuffled[n_cold // 2:].tolist())
cold_set = val_items | test_items

files = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
for user, item in sorted(pairs):
    files['val' if item in val_items else 'test' if item in test_items else 'train'][user].append(item)
os.makedirs(args.out, exist_ok=True)
for name, table in files.items():
    with open(os.path.join(args.out, f'{name}.txt'), 'w') as f:
        for user in sorted(table):
            f.write(f'{user} ' + ' '.join(map(str, table[user])) + '\n')
np.save(os.path.join(args.out, 'cold_item_index.npy'), cold.astype(np.int64))
for name in ('image_feat', 'text_feat'):
    link = os.path.join(args.out, f'{name}.npy')
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(os.path.join(args.source, f'{name}.npy')), link)

stats = {'source': os.path.basename(os.path.abspath(args.source)), 'seed': args.seed, 'cold_share': args.cold_share, 'n_pairs': len(pairs),
         'n_users': len({u for u, _ in pairs}), 'n_items': n_items, 'n_cold': n_cold, 'n_val_items': len(val_items), 'n_test_items': len(test_items),
         'pairs': {k: sum(len(v) for v in t.values()) for k, t in files.items()},
         'sha256': {n: hashlib.sha256(open(os.path.join(args.out, n), 'rb').read()).hexdigest() for n in ('train.txt', 'val.txt', 'test.txt', 'cold_item_index.npy')}}
with open(os.path.join(args.out, 'build_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
