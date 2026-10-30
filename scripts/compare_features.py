"""Compare the distributed Baby features with the LATTICE authors' published ones (arrays only; nothing in the download is executed).

usage: python -I scripts/compare_features.py <dir with image_feat.npy and text_feat.npy from LATTICE's baby folder> [processed_data_baby/cold]
"""
import sys

import numpy as np

lattice, ours = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'processed_data_baby/cold')
for name in ('image_feat', 'text_feat'):
    a = np.load(f'{lattice}/{name}.npy', allow_pickle=False)
    b = np.load(f'{ours}/{name}.npy', allow_pickle=False)
    same = a.shape == b.shape
    diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if same else float('nan')
    print(f'{name}: lattice {a.shape} {a.dtype}, ours {b.shape} {b.dtype}, same shape {same}, max abs difference {diff}')
