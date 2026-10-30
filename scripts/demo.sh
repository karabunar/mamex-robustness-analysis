#!/usr/bin/env bash
# Short runnable demonstration (not the reported results): the reported tables are rebuilt from the saved results,
# then a 21-epoch baseline is trained on the real data, stress-tested with VGR, and repaired without training.
#   STEP 1 and 2 need no GPU and no data; STEP 3 needs the data (DATA_PATH) and a GPU is recommended.
set -euo pipefail
export PYTHONPATH=src
PY=${PY:-python}
OUT=${OUT:-demo_out}
DATA_PATH=${DATA_PATH:-processed_data_baby/cold}

echo "== STEP 1: rebuild results/tables.md and results/h12_h13.md from the saved result files"
$PY -m mamex_robustness.reporting.make_tables > /dev/null && $PY scripts/h12_h13_tables.py > /dev/null && git --no-pager diff --stat -- results/tables.md results/h12_h13.md | tail -1 || true
echo "   (no diff above means the saved tables are reproduced exactly)"

echo "== STEP 2: unit tests (tiny synthetic data)"
$PY -m pytest -q 2>&1 | tail -1 || true

[ -d "$DATA_PATH" ] || { echo "STEP 3 skipped: no data at $DATA_PATH (see README, section Data)"; exit 0; }
echo "== STEP 3: 21-epoch clean baseline with a checkpoint, then the same model under corrupted cold-item text, with VGR, and repaired"
$PY main.py --data_path "$DATA_PATH" --mode baseline --seed 2025 --epoch 21 --eva_interval 10 --save_final 1 --results_dir "$OUT" > "$OUT.log" 2>&1
$PY main.py --data_path "$DATA_PATH" --mode improved --seed 2025 --noise_rate 0.4 --epoch 21 --eva_interval 10 --results_dir "$OUT" >> "$OUT.log" 2>&1
$PY -m mamex_robustness.repair_cli --data_path "$DATA_PATH" --seed 2025 --noise_rate 0.4 --ckpt_dir "$OUT/checkpoints" --out_dir "$OUT/repair" >> "$OUT.log" 2>&1
$PY - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
r = json.load(open(f'{out}/repair/repair_hashtag_r0.40_seed2025.json'))['report']
print('Recall@20 on cold items, hashtag noise on 40% of them, 21-epoch model (a demonstration, not the reported numbers):')
print(f"  clean text, unrepaired : {r['clean']['unrepaired']['recall']:.4f}")
for name, label in (('unrepaired', 'noisy text, unrepaired'), ('vgr', 'noisy text, VGR'), ('m1', 'noisy text, repair M1'), ('m3', 'noisy text, repair M3')):
    print(f"  {label:23s}: {r['noisy'][name]['recall']:.4f}")
print(f'Full log: {out}.log')
PYEOF
