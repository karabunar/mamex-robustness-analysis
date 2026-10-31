#!/usr/bin/env bash
# H18: a second dataset (Amazon Sports, from the LATTICE authors' files) built as a MAMEX-format cold split, then
# clean baselines with checkpoints and the repair evaluation on them. Run the build once, then the three trainings in parallel.
set -euo pipefail
export PYTHONPATH=src
PY=${PY:-python}
SRC=${SRC:?folder of the LATTICE Sports files (image_feat.npy, text_feat.npy, 5-core/)}
DATA=${DATA:-/media/baykar/disk27/mamex_data/sports/cold}
SEEDS=${SEEDS:-"2025 2026 2027"}
OUT=${OUT:-results/sports}
[ -f "$DATA/train.txt" ] || $PY scripts/build_cold_dataset.py "$SRC" "$DATA" --seed 2025
for seed in $SEEDS; do
  $PY main.py --data_path "$DATA" --mode baseline --seed "$seed" --save_final 1 --results_dir "$OUT" &
done
wait
for seed in $SEEDS; do
  $PY -m mamex_robustness.repair_cli --data_path "$DATA" --seed "$seed" --noise_rate 0.4 --ckpt_dir "$OUT/checkpoints" --out_dir "$OUT/repair"
done
$PY scripts/h18_table.py "$OUT/repair" sports
