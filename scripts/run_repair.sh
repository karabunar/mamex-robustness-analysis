#!/usr/bin/env bash
# Training-free text repair (H8 to H10) on the saved clean-trained checkpoints in results/ckpt291/checkpoints.
# H8/H9 (hashtag 0.4) is the default; GRID=1 adds the generalisation grid (H10, setting 1) and MISMATCH=1 the
# validation-mismatch check (H10, setting 2). No training happens here.
set -euo pipefail
export PYTHONPATH=src
DATA_PATH=${DATA_PATH:-processed_data_baby/cold}
SEEDS=${SEEDS:-"2025 2026 2027"}
PY=${PY:-python}

run() {  # noise_type noise_rate [val_noise_type val_noise_rate]
  for seed in $SEEDS; do
    $PY -m mamex_robustness.repair_cli --data_path "$DATA_PATH" --seed "$seed" --noise_type "$1" --noise_rate "$2" \
      ${3:+--val_noise_type "$3"} ${4:+--val_noise_rate "$4"} &
  done
  wait
}

if [ -z "${GRID:-}" ] && [ -z "${MISMATCH:-}" ]; then run hashtag 0.4; fi
if [ -n "${GRID:-}" ]; then
  for type in hashtag shuffle gaussian; do for rate in 0.2 0.4 0.6; do run $type $rate; done; done
fi
if [ -n "${MISMATCH:-}" ]; then
  run hashtag 0.2 hashtag 0.4; run hashtag 0.6 hashtag 0.4; run shuffle 0.4 hashtag 0.4; run gaussian 0.4 hashtag 0.4
fi
