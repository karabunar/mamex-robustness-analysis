#!/usr/bin/env bash
# Runs the full experiment grid on the Baby dataset. Results land in results/*.json.
#   SEEDS="2025 2026 2027" ./scripts/run_experiments.sh          # main grid (3 seeds)
#   SWEEPS=1 SEEDS="2025" ./scripts/run_experiments.sh           # + noise-rate and gamma sweeps (1 seed)
#   SKIP_CORE=1 SWEEPS=1 SEEDS="2025" ./scripts/run_experiments.sh   # sweeps only, core runs already done
#   PARALLEL=1 SKIP_CORE=1 SWEEPS=1 ./scripts/run_experiments.sh     # the two sweep streams (noise rate, gamma) run side by side on one GPU
#   DRY=1 ./scripts/run_experiments.sh                           # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repository root, wherever the script is called from

SEEDS=${SEEDS:-"2025 2026 2027"}
RATE=${RATE:-0.4}
GAMMA_GRID=${GAMMA_GRID:-"1 2 4 8 16"}   # gamma is selected on noisy validation data
DATA_PATH=${DATA_PATH:-baby}
EXTRA=${EXTRA:-}          # e.g. EXTRA="--epoch 200" for a quick pass

run() {
    echo "+ python main.py $*"
    if [ -z "${DRY:-}" ]; then python main.py --data_path "$DATA_PATH" $EXTRA "$@"; fi
}

if [ -z "${SKIP_CORE:-}" ]; then
    for seed in $SEEDS; do
        run --mode baseline    --seed "$seed"
        run --mode stress_test --seed "$seed" --noise_rate "$RATE"
        run --mode improved    --seed "$seed" --noise_rate "$RATE" --gamma_grid $GAMMA_GRID
    done
fi

sweep_rates() {
    for r in 0.2 0.6 0.8; do
        run --mode stress_test --seed "$1" --noise_rate "$r"
        run --mode improved    --seed "$1" --noise_rate "$r" --gamma_grid $GAMMA_GRID
    done
}
sweep_gamma() {
    for g in 1 2 4 8 16; do   # manual gamma: sensitivity table only
        run --mode improved --seed "$1" --noise_rate "$RATE" --vgr_gamma "$g"
    done
}

if [ -n "${SWEEPS:-}" ]; then
    for seed in $SEEDS; do
        if [ -n "${PARALLEL:-}" ]; then
            sweep_rates "$seed" & sweep_gamma "$seed" & wait
        else
            sweep_rates "$seed"
            sweep_gamma "$seed"
        fi
    done
fi

[ -n "${DRY:-}" ] && exit 0
python -m mamex_robustness.reporting.plot_results --results_dir results --k 20 2>/dev/null \
    || PYTHONPATH=src python -m mamex_robustness.reporting.plot_results --results_dir results --k 20
