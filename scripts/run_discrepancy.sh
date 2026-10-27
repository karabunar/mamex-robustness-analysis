#!/usr/bin/env bash
# Diagnostic runs for the Rec@10 gap to the published baseline (see docs/hypotheses.md).
# Files carry a suffix, so make_tables.py (default suffix only) ignores them.
#   STREAM=A ./scripts/run_discrepancy.sh    # D1: baseline seeds 2028-2034
#   STREAM=B ./scripts/run_discrepancy.sh    # D2: align_coeff 1.28 (3 seeds) and 12.8 (seed 2025)
#   STREAM=C ./scripts/run_discrepancy.sh    # D2: align_coeff 12.8 (seeds 2026, 2027) and 128 (3 seeds)
#   DRY=1 STREAM=A ./scripts/run_discrepancy.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repository root, wherever the script is called from
export PYTHONPATH=${PYTHONPATH:-src}
DATA_PATH=${DATA_PATH:-processed_data_baby/cold}

run() {
    echo "+ python main.py $*"
    if [ -z "${DRY:-}" ]; then python main.py --data_path "$DATA_PATH" "$@"; fi
}
align() {  # align <coeff> <seed>
    run --mode baseline --seed "$2" --align_coeff "$1" --suffix "align$1"
}

case "${STREAM:?set STREAM=A, B or C}" in
    A) for s in 2028 2029 2030 2031 2032 2033 2034; do run --mode baseline --seed "$s" --suffix seedext; done ;;
    B) for s in 2025 2026 2027; do align 1.28 "$s"; done; align 12.8 2025 ;;
    C) align 12.8 2026; align 12.8 2027; for s in 2025 2026 2027; do align 128 "$s"; done ;;
    *) echo "unknown stream" >&2; exit 1 ;;
esac
