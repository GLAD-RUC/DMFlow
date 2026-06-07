#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/generation.sh <csp|dng> /path/to/checkpoint.ckpt [subdir] [extra evaluate.py args...]

Examples:
  bash scripts/generation.sh csp /path/to/checkpoint.ckpt csp_eval
  GPUS=0,1 bash scripts/generation.sh dng /path/to/checkpoint.ckpt gen_eval
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

TASK="$1"
CKPT="$2"
shift 2

SUBDIR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    SUBDIR="$1"
    shift
fi

SLOPE="${SLOPE:-20.0}"

if [[ -n "${GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

case "${TASK}" in
    csp)
        SUBDIR="${SUBDIR:-csp_eval}"
        STAGE="${STAGE:-test}"
        BATCH_SIZE="${BATCH_SIZE:-256}"
        python -m dmflow.evaluate reconstruct "${CKPT}" \
            --subdir "${SUBDIR}" \
            --inference_anneal_slope "${SLOPE}" \
            --stage "${STAGE}" \
            --batch_size "${BATCH_SIZE}" \
            "$@"
        ;;
    dng)
        SUBDIR="${SUBDIR:-gen_eval}"
        BATCH_SIZE="${BATCH_SIZE:-800}"
        NUM_SAMPLES="${NUM_SAMPLES:-10000}"
        python -m dmflow.evaluate generate "${CKPT}" \
            --subdir "${SUBDIR}" \
            --inference_anneal_slope "${SLOPE}" \
            --batch_size "${BATCH_SIZE}" \
            --num_samples "${NUM_SAMPLES}" \
            "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
