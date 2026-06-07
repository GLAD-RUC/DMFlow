#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/evaluation.sh /path/to/checkpoint.ckpt [subdir] [extra old_eval_metrics args...]

Examples:
  bash scripts/evaluation.sh /path/to/checkpoint.ckpt csp_eval
  bash scripts/evaluation.sh /path/to/checkpoint.ckpt gen_eval
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

CKPT="$1"
shift

SUBDIR=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    SUBDIR="$1"
    shift
fi

STAGE="${STAGE:-test}"
metrics_args=()
if [[ "${LOG_WANDB:-0}" != "1" ]]; then
    metrics_args+=(--do_not_log_wandb)
fi

python -m dmflow.evaluate consolidate "${CKPT}" --subdir "${SUBDIR}"
python -m dmflow.evaluate old_eval_metrics "${CKPT}" \
    --subdir "${SUBDIR}" \
    --stage "${STAGE}" \
    "${metrics_args[@]}" \
    "$@"
