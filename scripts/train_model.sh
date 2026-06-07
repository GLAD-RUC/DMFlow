#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_model.sh <csp|dng> <dataset> [run_name] [hydra overrides...]

Examples:
  bash scripts/train_model.sh csp cod_sd_20
  bash scripts/train_model.sh csp cod_sd_20 csp_cod20_seed1
  GPUS=0,1 bash scripts/train_model.sh dng cod_spd_20
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

TASK="$1"
DATASET="$2"
shift 2

RUN_NAME="${RUN_NAME:-dmflow_${DATASET}_${TASK}}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != +* && "$1" != -* && "$1" != ~* ]]; then
    RUN_NAME="$1"
    shift
fi

case "${TASK}" in
    csp)
        MODEL="disorder_csp"
        ;;
    dng)
        MODEL="disorder_dng"
        ;;
    *)
        usage
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-False}"
CKPT_PATH="${CKPT_PATH:-}"

overrides=(
    "data=${DATASET}"
    "model=${MODEL}"
    "logging.wandb.name=${RUN_NAME}"
    "optim.optimizer.lr=\${data.train_model.${TASK}.lr}"
    "train.pl_trainer.gpus=\${data.train_model.${TASK}.num_gpus}"
    "data.datamodule.batch_size.train=\${data.train_model.${TASK}.batch_size}"
    "data.datamodule.batch_size.val=\${data.train_model.${TASK}.batch_size}"
    "model.emb_pd_info=\${data.train_model.${TASK}.emb_pd_info}"
    "model.manifold_getter.pd_frac_coords_manifold=\${data.train_model.${TASK}.pd_frac_coords_manifold}"
    "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
    "ckpt_path=${CKPT_PATH}"
)

if [[ -n "${GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    n_gpu="$(awk -F',' '{print NF}' <<< "${GPUS}")"
    overrides+=("train.pl_trainer.gpus=${n_gpu}")
fi

if [[ -n "${LR:-}" ]]; then
    overrides+=("optim.optimizer.lr=${LR}")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
    overrides+=(
        "data.datamodule.batch_size.train=${BATCH_SIZE}"
        "data.datamodule.batch_size.val=${BATCH_SIZE}"
    )
fi

if [[ -n "${EPOCHS:-}" ]]; then
    overrides+=("data.train_max_epochs=${EPOCHS}")
fi

echo "TASK=${TASK}" >&2
echo "DATA=${DATASET}" >&2
echo "MODEL=${MODEL}" >&2
echo "RUN_NAME=${RUN_NAME}" >&2

python -m dmflow.run "${overrides[@]}" "$@"
