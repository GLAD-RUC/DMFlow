#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dmflow}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Project root: ${PROJECT_ROOT} ==="
echo "=== Conda environment: ${ENV_NAME} ==="

# -------------------------------------------------------
# 1. Initialize & update git submodules
# -------------------------------------------------------
echo "=== [1/5] Initializing git submodules ==="
cd "${PROJECT_ROOT}"
git submodule update --init --recursive

# -------------------------------------------------------
# 2. Fix riemannian-fm: create manifm/__init__.py
# -------------------------------------------------------
echo "=== [2/5] Fixing riemannian-fm: creating manifm/__init__.py ==="
MANIFM_INIT="${PROJECT_ROOT}/remote/riemannian-fm/manifm/__init__.py"
if [ ! -f "${MANIFM_INIT}" ]; then
    touch "${MANIFM_INIT}"
    echo "Created ${MANIFM_INIT}"
else
    echo "${MANIFM_INIT} already exists, skipping."
fi

# -------------------------------------------------------
# 3. Create conda environment
# -------------------------------------------------------
echo "=== [3/5] Creating conda environment ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, updating with new dependencies..."
    conda env update -y -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml" --prune
else
    conda env create -y -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml"
    echo "Conda environment '${ENV_NAME}' created."
fi

# -------------------------------------------------------
# 4. Fix libtiff: Pillow needs libtiff.so.5 but conda provides .so.6
# -------------------------------------------------------
echo "=== [4/5] Fixing libtiff symlink ==="
CONDA_ENV_LIB="$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')/lib"
LIBTIFF_SO6="${CONDA_ENV_LIB}/libtiff.so.6"
LIBTIFF_SO5="${CONDA_ENV_LIB}/libtiff.so.5"
if [ -f "${LIBTIFF_SO6}" ] && [ ! -L "${LIBTIFF_SO5}" ] && [ ! -f "${LIBTIFF_SO5}" ]; then
    ln -sf "${LIBTIFF_SO6}" "${LIBTIFF_SO5}"
    echo "Created symlink ${LIBTIFF_SO5} -> ${LIBTIFF_SO6}"
else
    echo "libtiff.so.5 already exists or libtiff.so.6 not found, skipping."
fi

# -------------------------------------------------------
# 5. Prepare .env files
# -------------------------------------------------------
echo "=== [5/5] Preparing .env files ==="
cd "${PROJECT_ROOT}"
cat > "${PROJECT_ROOT}/.env" <<EOL
PROJECT_ROOT="${PROJECT_ROOT}"
HYDRA_JOBS="${PROJECT_ROOT}"
WANDB_DIR="${PROJECT_ROOT}"
EOL

cp "${PROJECT_ROOT}/.env" "${PROJECT_ROOT}/remote/DiffCSP-official/.env"
cp "${PROJECT_ROOT}/.env" "${PROJECT_ROOT}/remote/cdvae/.env"

echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
