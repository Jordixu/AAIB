#!/bin/bash
#SBATCH --job-name=kvasir-preprocess
#SBATCH -p std
#SBATCH --output=/home/%u/logs/preprocess_%j.out
#SBATCH --error=/home/%u/logs/preprocess_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# CPU-only preprocessing on the standard partition.
# Usage: sbatch --export=LABELLED_IMAGES=/path/to/labelled_images scripts/slurm/preprocess.sbatch
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
if [[ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]]; then
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_NO_VERSION_CHECK=1

if [[ "${CLEAN_PREPROCESSED:-0}" == "1" ]]; then
  echo "Removing existing outputs/preprocessed before preprocessing ..."
  rm -rf outputs/preprocessed
fi

EXTRA_ARGS=()
if [[ -n "${LABELLED_IMAGES:-}" ]]; then
  EXTRA_ARGS+=(--labelled-images "${LABELLED_IMAGES}")
fi

python scripts/preprocess.py "${EXTRA_ARGS[@]}" "$@"
