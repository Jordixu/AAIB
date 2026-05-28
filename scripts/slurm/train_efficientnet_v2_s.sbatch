#!/bin/bash
#SBATCH --job-name=kvasir-cv-v2s
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/cv_%A_%a.out
#SBATCH --error=/home/%u/logs/cv_%A_%a.err
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# 2-fold CV array job for EfficientNet V2-S (~22M params).
# Usage: sbatch scripts/slurm/train_efficientnet_v2_s.sbatch
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
export TORCH_HOME="$SLURM_SUBMIT_DIR/weights"

python scripts/train.py --config configs/efficientnet_v2_s.yaml --fold "$SLURM_ARRAY_TASK_ID"
