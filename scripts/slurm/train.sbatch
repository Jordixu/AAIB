#!/bin/bash
#SBATCH --job-name=kvasir-train
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/train_%j.out
#SBATCH --error=/home/%u/logs/train_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Usage: sbatch scripts/slurm/train.sbatch --config configs/densenet161_pretrained.yaml --fold 0
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_NO_VERSION_CHECK=1
export TORCH_HOME="$SLURM_SUBMIT_DIR/weights"

python scripts/train.py "$@"
