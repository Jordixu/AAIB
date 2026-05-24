#!/bin/bash
#SBATCH --job-name=kvasir-cv
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/cv_%A_%a.out
#SBATCH --error=/home/%u/logs/cv_%A_%a.err
#SBATCH --array=0-1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Array job: fold 0 and fold 1 run in parallel on separate GPU nodes.
# Usage: sbatch --export=CONFIG=configs/densenet161_pretrained.yaml scripts/slurm/train_cv.sbatch
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

FOLD=$SLURM_ARRAY_TASK_ID
CONFIG="${CONFIG:-configs/densenet161_pretrained.yaml}"

python scripts/train.py --config "$CONFIG" --fold "$FOLD"
