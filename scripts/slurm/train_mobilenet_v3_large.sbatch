#!/bin/bash
#SBATCH --job-name=kvasir-mnv3
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/mnv3_%j.out
#SBATCH --error=/home/%u/logs/mnv3_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Holdout training job for MobileNetV3-Large.
# Usage: sbatch scripts/slurm/train_mobilenet_v3_large.sbatch
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

python scripts/train.py --config configs/mobilenet_v3_large.yaml --fold 0
