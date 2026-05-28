#!/bin/bash
#SBATCH --job-name=kvasir-eval
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/eval_%j.out
#SBATCH --error=/home/%u/logs/eval_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Threshold calibration on a single validation fold.
# Usage: sbatch scripts/slurm/evaluate.sbatch outputs/runs/myrun_fold0
#        sbatch scripts/slurm/evaluate.sbatch outputs/runs/myrun_fold1
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

python scripts/evaluate.py "$@"
