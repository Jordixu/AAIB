#!/bin/bash
#SBATCH --job-name=kvasir-gradcam
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/gradcam_%j.out
#SBATCH --error=/home/%u/logs/gradcam_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Grad-CAM audit on selected validation outcomes.
# Usage: sbatch scripts/slurm/gradcam_audit.sbatch outputs/runs/myrun_fold0 --max-samples 20
#        sbatch scripts/slurm/gradcam_audit.sbatch outputs/runs/myrun_fold0 --outcome false-negative
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

python scripts/gradcam_audit.py "$@"
