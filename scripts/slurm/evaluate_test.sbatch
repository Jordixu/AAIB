#!/bin/bash
#SBATCH --job-name=kvasir-eval-test
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/eval_test_%j.out
#SBATCH --error=/home/%u/logs/eval_test_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Held-out test set evaluation.
# Requires threshold_calibration.json in each fold run directory (run evaluate.sbatch first).
# Usage: sbatch scripts/slurm/evaluate_test.sbatch outputs/runs/myrun_fold0 outputs/runs/myrun_fold1
#        sbatch scripts/slurm/evaluate_test.sbatch outputs/runs/myrun_fold0  (single model)
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

python scripts/evaluate_test.py "$@"
