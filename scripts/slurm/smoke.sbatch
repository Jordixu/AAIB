#!/bin/bash
#SBATCH --job-name=kvasir-smoke
#SBATCH -p express
#SBATCH --output=/home/%u/logs/smoke_%j.out
#SBATCH --error=/home/%u/logs/smoke_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# CPU smoke test on the express partition (testing queue, max 2h / 4 CPUs).
# Validates the training loop and data pipeline without occupying GPU resources.
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

python scripts/smoke_test.py --epochs 2
