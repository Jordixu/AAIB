#!/bin/bash
#SBATCH --job-name=kvasir-cv
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/cv_%A_%a.out
#SBATCH --error=/home/%u/logs/cv_%A_%a.err
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

# Array job: fold 0 and fold 1 run in parallel on separate GPU nodes.
# Usage: sbatch --export=CONFIG=configs/densenet161_pretrained.yaml scripts/slurm/train_cv.sbatch
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$PROJECT_ROOT/.venv/bin/activate"

FOLD=$SLURM_ARRAY_TASK_ID
CONFIG="${CONFIG:-configs/densenet161_pretrained.yaml}"

python scripts/train.py --config "$CONFIG" --fold "$FOLD"
