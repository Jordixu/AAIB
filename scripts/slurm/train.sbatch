#!/bin/bash
#SBATCH --job-name=kvasir-train
#SBATCH -p gpu
#SBATCH --output=/home/%u/logs/train_%j.out
#SBATCH --error=/home/%u/logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

# Usage: sbatch scripts/slurm/train.sbatch --config configs/densenet161_pretrained.yaml --fold 0
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$PROJECT_ROOT/.venv/bin/activate"

python scripts/train.py "$@"
