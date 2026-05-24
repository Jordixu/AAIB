#!/bin/bash
#SBATCH --job-name=kvasir-preprocess
#SBATCH -p std
#SBATCH --output=/home/%u/logs/preprocess_%j.out
#SBATCH --error=/home/%u/logs/preprocess_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# CPU-only preprocessing on the standard partition.
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$PROJECT_ROOT/.venv/bin/activate"

python scripts/preprocess.py "$@"
