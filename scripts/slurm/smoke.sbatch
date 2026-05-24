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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$PROJECT_ROOT/.venv/bin/activate"

python scripts/smoke_test.py --epochs 2
