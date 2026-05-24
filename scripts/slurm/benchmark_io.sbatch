#!/bin/bash
#SBATCH --job-name=kvasir-bench
#SBATCH -p std
#SBATCH --output=/home/%u/logs/benchmark_%j.out
#SBATCH --error=/home/%u/logs/benchmark_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

# Measures DataLoader throughput across in_memory / shard modes and worker counts.
# Run on std with 32 cores / 64 GB to mirror GPU node training conditions without
# occupying a GPU slot. Results are written to outputs/io_benchmark.csv.
#
# Usage: sbatch scripts/slurm/benchmark_io.sbatch [--fold 0]
# Requires: bash scripts/slurm/setup_cluster.sh run once beforehand.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi
source "$PROJECT_ROOT/.venv/bin/activate"

FOLD="${FOLD:-0}"

python scripts/benchmark_io.py --fold "$FOLD"

echo ""
echo "Results written to outputs/io_benchmark.csv"
echo "Update num_workers and in_memory in your config YAML before training."
