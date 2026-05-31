#!/bin/bash
# One-time cluster setup for Pirineus3 (run after uploading the project).
# Usage: bash scripts/slurm/setup_cluster.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$HOME/logs"

module load conda
conda activate aabi

echo "Creating .venv on top of aabi (inherits torch, numpy, pandas, ...)..."
python -m venv --system-site-packages "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "Installing albumentations and grad-cam (not in aabi)..."
pip install -q albumentations grad-cam

echo "Installing kvasir_triage package (no extra deps, aabi has them)..."
pip install -q --no-deps -e .

echo ""
python -c "
import torch, albumentations, kvasir_triage
print(f'torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')
print(f'albumentations {albumentations.__version__}')
print('kvasir_triage import OK')
"

echo ""
echo "Setup complete. Verify the GPU environment with:"
echo "  sbatch scripts/slurm/smoke.sbatch"
