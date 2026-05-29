#!/bin/bash
# Submit a clean preprocessing job followed by a short holdout training simulation.
#
# Usage:
#   bash scripts/slurm/submit_rare_oversampling_sim.sh
#   CONFIG=configs/efficientnet_v2_s.yaml MAX_EPOCHS=1 MAX_BATCHES=20 bash scripts/slurm/submit_rare_oversampling_sim.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/smoke.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
MAX_BATCHES="${MAX_BATCHES:-5}"

preprocess_job=$(sbatch \
  --parsable \
  --export=ALL,CLEAN_PREPROCESSED=1 \
  scripts/slurm/preprocess.sbatch)
echo "Submitted preprocessing job: ${preprocess_job}"

train_job=$(sbatch \
  --parsable \
  --dependency=afterok:"${preprocess_job}" \
  --export=ALL,CONFIG="${CONFIG}",MAX_EPOCHS="${MAX_EPOCHS}",MAX_BATCHES="${MAX_BATCHES}" \
  scripts/slurm/train_cv.sbatch)
echo "Submitted rare-oversampling simulation job: ${train_job}"
echo "Config: ${CONFIG}  MAX_EPOCHS=${MAX_EPOCHS}  MAX_BATCHES=${MAX_BATCHES}"
