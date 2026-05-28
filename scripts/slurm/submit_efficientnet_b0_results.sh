#!/bin/bash
# Submit EfficientNet-B0 two-fold training and validation evaluation jobs.
#
# Usage:
#   bash scripts/slurm/submit_efficientnet_b0_results.sh
#   RUN_ID=efficientnet_b0_trial MAX_EPOCHS=5 MAX_BATCHES=100 bash scripts/slurm/submit_efficientnet_b0_results.sh

set -euo pipefail

RUN_ID="${RUN_ID:-efficientnet_b0_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/efficientnet_b0.yaml}"

TRAIN_EXPORT="ALL,CONFIG=${CONFIG},RUN_ID=${RUN_ID}"
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  TRAIN_EXPORT="${TRAIN_EXPORT},MAX_EPOCHS=${MAX_EPOCHS}"
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  TRAIN_EXPORT="${TRAIN_EXPORT},MAX_BATCHES=${MAX_BATCHES}"
fi

train_job=$(sbatch \
  --parsable \
  --export="${TRAIN_EXPORT}" \
  scripts/slurm/train_cv.sbatch)
echo "Submitted EfficientNet-B0 CV training array: ${train_job}"

eval0_job=$(sbatch \
  --parsable \
  --dependency=afterok:"${train_job}" \
  scripts/slurm/evaluate.sbatch "outputs/runs/${RUN_ID}_fold0")
echo "Submitted fold 0 evaluation: ${eval0_job}"

eval1_job=$(sbatch \
  --parsable \
  --dependency=afterok:"${train_job}" \
  scripts/slurm/evaluate.sbatch "outputs/runs/${RUN_ID}_fold1")
echo "Submitted fold 1 evaluation: ${eval1_job}"

echo "RUN_ID=${RUN_ID}"
echo "When both eval jobs finish, inspect:"
echo "  outputs/runs/${RUN_ID}_fold0/threshold_calibration.json"
echo "  outputs/runs/${RUN_ID}_fold1/threshold_calibration.json"
echo "  outputs/runs/${RUN_ID}_fold0/eval_predictions_val.csv"
echo "  outputs/runs/${RUN_ID}_fold1/eval_predictions_val.csv"
