"""Train the fixed train split and write a holdout summary.

This script keeps the old name for compatibility, but it no longer trains a
second "fold" model. The dataset uses split=train for fitting, split=val for
threshold calibration, and split=test for held-out evaluation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

from kvasir_triage.eval.crossval import write_cv_summary
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import cv_summary_root, logs_root, runs_root
from kvasir_triage.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging("train_cv", logs_root())

    cv_id = f"{Path(args.config).stem}_holdout_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dirs = []

    run_id = f"{cv_id}_fold0"
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        "--config", str(args.config),
        "--fold", "0",
        "--run-id", run_id,
    ]
    if args.max_epochs:
        cmd += ["--max-epochs", str(args.max_epochs)]

    log.info("Training split=train and validating on split=val ...")
    subprocess.run(cmd, check=True)
    run_dirs.append(runs_root() / run_id)

    log.info("Writing holdout summary ...")
    # load metrics from evaluate.py outputs if available, else train_result
    fold_metrics = []
    for d in run_dirs:
        tcal = d / "threshold_calibration.json"
        if tcal.exists():
            import json
            with open(tcal) as f:
                fold_metrics.append(json.load(f))
        else:
            # fallback to train result
            import json
            with open(d / "train_result.json") as f:
                fold_metrics.append(json.load(f))

    out_dir = cv_summary_root() / cv_id
    write_cv_summary(fold_metrics, out_dir)
    save_json(
        {"cv_id": cv_id, "summary_type": "holdout", "run_dirs": [str(d) for d in run_dirs]},
        out_dir / "cv_index.json",
    )
    log.info("Holdout summary written to %s", out_dir)


if __name__ == "__main__":
    main()
