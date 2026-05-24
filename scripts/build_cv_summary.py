"""Assemble a cv_summary directory from two independently trained fold run dirs.

Use this after training with the SLURM array job (train_cv.sbatch), which trains
each fold independently and does not produce a cv_summary automatically.

Usage:
    python scripts/build_cv_summary.py \\
        outputs/runs/myrun_fold0 \\
        outputs/runs/myrun_fold1 \\
        --cv-id myrun_cv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvasir_triage.eval.crossval import write_cv_summary
from kvasir_triage.paths import cv_summary_root
from kvasir_triage.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=2, type=Path, metavar="RUN_DIR",
                        help="Path to fold 0 and fold 1 run directories")
    parser.add_argument("--cv-id", type=str, default=None,
                        help="Name for the cv_summary directory (default: derived from run dirs)")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            print(f"Error: run dir not found: {d}", file=sys.stderr)
            sys.exit(1)

    fold_metrics = []
    for d in run_dirs:
        cal = d / "threshold_calibration.json"
        fallback = d / "train_result.json"
        if cal.exists():
            src = cal
        elif fallback.exists():
            print(f"Warning: no threshold_calibration.json in {d}, using train_result.json")
            print("  Run evaluate.py first for proper calibrated thresholds.")
            src = fallback
        else:
            print(f"Error: neither threshold_calibration.json nor train_result.json found in {d}",
                  file=sys.stderr)
            sys.exit(1)
        with open(src) as f:
            fold_metrics.append(json.load(f))

    cv_id = args.cv_id or f"{run_dirs[0].name}_plus_{run_dirs[1].name}"
    out_dir = cv_summary_root() / cv_id
    write_cv_summary(fold_metrics, out_dir)
    save_json(
        {"cv_id": cv_id, "run_dirs": [str(d) for d in run_dirs]},
        out_dir / "cv_index.json",
    )
    print(f"CV summary written to {out_dir}")


if __name__ == "__main__":
    main()
