"""Evaluate trained model(s) on the held-out test set (fold=2).

Run after training and evaluate.py (threshold calibration must exist).
Uses the NPV-safe threshold calibrated on the validation fold — the test set
is never used for calibration.

Single model:
    python scripts/evaluate_test.py outputs/runs/myrun_fold0

Ensemble from a full CV run (recommended):
    python scripts/evaluate_test.py --cv-summary outputs/cv_summary/myrun
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from kvasir_triage.data.dataset import KvasirNPZShardDataset
from kvasir_triage.data.transforms import AlbumentationsWrapper, build_eval_transform
from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.models.registry import build_model
from kvasir_triage.paths import logs_root, manifest_path, norm_stats_path, outputs_root
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import load_json, save_json


def _load_run(run_dir: Path, device: torch.device, norm_stats: dict) -> tuple:
    """Load a checkpoint and return (model, t_safe, train_fold).

    Uses the t_safe threshold from threshold_calibration.json so the test set
    is never used to set the operating threshold.
    """
    cfg_path = run_dir / "config.snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.snapshot.yaml in {run_dir}")

    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"

    ckpt = load_checkpoint(ckpt_path)
    train_fold = cfg.get("fold", ckpt.get("cfg", {}).get("fold", 0))

    model = build_model(cfg["model"], cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    cal_path = run_dir / "threshold_calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"threshold_calibration.json not found in {run_dir}. "
            "Run evaluate.py on this run first."
        )
    cal = load_json(cal_path)
    t_safe = cal.get("t_safe")

    return model, t_safe, train_fold, cfg


def _run_inference(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    train_fold: int,
    norm_stats: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (logits, labels, subclasses) for the test set."""
    fold_key = f"fold_{train_fold}"
    stats = norm_stats.get(fold_key, norm_stats.get("all"))
    tf = AlbumentationsWrapper(build_eval_transform(stats["mean"], stats["std"]))
    ds = KvasirNPZShardDataset(test_df, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_logits, all_labels, all_subclasses = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs = imgs.to(device)
            with autocast(enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_subclasses.extend(meta["subclass"])

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_subclasses,
    )


def evaluate_test(run_dirs: list[Path], log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    norm_stats = load_json(norm_stats_path())

    mf = manifest_path()
    if not mf.exists():
        mf = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
    df = pd.read_csv(mf)
    test_df = df[df["fold"] == 2].reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError(
            "No fold=2 rows in manifest. "
            "Re-run preprocess.py with --test-fraction 0.2 to create the test set."
        )
    log.info("Test set: %d images from %d unique videos", len(test_df),
             test_df["stem"].str.split("_").str[0].nunique())

    all_probs: list[np.ndarray] = []
    t_safes: list[float] = []
    labels_np: np.ndarray | None = None
    subclasses: list[str] | None = None

    t0 = time.perf_counter()
    for run_dir in run_dirs:
        log.info("Loading run: %s", run_dir)
        model, t_safe, train_fold, cfg = _load_run(run_dir, device, norm_stats)
        logits, labels, subs = _run_inference(
            model, test_df, train_fold, norm_stats, device,
            batch_size=cfg.get("batch_size", 32),
            num_workers=0,
        )
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        all_probs.append(probs)
        if t_safe is not None:
            t_safes.append(t_safe)

        if labels_np is None:
            labels_np = labels
            subclasses = subs

    elapsed = time.perf_counter() - t0

    # Average probabilities across models (ensemble)
    ensemble_probs = np.mean(all_probs, axis=0)
    # Convert back to logit-like score for compute_metrics (expects raw scores, higher = more positive)
    ensemble_logits = np.log(np.clip(ensemble_probs, 1e-7, 1 - 1e-7) / (1 - np.clip(ensemble_probs, 1e-7, 1 - 1e-7)))

    metrics_default = compute_metrics(ensemble_logits, labels_np)
    subclass_recall = per_subclass_recall(ensemble_logits, labels_np, subclasses)

    # Apply val-calibrated threshold if available
    t_ensemble = float(np.mean(t_safes)) if t_safes else None
    metrics_calibrated: dict | None = None
    if t_ensemble is not None:
        preds_cal = (ensemble_probs >= t_ensemble).astype(int)
        tp = int(((preds_cal == 1) & (labels_np == 1)).sum())
        tn = int(((preds_cal == 0) & (labels_np == 0)).sum())
        fp = int(((preds_cal == 1) & (labels_np == 0)).sum())
        fn = int(((preds_cal == 0) & (labels_np == 1)).sum())
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        metrics_calibrated = {
            "threshold": t_ensemble,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "npv": round(npv, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    speed = {
        "images": len(test_df),
        "total_seconds": round(elapsed, 2),
        "images_per_sec": round(len(test_df) / elapsed, 1),
    }

    result = {
        "n_images": len(test_df),
        "n_models_ensembled": len(run_dirs),
        "metrics_at_0.5": metrics_default,
        "metrics_at_calibrated_threshold": metrics_calibrated,
        "subclass_recall": subclass_recall,
        "speed": speed,
        "run_dirs": [str(d) for d in run_dirs],
    }

    log.info("=== Test set results ===")
    log.info("PR-AUC:  %.4f", metrics_default["pr_auc"])
    log.info("MCC:     %.4f", metrics_default["mcc"])
    log.info("NPV@0.5: %.4f", metrics_default["npv"])
    if metrics_calibrated:
        log.info("NPV@t_safe(%.3f): %.4f  (TP=%d FN=%d)",
                 t_ensemble, metrics_calibrated["npv"],
                 metrics_calibrated["tp"], metrics_calibrated["fn"])
    log.info("Subclass recall: %s", subclass_recall)

    return result


def _find_run_dirs_from_cv(cv_summary_dir: Path) -> list[Path]:
    index_path = cv_summary_dir / "cv_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"cv_index.json not found in {cv_summary_dir}")
    index = load_json(index_path)
    run_dirs = [Path(d) for d in index["run_dirs"]]
    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"Run dirs listed in cv_index.json not found: {missing}")
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description="Evaluate on held-out test set (fold=2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("run_dir", type=Path, nargs="?",
                       help="Single run directory (outputs/runs/myrun_fold0)")
    group.add_argument("--cv-summary", type=Path, metavar="CV_DIR",
                       help="CV summary directory to ensemble both fold models")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for test_results.json (default: outputs/test_evaluation/)")
    args = parser.parse_args()

    log = setup_logging("evaluate_test", logs_root())

    if args.cv_summary:
        run_dirs = _find_run_dirs_from_cv(args.cv_summary)
        log.info("Ensembling %d models from CV run: %s", len(run_dirs), args.cv_summary)
    else:
        run_dirs = [args.run_dir]

    result = evaluate_test(run_dirs, log)

    if args.out is None:
        out_dir = outputs_root() / "test_evaluation"
    else:
        out_dir = args.out if args.out.is_dir() else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_results.json"
    save_json(result, out_path)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
