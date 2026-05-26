"""Load a run checkpoint, compute metrics + threshold calibration."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

from kvasir_triage.config import load_config
from kvasir_triage.data.dataloaders import make_dataloaders
from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.eval.threshold import find_safe_threshold
from kvasir_triage.models.registry import build_model
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, manifest_path, outputs_root
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = run_dir / "config.snapshot.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.snapshot.yaml or .json in {run_dir}")

    if cfg_path.suffix == ".json":
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    log = setup_logging("evaluate", logs_root())

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"

    ckpt = load_checkpoint(ckpt_path)
    fold = args.fold if args.fold is not None else ckpt.get("cfg", {}).get("fold", 0)

    model = build_model(cfg["model"], cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mf = manifest_path()
    if not mf.exists():
        mf = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
    df = pd.read_csv(mf)

    eval_cfg = {**cfg}
    if device.type == "cpu":
        eval_cfg["num_workers"] = 0
        eval_cfg["batch_size"] = min(eval_cfg.get("batch_size", 32), 4)
    _, val_loader = make_dataloaders(df, fold, eval_cfg)

    all_logits, all_labels, all_subclasses = [], [], []
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for imgs, labels, meta in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_subclasses.extend(meta["subclass"])
            n += len(imgs)

    elapsed = time.perf_counter() - t0
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    metrics = compute_metrics(logits_np, labels_np)
    subclass_recall = per_subclass_recall(logits_np, labels_np, all_subclasses)
    threshold_result = find_safe_threshold(logits_np, labels_np)

    speed = {
        "images": n,
        "total_seconds": elapsed,
        "images_per_sec": round(n / elapsed, 1),
        "median_ms_per_image": round(elapsed / n * 1000, 2),
    }

    save_json({**metrics, **threshold_result, "subclass_recall": subclass_recall}, run_dir / "threshold_calibration.json")
    save_json(speed, run_dir / "inference_speed.json")

    log.info("PR-AUC:  %.4f", metrics["pr_auc"])
    log.info("MCC:     %.4f", metrics["mcc"])
    log.info("NPV:     %.4f", metrics["npv"])
    log.info("t_safe:  %s", threshold_result.get("t_safe"))
    log.info("Speed:   %s img/s  (%s ms/img)", speed["images_per_sec"], speed["median_ms_per_image"])


if __name__ == "__main__":
    main()
