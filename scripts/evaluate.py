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
from torch.utils.data import DataLoader
from tqdm import tqdm

from kvasir_triage.config import load_config
from kvasir_triage.data.dataloaders import make_dataloaders
from kvasir_triage.data.transforms import AlbumentationsWrapper, make_tta_transforms_for_fold
from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.eval.threshold import find_safe_threshold
from kvasir_triage.models.registry import build_model
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, manifest_path, outputs_root
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import save_json


def _meta_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _batched_meta_to_rows(meta: dict, batch_size: int) -> list[dict]:
    rows = []
    for idx in range(batch_size):
        rows.append({
            key: _meta_scalar(values[idx])
            for key, values in meta.items()
        })
    return rows


def _as_2d_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return logits[:, None]
    return logits


def _save_eval_predictions(
    run_dir: Path,
    split: str,
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    metas: list[dict],
    threshold_result: dict,
) -> tuple[Path, Path]:
    logits_2d = _as_2d_logits(logits_np)
    is_binary = logits_2d.shape[1] == 1

    if is_binary:
        probs = 1.0 / (1.0 + np.exp(-logits_2d[:, 0]))
        pred_labels = (probs >= 0.5).astype(np.int64)
        t_safe = threshold_result.get("t_safe")
        pred_labels_safe = (
            (probs >= t_safe).astype(np.int64) if t_safe is not None else np.full(len(probs), -1)
        )
    else:
        shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs_2d = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs = probs_2d.max(axis=1)
        pred_labels = logits_2d.argmax(axis=1).astype(np.int64)
        pred_labels_safe = np.full(len(pred_labels), -1)

    rows = []
    for idx, (label, pred, pred_safe, prob, meta) in enumerate(
        zip(labels_np, pred_labels, pred_labels_safe, probs, metas)
    ):
        row = {
            "index": idx,
            "split": split,
            "true_label": int(label),
            "predicted_label": int(pred),
            "predicted_label_t_safe": int(pred_safe),
            "probability": float(prob),
            "class": meta.get("subclass", ""),
            "subclass": meta.get("subclass", ""),
            "rel_path": meta.get("rel_path", ""),
            "path": meta.get("path", ""),
            "filename": meta.get("filename", ""),
            "stem": meta.get("stem", ""),
            "fold": meta.get("fold", ""),
            "shard_rel_path": meta.get("shard_rel_path", ""),
            "local_idx": meta.get("local_idx", ""),
            "bbox_norm_area": meta.get("bbox_norm_area", ""),
        }
        for output_idx in range(logits_2d.shape[1]):
            row[f"logit_{output_idx}"] = float(logits_2d[idx, output_idx])
        if is_binary:
            row["logit"] = float(logits_2d[idx, 0])
        rows.append(row)

    csv_path = run_dir / f"eval_predictions_{split}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    npz_path = run_dir / f"eval_predictions_{split}.npz"
    np.savez(
        npz_path,
        logits=logits_2d,
        true_labels=labels_np,
        predicted_labels=pred_labels,
        predicted_labels_t_safe=pred_labels_safe,
        probabilities=probs,
        classes=np.array([m.get("subclass", "") for m in metas]),
        subclasses=np.array([m.get("subclass", "") for m in metas]),
        rel_paths=np.array([m.get("rel_path", "") for m in metas]),
        filenames=np.array([m.get("filename", "") for m in metas]),
        stems=np.array([m.get("stem", "") for m in metas]),
    )
    return csv_path, npz_path


def _run_eval_pass(model, loader, device, desc: str):
    all_logits, all_labels, all_subclasses, all_metas = [], [], [], []
    n = 0
    with torch.no_grad():
        for imgs, labels, meta in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_subclasses.extend(meta["subclass"])
            all_metas.extend(_batched_meta_to_rows(meta, len(labels)))
            n += len(imgs)
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_subclasses,
        all_metas,
        n,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--tta", action="store_true", help="Average logits over deterministic TTA variants.")
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
    train_loader, val_loader = make_dataloaders(df, fold, eval_cfg)
    if args.split == "train":
        persistent = eval_cfg.get("persistent_workers", True) and eval_cfg.get("num_workers", 4) > 0
        eval_loader = DataLoader(
            train_loader.dataset,
            batch_size=eval_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=eval_cfg.get("num_workers", 4),
            persistent_workers=persistent,
            pin_memory=eval_cfg.get("pin_memory", True),
        )
    else:
        eval_loader = val_loader

    t0 = time.perf_counter()
    if args.tta:
        from kvasir_triage.utils import load_json
        from kvasir_triage.paths import norm_stats_path

        tta_transforms = make_tta_transforms_for_fold(load_json(norm_stats_path()), fold)
        log.info("TTA active: averaging logits over %d deterministic variants", len(tta_transforms))
        logits_runs = []
        labels_np = None
        all_subclasses = None
        all_metas = None
        n = 0
        original_transform = eval_loader.dataset.transform
        try:
            for idx, tta_tf in enumerate(tta_transforms):
                eval_loader.dataset.transform = AlbumentationsWrapper(tta_tf)
                raw_logits_np, labels_i, subclasses_i, metas_i, n_i = _run_eval_pass(
                    model,
                    eval_loader,
                    device,
                    desc=f"Evaluating {args.split} TTA {idx + 1}/{len(tta_transforms)}",
                )
                logits_runs.append(raw_logits_np)
                if labels_np is None:
                    labels_np = labels_i
                    all_subclasses = subclasses_i
                    all_metas = metas_i
                    n = n_i
        finally:
            eval_loader.dataset.transform = original_transform
        raw_logits_np = np.mean(np.stack(logits_runs, axis=0), axis=0)
    else:
        raw_logits_np, labels_np, all_subclasses, all_metas, n = _run_eval_pass(
            model,
            eval_loader,
            device,
            desc=f"Evaluating {args.split}",
        )

    elapsed = time.perf_counter() - t0
    logits_np = raw_logits_np.squeeze(1) if raw_logits_np.ndim == 2 and raw_logits_np.shape[1] == 1 else raw_logits_np

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
    csv_path, npz_path = _save_eval_predictions(
        run_dir,
        args.split,
        logits_np,
        labels_np,
        all_metas,
        threshold_result,
    )

    log.info("PR-AUC:  %.4f", metrics["pr_auc"])
    log.info("MCC:     %.4f", metrics["mcc"])
    log.info("NPV:     %.4f", metrics["npv"])
    log.info("t_safe:  %s", threshold_result.get("t_safe"))
    log.info("Speed:   %s img/s  (%s ms/img)", speed["images_per_sec"], speed["median_ms_per_image"])
    log.info("Predictions saved: %s", csv_path)
    log.info("Prediction arrays saved: %s", npz_path)


if __name__ == "__main__":
    main()
