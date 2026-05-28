"""Small test to see if everything is working correctly
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from kvasir_triage.data.transforms import AlbumentationsWrapper, build_eval_transform, build_train_transform
from kvasir_triage.eval.metrics import compute_metrics
from kvasir_triage.losses.focal import BinaryFocalLoss
from kvasir_triage.models.tiny import build_tiny
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, runs_root
from kvasir_triage.seeding import set_all_seeds
from kvasir_triage.tracking.jsonl_logger import JsonlLogger
from kvasir_triage.training.checkpoint import save_best_and_last
from kvasir_triage.training.early_stop import EarlyStopping
from kvasir_triage.training.optim import make_optimizer, make_scheduler
from kvasir_triage.utils import save_json


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_synthetic_loader(n: int, label_frac: float = 0.5, batch_size: int = 16):
    """Random uint8 images, 336x336x3, stratified labels."""
    imgs = np.random.randint(0, 255, (n, 336, 336, 3), dtype=np.uint8)
    labels = np.array([1 if i < int(n * label_frac) else 0 for i in range(n)], dtype=np.int64)
    np.random.shuffle(labels)

    tf = AlbumentationsWrapper(build_eval_transform(IMAGENET_MEAN, IMAGENET_STD))

    tensors = torch.stack([tf(img) for img in imgs])
    label_t = torch.from_numpy(labels)
    ds = TensorDataset(tensors, label_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def run_smoke(epochs: int = 2, n_train: int = 160, n_val: int = 40, log=None) -> dict:
    set_all_seeds(42)

    run_dir = runs_root() / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_tiny()
    device = torch.device("cpu")

    cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "early_stop_patience": 999,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
    }
    criterion = BinaryFocalLoss(gamma=cfg["focal_gamma"], alpha=cfg["focal_alpha"])
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    logger = JsonlLogger(run_dir)

    train_loader = make_synthetic_loader(n_train)
    val_loader = make_synthetic_loader(n_val)

    best_prauc = -1.0
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                all_logits.append(model(imgs).squeeze(1))
                all_labels.append(labels)
        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        metrics = compute_metrics(logits_np, labels_np)
        val_prauc = metrics["pr_auc"]

        scheduler.step(val_prauc)
        is_best = val_prauc > best_prauc
        if is_best:
            best_prauc = val_prauc
        save_best_and_last({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, run_dir, is_best)
        logger.log_epoch({"epoch": epoch, "val_pr_auc": val_prauc, **metrics})
        msg = f"Epoch {epoch}: val_pr_auc={val_prauc:.4f}"
        if log:
            log.info(msg)
        else:
            print(msg)

    result = {"best_pr_auc": best_prauc, "epochs": epochs}
    save_json(result, run_dir / "smoke_result.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=160)
    parser.add_argument("--n-val", type=int, default=40)
    args = parser.parse_args()

    log = setup_logging("smoke_test", logs_root())
    log.info("Running smoke test (synthetic data, tiny model, CPU) ...")
    result = run_smoke(args.epochs, args.n_train, args.n_val, log=log)

    log.info("Smoke test complete. Best PR-AUC: %.4f", result["best_pr_auc"])
    if result["best_pr_auc"] <= 0.5:
        log.warning("PR-AUC <= 0.5 - gradients may not be flowing.")
    else:
        log.info("PASS: PR-AUC > 0.5, gradients flowing.")


if __name__ == "__main__":
    main()
