"""Main training loop with AMP, grad clipping, logging, checkpointing."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.training.checkpoint import save_best_and_last
from kvasir_triage.training.conditional_loss import ConditionalLoss
from kvasir_triage.training.early_stop import EarlyStopping
from kvasir_triage.training.optim import make_optimizer, make_scheduler


def _collect_val_outputs(model, loader, device):
    model.eval()
    all_logits, all_labels, all_subclasses = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_subclasses.extend(meta["subclass"])
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
        all_subclasses,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    run_dir: Path,
    loggers: list,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Full training loop. Returns final metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = ConditionalLoss(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=device.type == "cuda")
    stopper = EarlyStopping(patience=cfg.get("early_stop_patience", 10))

    start_epoch = 0
    best_metric = -float("inf")

    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -float("inf"))

    max_epochs = cfg.get("max_epochs", 50)
    max_batches = cfg.get("max_batches", None)  # for quick tests
    grad_clip = cfg.get("grad_clip", 1.0)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.perf_counter()

        for batch_idx, (imgs, labels, _meta) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs

            for lg in loggers:
                lg.log_step({
                    "epoch": epoch,
                    "step": batch_idx,
                    "loss": loss.item(),
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss_fn": criterion.name,
                })

        train_time = time.perf_counter() - t0
        train_loss = epoch_loss / max(n_samples, 1)
        samples_per_sec = n_samples / train_time

        # validation
        logits_np, labels_np, subclasses = _collect_val_outputs(model, val_loader, device)
        metrics = compute_metrics(logits_np, labels_np)
        subclass_recall = per_subclass_recall(logits_np, labels_np, subclasses)

        val_prauc = metrics["pr_auc"]
        scheduler.step(val_prauc)

        switched = criterion.maybe_switch(subclass_recall)

        is_best = val_prauc > best_metric
        if is_best:
            best_metric = val_prauc

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "cfg": cfg,
        }
        save_best_and_last(state, run_dir, is_best)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pr_auc": val_prauc,
            "val_mcc": metrics["mcc"],
            "val_npv": metrics.get("npv", None),
            "train_time_s": train_time,
            "samples_per_sec": samples_per_sec,
            "loss_fn": criterion.name,
            "loss_switched": switched,
            **{f"recall_{k}": v for k, v in subclass_recall.items()},
        }
        for lg in loggers:
            lg.log_epoch(log_entry)

        if stopper.step(val_prauc):
            break

    return {"best_pr_auc": best_metric, "epochs_run": epoch + 1}
