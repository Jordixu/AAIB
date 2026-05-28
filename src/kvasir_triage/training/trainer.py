"""Main training loop with AMP, grad clipping, logging, checkpointing."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.eval.suspicious_report import save_suspicious_frame_report
from kvasir_triage.losses.focal import BinaryFocalLoss
from kvasir_triage.models.registry import freeze_backbone, get_param_groups, unfreeze_backbone
from kvasir_triage.training.checkpoint import save_best_and_last
from kvasir_triage.training.early_stop import EarlyStopping
from kvasir_triage.training.optim import make_optimizer, make_scheduler


def _meta_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_val_outputs(model, loader, device):
    model.eval()
    all_logits, all_labels, all_subclasses, all_metas = [], [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_subclasses.extend(meta["subclass"])
            batch_size = len(labels)
            for idx in range(batch_size):
                all_metas.append({
                    key: _meta_scalar(values[idx])
                    for key, values in meta.items()
                })
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
        all_subclasses,
        all_metas,
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
    model_name = cfg["model"]

    criterion = BinaryFocalLoss(
        gamma=cfg.get("focal_gamma", 2.0),
        alpha=cfg.get("focal_alpha", 0.25),
    )
    loss_name = "focal"
    scaler = GradScaler(enabled=device.type == "cuda")
    stopper = EarlyStopping(patience=cfg.get("early_stop_patience", 10))

    start_epoch = 0
    best_metric = -float("inf")
    ckpt = None

    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -float("inf"))

    lr = cfg.get("lr", 3e-4)
    freeze_epochs = cfg.get("freeze_epochs", 0)
    backbone_lr = cfg.get("backbone_lr")
    phase2_backbone_lr = backbone_lr if backbone_lr is not None else lr / 10

    if freeze_epochs > 0 and start_epoch < freeze_epochs:
        freeze_backbone(model_name, model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer(model, cfg, param_groups=[{"params": trainable_params, "lr": lr}])
        phase2_started = False
        log.info("Phase 1: training head only for %d epoch(s)", freeze_epochs)
    else:
        unfreeze_backbone(model_name, model)
        if freeze_epochs > 0 or backbone_lr is not None:
            param_groups = get_param_groups(model_name, model, phase2_backbone_lr, lr)
            optimizer = make_optimizer(model, cfg, param_groups=param_groups)
        else:
            optimizer = make_optimizer(model, cfg)
        phase2_started = freeze_epochs > 0
        if freeze_epochs > 0:
            log.info(
                "Phase 2: fine-tuning full model (backbone_lr=%.2e, head_lr=%.2e)",
                phase2_backbone_lr,
                lr,
            )

    scheduler = make_scheduler(optimizer, cfg)

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "stopper" in ckpt:
            stopper.load_state_dict(ckpt["stopper"])

    max_epochs = cfg.get("max_epochs", 50)
    max_batches = cfg.get("max_batches", None)  # for quick tests
    grad_clip = cfg.get("grad_clip", 1.0)
    log_interval = cfg.get("log_interval", 20)  # batches between progress lines

    n_batches = len(train_loader)
    log.info("Starting training: epochs %d-%d on %s", start_epoch, max_epochs - 1, device)

    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        if epoch == freeze_epochs and freeze_epochs > 0 and not phase2_started:
            unfreeze_backbone(model_name, model)
            param_groups = get_param_groups(model_name, model, phase2_backbone_lr, lr)
            optimizer = make_optimizer(model, cfg, param_groups=param_groups)
            scheduler = make_scheduler(optimizer, cfg)
            phase2_started = True
            log.info(
                "Phase 2: fine-tuning full model (backbone_lr=%.2e, head_lr=%.2e)",
                phase2_backbone_lr,
                lr,
            )

        phase = 1 if freeze_epochs > 0 and epoch < freeze_epochs else 2
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
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
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
                    "loss_fn": loss_name,
                    "phase": phase,
                })

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
                log.info(
                    "Epoch %d  phase=%d  [%d/%d]  loss=%.4f  grad_norm=%.3f  lr=%.2e",
                    epoch, phase, batch_idx + 1, n_batches,
                    loss.item(), float(grad_norm),
                    optimizer.param_groups[0]["lr"],
                )

        train_time = time.perf_counter() - t0
        train_loss = epoch_loss / max(n_samples, 1)
        samples_per_sec = n_samples / train_time

        # validation
        logits_np, labels_np, subclasses, val_metas = _collect_val_outputs(model, val_loader, device)
        metrics = compute_metrics(logits_np, labels_np)
        subclass_recall = per_subclass_recall(logits_np, labels_np, subclasses)

        val_prauc = metrics["pr_auc"]
        scheduler.step(val_prauc)

        is_best = val_prauc > best_metric
        if is_best:
            best_metric = val_prauc

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stopper": stopper.state_dict(),
            "best_metric": best_metric,
            "cfg": cfg,
        }
        save_best_and_last(state, run_dir, is_best)
        if is_best:
            np.savez(
                run_dir / "val_preds_best.npz",
                logits=logits_np,
                labels=labels_np,
                subclasses=np.array(subclasses),
                rel_paths=np.array([m.get("rel_path", "") for m in val_metas]),
                original_paths=np.array([m.get("path", "") for m in val_metas]),
                filenames=np.array([m.get("filename", "") for m in val_metas]),
                stems=np.array([m.get("stem", "") for m in val_metas]),
            )
            report_paths = save_suspicious_frame_report(
                logits_np,
                labels_np,
                val_metas,
                run_dir,
                threshold=cfg.get("suspicious_report_threshold", 0.5),
            )
            log.info("Suspicious-frame report saved: %s", report_paths)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pr_auc": val_prauc,
            "val_mcc": metrics["mcc"],
            "val_npv": metrics.get("npv", None),
            "train_time_s": train_time,
            "samples_per_sec": samples_per_sec,
            "loss_fn": loss_name,
            "phase": phase,
            **{f"recall_{k}": v for k, v in subclass_recall.items()},
        }
        for lg in loggers:
            lg.log_epoch(log_entry)

        log.info(
            "Epoch %d done  train_loss=%.4f  val_pr_auc=%.4f  val_mcc=%.4f"
            "  time=%.1fs  %.1f smp/s%s  loss_fn=%s  phase=%d",
            epoch, train_loss, val_prauc, metrics["mcc"],
            train_time, samples_per_sec,
            "  [best]" if is_best else "",
            loss_name,
            phase,
        )

        if stopper.step(val_prauc):
            log.info("Early stopping at epoch %d (best val_pr_auc=%.4f)", epoch, best_metric)
            break

    return {"best_pr_auc": best_metric, "epochs_run": epoch + 1 if max_epochs > 0 else 0}
