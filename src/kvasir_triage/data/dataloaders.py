"""Build train/val DataLoaders from a manifest and config."""
from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader

from kvasir_triage.data.dataset import KvasirInMemoryDataset, KvasirNPZShardDataset
from kvasir_triage.data.sampler import make_weighted_sampler
from kvasir_triage.data.splits import TRAIN_SPLIT, VAL_SPLIT, add_split_column
from kvasir_triage.data.transforms import AlbumentationsWrapper, make_transforms_for_fold
from kvasir_triage.utils import load_json
from kvasir_triage.paths import norm_stats_path


def make_dataloaders(
    manifest: pd.DataFrame,
    train_fold: int,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for the fixed train/val/test split.

    cfg keys used:
      in_memory (bool), batch_size (int), num_workers (int),
      persistent_workers (bool), lru_size (int, shard mode only),
      pin_memory (bool), sampler_balance ("subclass" | "binary" | "none"),
      sampler_weight_power (float), sampler_num_samples_multiplier (float).
    """
    if int(train_fold) != 0:
        raise ValueError(
            "Refusing to train on fold 1. The manifest uses a fixed split: "
            "fold 0 / split='train' for training, fold 1 / split='val' for validation, "
            "fold 2 / split='test' for held-out testing."
        )
    manifest = add_split_column(manifest)
    norm_stats = load_json(norm_stats_path())

    train_tf, eval_tf = make_transforms_for_fold(norm_stats, train_fold)
    train_tf = AlbumentationsWrapper(train_tf)
    eval_tf = AlbumentationsWrapper(eval_tf)

    train_df = manifest[manifest["split"] == TRAIN_SPLIT].reset_index(drop=True)
    val_df = manifest[manifest["split"] == VAL_SPLIT].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No training rows found in split='train' / fold=0.")
    if val_df.empty:
        raise ValueError("No validation rows found in split='val' / fold=1.")
    train_paths = set(train_df["rel_path"]) if "rel_path" in train_df else set(train_df.index)
    val_paths = set(val_df["rel_path"]) if "rel_path" in val_df else set(val_df.index)
    overlap = train_paths & val_paths
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"Train/validation leakage detected for {len(overlap)} images: {sample}")

    in_memory = cfg.get("in_memory", False)
    DatasetCls = KvasirInMemoryDataset if in_memory else KvasirNPZShardDataset

    ds_kwargs = {} if in_memory else {"lru_size": cfg.get("lru_size", 8)}
    train_ds = DatasetCls(train_df, transform=train_tf, **ds_kwargs)
    val_ds = DatasetCls(val_df, transform=eval_tf, **ds_kwargs)

    sampler_balance = cfg.get("sampler_balance", "subclass")
    if sampler_balance == "subclass":
        sampler = make_weighted_sampler(
            train_df["subclass"].tolist(),
            weight_power=cfg.get("sampler_weight_power", 1.0),
            num_samples_multiplier=cfg.get("sampler_num_samples_multiplier", 1.0),
        )
    elif sampler_balance == "binary":
        sampler = make_weighted_sampler(
            train_df["binary_label"].tolist(),
            weight_power=cfg.get("sampler_weight_power", 1.0),
            num_samples_multiplier=cfg.get("sampler_num_samples_multiplier", 1.0),
        )
    elif sampler_balance in {None, "none"}:
        sampler = None
    else:
        raise ValueError(
            "sampler_balance must be one of: 'subclass', 'binary', 'none'. "
            f"Got {sampler_balance!r}."
        )

    num_workers = cfg.get("num_workers", 4)
    persistent = cfg.get("persistent_workers", True) and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=cfg.get("pin_memory", True),
    )
    return train_loader, val_loader
