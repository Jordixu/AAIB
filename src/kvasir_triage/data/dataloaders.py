"""Build train/val DataLoaders from a manifest and config."""
from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader

from kvasir_triage.data.dataset import KvasirInMemoryDataset, KvasirNPZShardDataset
from kvasir_triage.data.sampler import make_weighted_sampler
from kvasir_triage.data.transforms import AlbumentationsWrapper, make_transforms_for_fold
from kvasir_triage.utils import load_json
from kvasir_triage.paths import norm_stats_path


def make_dataloaders(
    manifest: pd.DataFrame,
    train_fold: int,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader).

    cfg keys used:
      in_memory (bool), batch_size (int), num_workers (int),
      persistent_workers (bool), lru_size (int, shard mode only),
      pin_memory (bool).
    """
    norm_stats = load_json(norm_stats_path())

    train_tf, eval_tf = make_transforms_for_fold(norm_stats, train_fold)
    train_tf = AlbumentationsWrapper(train_tf)
    eval_tf = AlbumentationsWrapper(eval_tf)

    val_fold = 1 - train_fold
    train_df = manifest[manifest["fold"] == train_fold].reset_index(drop=True)
    val_df = manifest[manifest["fold"] == val_fold].reset_index(drop=True)

    in_memory = cfg.get("in_memory", False)
    DatasetCls = KvasirInMemoryDataset if in_memory else KvasirNPZShardDataset

    ds_kwargs = {} if in_memory else {"lru_size": cfg.get("lru_size", 8)}
    train_ds = DatasetCls(train_df, transform=train_tf, **ds_kwargs)
    val_ds = DatasetCls(val_df, transform=eval_tf, **ds_kwargs)

    sampler = make_weighted_sampler(train_df["binary_label"].tolist())

    num_workers = cfg.get("num_workers", 4)
    persistent = cfg.get("persistent_workers", True) and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        sampler=sampler,
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
