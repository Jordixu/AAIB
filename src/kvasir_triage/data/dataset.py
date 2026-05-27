"""KvasirNPZShardDataset (LRU shard cache) and KvasirInMemoryDataset."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kvasir_triage.paths import outputs_root


def _meta_from_row(row) -> dict[str, Any]:
    keys = [
        "subclass",
        "rel_path",
        "path",
        "filename",
        "stem",
        "shard_rel_path",
        "local_idx",
        "bbox_norm_area",
    ]
    meta = {key: row.get(key, "") for key in keys}
    meta["fold"] = int(row["fold"])
    meta["local_idx"] = int(row["local_idx"])
    return meta


def _shard_rel_to_abs(rel: str) -> Path:
    """Resolve a manifest shard_rel_path to an absolute path via outputs_root."""
    return outputs_root() / rel


@lru_cache(maxsize=8)
def _load_shard(shard_path: str) -> dict[str, np.ndarray]:
    data = np.load(shard_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


class KvasirNPZShardDataset(Dataset):
    """Reads images from NPZ shards with a small LRU cache.

    Good for low-RAM machines. Cache of 8 shards covers ~4 GB.
    Due to random sampling the cache hit rate is low - prefer
    KvasirInMemoryDataset on cluster nodes with enough RAM.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform=None,
        lru_size: int = 8,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        # rebuild cache with the desired size
        global _load_shard
        _load_shard = lru_cache(maxsize=lru_size)(_load_shard.__wrapped__)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.manifest.iloc[idx]
        shard_path = str(_shard_rel_to_abs(row["shard_rel_path"]))
        shard = _load_shard(shard_path)
        local_idx = int(row["local_idx"])
        img = shard["images"][local_idx]  # uint8 HWC
        label = int(row["binary_label"])
        meta = _meta_from_row(row)
        if self.transform:
            img = self.transform(img)
        return img, label, meta


class KvasirInMemoryDataset(Dataset):
    """Loads all shards into RAM at construction (~10 GB per fold).

    Same __getitem__ contract as KvasirNPZShardDataset.
    Preferred on cluster nodes where data fits in memory.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        _load_shard.cache_clear()  # free shard LRU cache before bulk load
        self._images, self._labels, self._subclasses = self._load_all()

    def _load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.manifest)
        groups = self.manifest.groupby("shard_rel_path", sort=False)
        images_out = None
        lbl_arr = np.empty(n, dtype=np.int8)
        sub_arr = np.empty(n, dtype=object)

        for rel, grp in groups:
            shard_path = str(_shard_rel_to_abs(rel))
            shard = np.load(shard_path, allow_pickle=False)
            shard_images = shard["images"]
            if images_out is None:
                images_out = np.empty((n, *shard_images.shape[1:]), dtype=shard_images.dtype)
            dest = grp.index.to_numpy()
            local_idxs = grp["local_idx"].to_numpy(dtype=int)
            images_out[dest] = shard_images[local_idxs]
            lbl_arr[dest] = grp["binary_label"].to_numpy(dtype=np.int8)
            sub_arr[dest] = grp["subclass"].to_numpy()
            del shard, shard_images  # free shard before loading the next one

        return images_out, lbl_arr, np.array(sub_arr, dtype=str)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.manifest.iloc[idx]
        img = self._images[idx]
        label = int(self._labels[idx])
        meta = _meta_from_row(row)
        meta["subclass"] = str(self._subclasses[idx])
        if self.transform:
            img = self.transform(img)
        return img, label, meta
