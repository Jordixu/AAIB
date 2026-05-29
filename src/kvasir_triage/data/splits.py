"""Subclass-stratified video-level train/val/test split assignment."""
from __future__ import annotations

import hashlib
import warnings

import pandas as pd

SPLIT_ID_TO_NAME = {0: "train", 1: "val", 2: "test"}
SPLIT_NAME_TO_ID = {name: idx for idx, name in SPLIT_ID_TO_NAME.items()}
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


def infer_video_id(stem: str) -> str:
    """Best-effort video group from filename stem (everything before first '_')."""
    return stem.split("_")[0]


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with named train/val/test splits derived from legacy fold ids.

    The project historically used ``fold`` for fixed train/validation/test
    partitions. Keeping the numeric column preserves compatibility with old
    manifests and notebooks, while ``split`` is the unambiguous source of truth.
    """
    out = df.copy()
    if "split" not in out.columns:
        if "fold" not in out.columns:
            raise ValueError("Manifest must contain either a 'split' or legacy 'fold' column.")
        unknown = sorted(set(out["fold"].dropna().astype(int)) - set(SPLIT_ID_TO_NAME))
        if unknown:
            raise ValueError(f"Unknown legacy fold ids: {unknown}")
        out["split"] = out["fold"].astype(int).map(SPLIT_ID_TO_NAME)
    if "fold" not in out.columns:
        unknown = sorted(set(out["split"].dropna()) - set(SPLIT_NAME_TO_ID))
        if unknown:
            raise ValueError(f"Unknown split names: {unknown}")
        out["fold"] = out["split"].map(SPLIT_NAME_TO_ID).astype(int)
    else:
        expected = out["fold"].astype(int).map(SPLIT_ID_TO_NAME)
        mismatch = out["split"].notna() & expected.ne(out["split"])
        if mismatch.any():
            sample = out.loc[mismatch, ["fold", "split"]].head().to_dict("records")
            raise ValueError(f"Inconsistent legacy fold ids and split names: {sample}")
    return out


def build_subclass_stratified_splits(
    image_df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each image to split=train, val, or test.

    Video-level assignment using multi-label iterative stratification over the
    full set of subclasses present in each video — not just the primary subclass.
    This guarantees proportional representation of every subclass across all splits
    (subject to having enough videos; classes with only 1–2 videos cannot appear
    in all three splits simultaneously).

    Algorithm: Sechidis et al. 2011 iterative stratification, adapted for three
    splits. Videos are processed rarest-subclass-first; each is assigned to the
    split with the highest remaining desired image count for the triggering subclass,
    then the desired counts for all subclasses present in that video are
    decremented for the chosen split.

    Parameters
    ----------
    image_df : DataFrame with columns: stem, subclass, binary_label
    val_fraction : proportion reserved for validation
    test_fraction : proportion reserved for held-out test
    seed : integer seed for deterministic hash-based ordering
    """
    train_fraction = 1.0 - val_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    df = image_df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    # Count matrix: video × subclass. The assignment unit is the whole video, but
    # using image counts for desired quotas keeps image-level train/val/test
    # proportions closer to 70/15/15 than video-presence counts alone.
    vid_sc_counts = (
        df.groupby(["_vid", "subclass"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    vid_sc = (
        vid_sc_counts
        .gt(0)
        .astype(int)
    )
    vids = vid_sc.index.tolist()
    subclasses = vid_sc.columns.tolist()

    target = {0: train_fraction, 1: val_fraction, 2: test_fraction}

    # n_vids_with_sc[sc] = number of videos that contain subclass sc
    n_vids_with_sc = vid_sc.sum(axis=0).to_dict()
    n_images_with_sc = vid_sc_counts.sum(axis=0).to_dict()

    # Desired number of images per subclass and split, updated as videos are assigned.
    desired: dict[int, dict[str, float]] = {
        split_id: {sc: n_images_with_sc[sc] * frac for sc in subclasses}
        for split_id, frac in target.items()
    }

    def _hash(vid: str) -> str:
        return hashlib.md5(f"{seed}:{vid}".encode()).hexdigest()

    # Process subclasses rarest-first (fewest videos containing that class)
    subclasses_by_rarity = sorted(subclasses, key=lambda sc: n_vids_with_sc[sc])

    vid_fold: dict[str, int] = {}
    assigned: set[str] = set()

    for sc in subclasses_by_rarity:
        sc_vids = sorted(
            [v for v in vids if vid_sc.loc[v, sc] and v not in assigned],
            key=_hash,
        )
        for vid in sc_vids:
            present_splits = {
                vid_fold[v]
                for v in assigned
                if vid_sc.loc[v, sc]
            }
            missing_splits = [f for f in target if f not in present_splits]
            best_split = max(missing_splits or desired, key=lambda f: desired[f][sc])
            vid_fold[vid] = best_split
            assigned.add(vid)
            # Decrement desired for every subclass this video covers
            for sc2 in subclasses:
                n_sc2 = int(vid_sc_counts.loc[vid, sc2])
                if n_sc2:
                    desired[best_split][sc2] = max(0.0, desired[best_split][sc2] - n_sc2)

    # Assign any remaining videos proportionally by image count.
    video_sizes = df.groupby("_vid").size().to_dict()
    fold_counts = {
        f: sum(video_sizes[v] for v, assigned_f in vid_fold.items() if assigned_f == f)
        for f in target
    }
    for vid in sorted(set(vids) - assigned, key=_hash):
        best_fold = min(fold_counts, key=lambda f: fold_counts[f] / target[f])
        vid_fold[vid] = best_fold
        fold_counts[best_fold] += video_sizes[vid]

    df["fold"] = df["_vid"].map(vid_fold)

    n_missing = df["fold"].isna().sum()
    if n_missing > 0:
        warnings.warn(f"{n_missing} images have no fold assignment and will be dropped.")
        df = df[df["fold"].notna()].copy()

    df["fold"] = df["fold"].astype(int)
    df["split"] = df["fold"].map(SPLIT_ID_TO_NAME)
    df.drop(columns=["_vid"], inplace=True)
    return df
