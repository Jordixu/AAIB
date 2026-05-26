"""Subclass-stratified video-level split assignment."""
from __future__ import annotations

import hashlib
import warnings

import pandas as pd


def infer_video_id(stem: str) -> str:
    """Best-effort video group from filename stem (everything before first '_')."""
    return stem.split("_")[0]


def build_subclass_stratified_splits(
    image_df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each image a fold: 0 (CV train A), 1 (CV train B), 2 (held-out test).

    Video-level assignment using multi-label iterative stratification over the
    full set of subclasses present in each video — not just the primary subclass.
    This guarantees proportional representation of every subclass across all folds
    (subject to having enough videos; classes with only 1–2 videos cannot appear
    in all three folds simultaneously).

    Algorithm: Sechidis et al. 2011 iterative stratification, adapted for three
    folds. Videos are processed rarest-subclass-first; each is assigned to the
    fold with the highest remaining desired count for the triggering subclass,
    then the desired counts for all subclasses present in that video are
    decremented for the chosen fold.

    Parameters
    ----------
    image_df : DataFrame with columns: stem, subclass, binary_label
    test_fraction : proportion reserved for held-out test (fold=2)
    seed : integer seed for deterministic hash-based ordering
    """
    df = image_df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    # Binary presence matrix: video × subclass (True if ≥1 frame of that class)
    vid_sc = (
        df.groupby(["_vid", "subclass"])
        .size()
        .unstack(fill_value=0)
        .gt(0)
        .astype(int)
    )
    vids = vid_sc.index.tolist()
    subclasses = vid_sc.columns.tolist()

    target = {0: (1 - test_fraction) / 2, 1: (1 - test_fraction) / 2, 2: test_fraction}

    # n_vids_with_sc[sc] = number of videos that contain subclass sc
    n_vids_with_sc = vid_sc.sum(axis=0).to_dict()

    # Desired number of videos (containing sc) per fold, updated as we assign
    desired: dict[int, dict[str, float]] = {
        fold: {sc: n_vids_with_sc[sc] * frac for sc in subclasses}
        for fold, frac in target.items()
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
            best_fold = max(desired, key=lambda f: desired[f][sc])
            vid_fold[vid] = best_fold
            assigned.add(vid)
            # Decrement desired for every subclass this video covers
            for sc2 in subclasses:
                if vid_sc.loc[vid, sc2]:
                    desired[best_fold][sc2] = max(0.0, desired[best_fold][sc2] - 1)

    # Assign any remaining videos proportionally by fold size
    fold_counts = {f: sum(1 for v in vid_fold.values() if v == f) for f in target}
    for vid in sorted(set(vids) - assigned, key=_hash):
        best_fold = min(fold_counts, key=lambda f: fold_counts[f] / target[f])
        vid_fold[vid] = best_fold
        fold_counts[best_fold] += 1

    df["fold"] = df["_vid"].map(vid_fold)

    n_missing = df["fold"].isna().sum()
    if n_missing > 0:
        warnings.warn(f"{n_missing} images have no fold assignment and will be dropped.")
        df = df[df["fold"].notna()].copy()

    df["fold"] = df["fold"].astype(int)
    df.drop(columns=["_vid"], inplace=True)
    return df
