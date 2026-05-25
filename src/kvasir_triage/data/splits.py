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

    Video-level assignment stratified by subclass so every subclass has
    representation in all three folds proportional to its total video count.

    For each subclass:
        - Videos are ranked by deterministic hash(seed, video_id).
        - round(n * test_fraction) go to fold=2; skipped if the subclass has
          fewer than 3 videos (all kept in CV folds to avoid empty training sets).
        - Remaining videos split evenly: first half → fold=0, second half → fold=1.

    When a video's frames span multiple subclasses (rare in Kvasir-Capsule),
    it is assigned to the subclass contributing the most frames, with hash
    tie-breaking.

    Parameters
    ----------
    image_df : DataFrame with columns: stem, subclass, binary_label
    test_fraction : proportion of each subclass's videos reserved for test
    seed : integer seed for deterministic hash-based ordering
    """
    df = image_df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    # Primary subclass per video (max-count wins, hash tie-break for equal counts)
    vid_subclass = (
        df.groupby(["_vid", "subclass"])
        .size()
        .reset_index(name="n")
        .sort_values(
            ["_vid", "n", "subclass"],
            ascending=[True, False, True],
        )
        .drop_duplicates("_vid")
        .set_index("_vid")["subclass"]
    )

    vid_fold: dict[str, int] = {}

    for subclass, vids_series in vid_subclass.groupby(vid_subclass):
        vids = sorted(vids_series.index.tolist())
        n = len(vids)

        # Deterministic shuffle via hash
        ranked = sorted(
            vids,
            key=lambda v: hashlib.md5(f"{seed}:{v}".encode()).hexdigest(),
        )

        # Only reserve a test slice when we have enough videos for all three
        # folds to remain non-empty
        n_test = round(n * test_fraction) if n >= 3 else 0
        n_remaining = n - n_test
        n_fold0 = n_remaining // 2

        for i, vid in enumerate(ranked):
            if i < n_test:
                vid_fold[vid] = 2
            elif i < n_test + n_fold0:
                vid_fold[vid] = 0
            else:
                vid_fold[vid] = 1

    df["fold"] = df["_vid"].map(vid_fold)

    n_missing = df["fold"].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} images have no fold assignment and will be dropped."
        )
        df = df[df["fold"].notna()].copy()

    df["fold"] = df["fold"].astype(int)
    df.drop(columns=["_vid"], inplace=True)
    return df
