"""Official and hybrid patient-level split assignment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def load_official_splits(splits_dir: Path) -> dict[str, int]:
    """Read split_0.csv / split_1.csv and return image stem -> fold (0 or 1)."""
    mapping: dict[str, int] = {}
    for fold_idx, csv_name in enumerate(["split_0.csv", "split_1.csv"]):
        p = splits_dir / csv_name
        if not p.exists():
            raise FileNotFoundError(f"Missing official split file: {p}")
        df = pd.read_csv(p)
        for fname in df["filename"]:
            stem = Path(fname).stem
            mapping[stem] = fold_idx
    return mapping


def infer_video_id(stem: str) -> str:
    """Best-effort video group from filename stem (everything before first '_')."""
    return stem.split("_")[0]


def _resolve_split_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign any image whose video_id spans both folds to a single fold.

    The official split files sometimes place frames from the same video in
    different folds.  We unify each conflicted video to whichever fold holds
    the majority of its frames; ties are broken deterministically by hash.
    """
    df = df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    counts = df.groupby(["_vid", "fold"]).size().unstack(fill_value=0)
    if 0 not in counts.columns or 1 not in counts.columns:
        df.drop(columns=["_vid"], inplace=True)
        return df

    conflicted = counts[(counts[0] > 0) & (counts[1] > 0)].index
    if len(conflicted):
        import warnings
        warnings.warn(
            f"{len(conflicted)} video IDs span both folds in the official splits; "
            "reassigning each to its majority fold."
        )
        for vid in conflicted:
            n0, n1 = int(counts.loc[vid, 0]), int(counts.loc[vid, 1])
            if n0 != n1:
                target = 0 if n0 > n1 else 1
            else:
                target = int(hashlib.md5(f"conflict:{vid}".encode()).hexdigest(), 16) % 2
            df.loc[df["_vid"] == vid, "fold"] = target

    df.drop(columns=["_vid"], inplace=True)
    return df


def reserve_test_videos(
    image_df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
    protected_video_ids: set[str] | None = None,
) -> set[str]:
    """Return a set of video IDs to hold out as a stratified test set.

    Stratified by video-level binary label (max over all frames in the video).
    Videos whose removal would eliminate all training examples of a subclass are
    automatically protected unless protected_video_ids is explicitly provided.

    The selection is deterministic: same inputs always produce the same test set.
    """
    df = image_df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    if protected_video_ids is None:
        vids_per_subclass = df.groupby("subclass")["_vid"].nunique()
        sole_subclasses = set(vids_per_subclass[vids_per_subclass == 1].index)
        protected_video_ids = set()
        for sc in sole_subclasses:
            protected_video_ids |= set(df[df["subclass"] == sc]["_vid"])

    vid_df = (
        df.groupby("_vid")["binary_label"]
        .max()
        .reset_index()
        .rename(columns={"_vid": "video_id"})
    )
    candidates = vid_df[~vid_df["video_id"].isin(protected_video_ids)]

    test_vids: set[str] = set()
    for _label, grp in candidates.groupby("binary_label"):
        vids = sorted(grp["video_id"].tolist())
        n_test = max(1, round(len(vids) * test_fraction))
        ranked = sorted(vids, key=lambda v: hashlib.md5(f"{seed}:test:{v}".encode()).hexdigest())
        test_vids |= set(ranked[:n_test])

    return test_vids


def build_split_index(
    image_df: pd.DataFrame,
    official_mapping: dict[str, int],
    test_video_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Attach a 'fold' column to image_df.

    fold=0 and fold=1 are the cross-validation folds.
    fold=2 is the held-out test set (assigned when test_video_ids is provided).

    image_df must have columns: stem, subclass.
    Images not in official splits that are Polyps use polyp_fold_override.
    Any remaining unassigned images are dropped with a warning.
    """
    df = image_df.copy()
    df["fold"] = df["stem"].map(official_mapping)

    n_missing = df["fold"].isna().sum()
    if n_missing > 0:
        import warnings
        warnings.warn(f"{n_missing} images have no fold assignment and will be dropped.")
        df = df[df["fold"].notna()].copy()

    df["fold"] = df["fold"].astype(int)
    df = _resolve_split_conflicts(df)

    if test_video_ids:
        test_mask = df["stem"].apply(infer_video_id).isin(test_video_ids)
        df.loc[test_mask, "fold"] = 2

    return df
