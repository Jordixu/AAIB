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
    rare_threshold: int = 0,
    rare_train_fraction: float = 0.80,
) -> pd.DataFrame:
    """Assign each image to split=train, val, or test.

    Assignment unit: (video_id, subclass) groups. All frames sharing the same
    video and subclass are kept together in one split to avoid within-class
    within-video leakage. Frames from the same video but different subclasses
    may be assigned to different splits (cross-class assignment), which avoids
    rare-class frames being dragged into val/test by a dominant co-occurring class.

    For rare classes (total frames < rare_threshold), the largest (video, subclass)
    groups are preferentially assigned to train first. For common classes, groups
    are shuffled by a deterministic hash to approximate target proportions.

    Parameters
    ----------
    image_df : DataFrame with columns: stem, subclass, binary_label
    val_fraction : proportion reserved for validation
    test_fraction : proportion reserved for held-out test
    seed : integer seed for deterministic hash-based ordering
    rare_threshold : classes with fewer total images than this value are treated as
        rare and receive a boosted train fraction. 0 disables the feature.
    rare_train_fraction : train fraction applied to rare classes (the remainder is
        split between val and test in their original ratio).
    """
    train_fraction = 1.0 - val_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    df = image_df.copy()
    df["_vid"] = df["stem"].apply(infer_video_id)

    subclasses = sorted(df["subclass"].unique().tolist())
    n_images_with_sc = df.groupby("subclass").size().to_dict()

    def _class_fracs(sc: str) -> dict[int, float]:
        if rare_threshold > 0 and n_images_with_sc[sc] < rare_threshold:
            non_train = 1.0 - rare_train_fraction
            val_test_sum = val_fraction + test_fraction
            if val_test_sum > 0:
                rare_val = val_fraction / val_test_sum * non_train
                rare_test = test_fraction / val_test_sum * non_train
            else:
                rare_val = rare_test = non_train / 2.0
            return {0: rare_train_fraction, 1: rare_val, 2: rare_test}
        return {0: train_fraction, 1: val_fraction, 2: test_fraction}

    def _hash(vid: str, sc: str) -> str:
        return hashlib.md5(f"{seed}:{vid}:{sc}".encode()).hexdigest()

    # (video_id, subclass) → split_id
    vid_sc_split: dict[tuple[str, str], int] = {}

    for sc in subclasses:
        grp = (
            df[df["subclass"] == sc]
            .groupby("_vid")
            .size()
            .reset_index(name="n")
        )
        is_rare = rare_threshold > 0 and n_images_with_sc[sc] < rare_threshold

        if is_rare:
            # Largest groups first so the most frames flow into train.
            grp = grp.sort_values("n", ascending=False).reset_index(drop=True)
        else:
            grp["_h"] = grp["_vid"].apply(lambda v: _hash(v, sc))
            grp = grp.sort_values("_h").reset_index(drop=True)

        n_total = int(grp["n"].sum())
        fracs = _class_fracs(sc)
        remaining = {f: n_total * fracs[f] for f in (0, 1, 2)}
        covered: set[int] = set()
        rows = list(zip(grp["_vid"], grp["n"]))
        n_groups = len(rows)

        for idx, (vid, n) in enumerate(rows):
            still_needed = [f for f in (0, 1, 2) if f not in covered]
            future = n_groups - idx - 1

            if still_needed and future < len(still_needed):
                # Too few groups left to cover every missing split individually;
                # force coverage of the split with the highest remaining quota.
                best = max(still_needed, key=lambda f: remaining[f])
            else:
                best = max((0, 1, 2), key=lambda f: remaining[f])

            vid_sc_split[(vid, sc)] = best
            covered.add(best)
            remaining[best] = max(0.0, remaining[best] - n)

    df["fold"] = [vid_sc_split.get((v, s), -1) for v, s in zip(df["_vid"], df["subclass"])]

    n_missing = (df["fold"] == -1).sum()
    if n_missing > 0:
        warnings.warn(f"{n_missing} images have no fold assignment and will be dropped.")
        df = df[df["fold"] != -1].copy()

    df["fold"] = df["fold"].astype(int)
    df["split"] = df["fold"].map(SPLIT_ID_TO_NAME)
    df.drop(columns=["_vid"], inplace=True)
    return df
