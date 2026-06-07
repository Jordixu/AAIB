"""Build a small, class-balanced sample of raw frames for the demo notebook.

Copies a handful of frames per (subclass, source video) into data/demo_sample/,
mirroring the data/labelled_images/ layout, plus a filtered metadata_sample.csv.
Deterministic given --seed, so re-running with the same args reproduces the
same sample (the copy is skipped/overwritten, not appended to).

    python scripts/build_demo_sample.py
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from kvasir_triage.data.labels import DIR_NAME_MAP, is_excluded
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, project_root


def scan_raw_images(labelled_images_dir: Path) -> pd.DataFrame:
    """Discovery logic mirrors scripts/preprocess.py: map dir names, skip excluded."""
    rows = []
    for class_dir in sorted(labelled_images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        subclass = DIR_NAME_MAP.get(class_dir.name, class_dir.name)
        if is_excluded(subclass):
            continue
        for path in sorted(class_dir.rglob("*.jpg")):
            stem = path.stem
            rows.append({
                "path": path,
                "rel_dir": path.parent.relative_to(labelled_images_dir),
                "filename": path.name,
                "stem": stem,
                "subclass": subclass,
                "video_id": stem.split("_")[0],
            })
    return pd.DataFrame(rows)


def pick_sample(df: pd.DataFrame, max_videos: int, frames_per_video: int, seed: int) -> pd.DataFrame:
    """Deterministically pick a few frames from a few videos per subclass.

    Videos are ranked by frame count (most frames first) so the sample favors
    well-represented sources - this keeps at least a couple of videos per
    subclass, which the demo split needs to put something in train, val, AND
    test (the real splitter assigns whole (video, subclass) groups). Frames
    within a chosen video are spread across its frame range with linspace
    rather than taken from one burst, just for a bit more visual variety.
    """
    chunks = []
    for subclass, sub_df in df.groupby("subclass"):
        video_sizes = sub_df.groupby("video_id").size().sort_values(ascending=False)
        chosen_videos = video_sizes.index[:max_videos]
        for video_id in chosen_videos:
            vid_df = sub_df[sub_df["video_id"] == video_id].sort_values("filename")
            n_pick = min(frames_per_video, len(vid_df))
            idx = np.linspace(0, len(vid_df) - 1, num=n_pick).round().astype(int)
            idx = sorted(set(idx.tolist()))
            chunks.append(vid_df.iloc[idx])
    sample = pd.concat(chunks, ignore_index=True)
    # shuffle so the manifest doesn't read class-by-class, closer to how frames actually arrive
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-videos", type=int, default=6,
                        help="max source videos to draw from per subclass")
    parser.add_argument("--frames-per-video", type=int, default=10,
                        help="max frames to take from each chosen video")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: data/demo_sample)")
    args = parser.parse_args()

    log = setup_logging("build_demo_sample", logs_root())

    root = project_root()
    labelled_images_dir = root / "data" / "labelled_images"
    metadata_csv = root / "data" / "metadata.csv"
    out_dir = args.out or (root / "data" / "demo_sample")
    out_images_dir = out_dir / "labelled_images"

    log.info("Scanning %s ...", labelled_images_dir)
    df = scan_raw_images(labelled_images_dir)
    log.info("Found %d usable frames across %d subclasses", len(df), df["subclass"].nunique())

    sample = pick_sample(df, args.max_videos, args.frames_per_video, args.seed)
    log.info("Sampled %d frames from %d source videos", len(sample), sample["video_id"].nunique())

    if out_images_dir.exists():
        shutil.rmtree(out_images_dir)
    for row in sample.itertuples():
        dest_dir = out_images_dir / row.rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.path, dest_dir / row.filename)

    meta = pd.read_csv(metadata_csv, sep=";")
    meta_sample = meta[meta["filename"].isin(set(sample["filename"]))].reset_index(drop=True)
    meta_sample.to_csv(out_dir / "metadata_sample.csv", sep=";", index=False)

    counts = sample.groupby("subclass").agg(
        frames=("filename", "size"), videos=("video_id", "nunique"),
    )
    log.info("Per-subclass sample composition:\n%s", counts.to_string())
    log.info(
        "Wrote %d images under %s and %d metadata rows to %s",
        len(sample), out_images_dir, len(meta_sample), out_dir / "metadata_sample.csv",
    )


if __name__ == "__main__":
    main()
