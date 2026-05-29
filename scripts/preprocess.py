"""Preprocess the data and pack into shards, building a manifest with metadata and computing normalization stats.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from kvasir_triage.data.labels import BINARY_LABEL, DIR_NAME_MAP, EXCLUDED_CLASSES, is_excluded
from kvasir_triage.data.masks import make_corner_polygon_mask
from kvasir_triage.data.preprocess import SHARD_SIZE, TARGET_SIZE, write_npz_shard
from kvasir_triage.data.splits import build_subclass_stratified_splits
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, outputs_root, preprocessed_root, shards_root
from kvasir_triage.seeding import set_all_seeds
from kvasir_triage.utils import save_json


def scan_images(labelled_images_dir: Path) -> pd.DataFrame:
    """Walk labelled_images/ and return a dataframe of all (non-excluded) images."""
    rows = []
    for class_dir in sorted(labelled_images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        subclass = DIR_NAME_MAP.get(class_dir.name, class_dir.name)
        if is_excluded(subclass):
            continue
        for img_path in sorted(class_dir.rglob("*.jpg")):
            rel = img_path.relative_to(labelled_images_dir).as_posix()
            rows.append({
                "path": str(img_path),
                "rel_path": rel,
                "filename": img_path.name,
                "stem": img_path.stem,
                "subclass": subclass,
                "binary_label": BINARY_LABEL.get(subclass, -1),
            })
    return pd.DataFrame(rows)


def _load_bbox_norm_areas(metadata_csv: Path, img_size: int, log) -> pd.Series:
    """Return a Series of normalized bbox area keyed by image stem.

    Handles both the raw metadata format (x1..x4, y1..y4) and the already
    processed bbox_areas.csv (bbox_norm_area column).
    """
    # Kvasir metadata uses semicolon separation
    sep = ";" if metadata_csv.suffix.lower() == ".csv" else ","
    try:
        md = pd.read_csv(metadata_csv, sep=sep)
    except Exception:
        md = pd.read_csv(metadata_csv)

    log.debug("Metadata CSV columns: %s", list(md.columns))

    if "bbox_norm_area" in md.columns:
        # Already processed (bbox_areas.csv)
        stem_col = (
            md["stem"]
            if "stem" in md.columns
            else md["filename"].apply(lambda f: Path(str(f)).stem)
        )
        return md.set_index(stem_col.values)["bbox_norm_area"].groupby(level=0).max()

    x_cols = ["x1", "x2", "x3", "x4"]
    y_cols = ["y1", "y2", "y3", "y4"]
    if not all(c in md.columns for c in x_cols + y_cols):
        log.warning("Could not find bbox columns in %s; skipping bbox enrichment.", metadata_csv)
        return pd.Series(dtype=float)

    xs = md[x_cols].apply(pd.to_numeric, errors="coerce")
    ys = md[y_cols].apply(pd.to_numeric, errors="coerce")
    area_norm = (
        (xs.max(axis=1) - xs.min(axis=1)).abs()
        * (ys.max(axis=1) - ys.min(axis=1)).abs()
        / float(img_size * img_size)
    )
    stems = md["filename"].apply(lambda f: Path(str(f).replace("\\", "/")).stem)
    return pd.Series(area_norm.values, index=stems.values).groupby(level=0).max()


def _compute_norm_stats(manifest: pd.DataFrame, mask: np.ndarray, log) -> dict:
    """Compute per-channel mean/std over mucosal region from written shards."""
    stats: dict = {}
    scopes = [
        ("all", manifest[manifest["fold"] < 2]),
        ("fold_0", manifest[manifest["fold"] == 0]),
        ("fold_1", manifest[manifest["fold"] == 1]),
    ]
    for scope_name, scope_df in scopes:
        rel_paths = scope_df["shard_rel_path"].unique()
        sum_c = np.zeros(3, dtype=np.float64)
        sumsq_c = np.zeros(3, dtype=np.float64)
        n_pixels = 0

        log.info("Computing normalization stats (%s) over %d shards ...", scope_name, len(rel_paths))
        for rel in rel_paths:
            shard_abs = (outputs_root() / rel).resolve()
            data = np.load(str(shard_abs), allow_pickle=False)
            imgs = data["images"]  # uint8 N,H,W,C
            # Only mucosal pixels; scale to [0, 1]
            vals = imgs[:, mask, :].astype(np.float64) / 255.0
            sum_c += vals.sum(axis=(0, 1))
            sumsq_c += (vals * vals).sum(axis=(0, 1))
            n_pixels += vals.shape[0] * vals.shape[1]

        mean = (sum_c / n_pixels).tolist()
        var = (sumsq_c / n_pixels) - np.array(mean) ** 2
        std = np.sqrt(np.maximum(var, 1e-12)).tolist()
        stats[scope_name] = {"mean": mean, "std": std, "n_pixels": int(n_pixels)}
        log.info(
            "  %s  mean=[%.4f, %.4f, %.4f]  std=[%.4f, %.4f, %.4f]",
            scope_name, *mean, *std,
        )

    stats["note"] = (
        "Fold 0 is train, fold 1 is validation, and fold 2 is held-out test. "
        "Use fold_0 stats for training/evaluation normalization."
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess Kvasir-Capsule images to NPZ shards")
    parser.add_argument("--labelled-images", type=Path, default=None,
                        help="Path to labelled_images/ directory")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip shards that already exist (default: True)")
    parser.add_argument("--no-norm-stats", action="store_true",
                        help="Skip normalization statistics computation")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of each subclass's videos held out as validation (fold=1). "
                             "Default: 0.15.")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="Fraction of each subclass's videos held out as test (fold=2). "
                             "Default: 0.15.")
    args = parser.parse_args()

    log = setup_logging("preprocess", logs_root())
    set_all_seeds(args.seed)

    root = outputs_root().parent

    # Locate labelled images
    if args.labelled_images is None:
        candidates = [
            root / "labelled_images-20260522T151751Z-3-001" / "labelled_images",
            root / "data" / "labelled_images",
        ]
        for c in candidates:
            if c.exists():
                args.labelled_images = c
                break
        if args.labelled_images is None:
            log.error("labelled_images/ not found. Tried: %s", candidates)
            sys.exit(1)

    log.info("labelled_images: %s", args.labelled_images)

    # Scan images
    log.info("Scanning images ...")
    df = scan_images(args.labelled_images)
    log.info("Found %d images across %d subclasses", len(df), df["subclass"].nunique())
    for sub, cnt in df["subclass"].value_counts().items():
        log.debug("  %-30s %d", sub, cnt)

    # Load bbox metadata
    metadata_csv = root / "data" / "metadata.csv"
    if metadata_csv.exists():
        log.info("Loading bbox metadata from %s", metadata_csv)
        bbox_areas = _load_bbox_norm_areas(metadata_csv, args.target_size, log)
        df["bbox_norm_area"] = df["stem"].map(bbox_areas)
        n_with_bbox = df["bbox_norm_area"].notna().sum()
        log.info("  bbox_norm_area populated for %d / %d rows", n_with_bbox, len(df))
    else:
        log.warning("metadata.csv not found; bbox_norm_area will be NaN in manifest.")
        df["bbox_norm_area"] = float("nan")

    # Assign folds with subclass-stratified video-level splits
    log.info(
        "Building subclass-stratified splits (val_fraction=%.2f, test_fraction=%.2f, seed=%d) ...",
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    df = build_subclass_stratified_splits(
        df,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    log.info(
        "Split: train=%d  val=%d  test=%d",
        (df["fold"] == 0).sum(),
        (df["fold"] == 1).sum(),
        (df["fold"] == 2).sum(),
    )
    for sub, grp in df.groupby("subclass"):
        counts = grp["fold"].value_counts().sort_index()
        log.info(
            "  %-30s  fold0=%d  fold1=%d  test=%d",
            sub,
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(2, 0),
        )

    # --- Pre-shard distribution check ---
    folds_present = sorted(df["fold"].unique())
    subclass_fold_counts = df.groupby(["subclass", "fold"]).size().unstack(fill_value=0)
    log.info("Subclass × fold distribution:\n%s", subclass_fold_counts.to_string())

    missing_coverage = []
    for subclass in subclass_fold_counts.index:
        for fold in folds_present:
            if subclass_fold_counts.loc[subclass, fold] == 0:
                missing_coverage.append((subclass, fold))

    if missing_coverage:
        for subclass, fold in missing_coverage:
            log.error("FAIL: subclass '%s' has 0 images in fold %d", subclass, fold)
        log.error(
            "Aborting before shard creation: %d subclass/fold slot(s) are empty. "
            "Re-run with a different --seed or inspect video distribution.",
            len(missing_coverage),
        )
        sys.exit(1)

    log.info("Distribution check PASSED: all subclasses present in all folds.")

    # Build corner mask and save
    mask = make_corner_polygon_mask(args.target_size)
    preprocessed_root().mkdir(parents=True, exist_ok=True)
    mask_path = preprocessed_root() / "corner_keep_mask_336.npy"
    np.save(str(mask_path), mask)
    log.info("Corner mask saved to %s", mask_path)
    shards_root().mkdir(parents=True, exist_ok=True)

    # Write shards and build manifest
    shard_meta = []
    active_folds = [0, 1] + ([2] if (df["fold"] == 2).any() else [])
    for fold in active_folds:
        fold_df = df[df["fold"] == fold].reset_index(drop=True)
        fold_shard_dir = shards_root() / f"fold_{fold}"
        fold_shard_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing fold %d (%d images) ...", fold, len(fold_df))

        rows_list = fold_df.to_dict("records")
        for i in range(0, len(rows_list), args.shard_size):
            shard_idx = i // args.shard_size
            batch = rows_list[i:i + args.shard_size]
            shard_name = f"fold{fold}_shard_{shard_idx:04d}.npz"
            shard_path = fold_shard_dir / shard_name
            rel_shard = f"preprocessed/shards/fold_{fold}/{shard_name}"

            if args.skip_existing and shard_path.exists():
                log.debug("  Skipping existing shard: %s", shard_name)
            else:
                log.info("  Writing %s (%d images) ...", shard_name, len(batch))
                write_npz_shard(batch, shard_path, mask, args.target_size)

            for j, row in enumerate(batch):
                shard_meta.append({
                    "fold": fold,
                    "shard_rel_path": rel_shard,
                    "local_idx": j,
                    "bbox_norm_area": row.get("bbox_norm_area"),
                    **{k: row[k] for k in ["path", "rel_path", "filename", "stem", "subclass", "binary_label"]},
                })

    manifest_df = pd.DataFrame(shard_meta)
    manifest_path = preprocessed_root() / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    log.info("Manifest written to %s  (%d rows)", manifest_path, len(manifest_df))

    # Normalization statistics
    if not args.no_norm_stats:
        norm_stats = _compute_norm_stats(manifest_df, mask, log)
        norm_path = preprocessed_root() / "normalization_stats.json"
        save_json(norm_stats, norm_path)
        log.info("Normalization stats written to %s", norm_path)

    # Preprocessing config
    save_json({
        "target_size": args.target_size,
        "shard_size": args.shard_size,
        "n_total": len(manifest_df),
        "excluded_classes": list(EXCLUDED_CLASSES),
        "bbox_source": str(metadata_csv) if metadata_csv else None,
    }, preprocessed_root() / "preprocessing_config.json")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
