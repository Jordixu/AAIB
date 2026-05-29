"""Verify split integrity: no video leakage across train/val/test splits."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvasir_triage.data.splits import SPLIT_ID_TO_NAME, add_split_column, infer_video_id
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, outputs_root


def audit(manifest_path: Path, log=None) -> bool:
    def _log(level, msg, *args):
        if log:
            getattr(log, level)(msg, *args)
        else:
            print(msg % args if args else msg)

    df = add_split_column(pd.read_csv(manifest_path))

    if "stem" not in df.columns:
        df["stem"] = df["filename"].apply(lambda f: Path(f).stem)

    df["video_id"] = df["stem"].apply(infer_video_id)

    splits_present = [SPLIT_ID_TO_NAME[f] for f in sorted(df["fold"].unique())]
    _log("info", "Splits in manifest: %s", splits_present)

    # --- Video-level leakage check across all split pairs ---
    fold_vids = {s: set(df[df["split"] == s]["video_id"]) for s in splits_present}

    rows = []
    any_leakage = False
    for i, fa in enumerate(splits_present):
        for fb in splits_present[i + 1 :]:
            overlap = fold_vids[fa] & fold_vids[fb]
            if overlap:
                any_leakage = True
                _log("error", "FAIL: %d videos appear in both split=%s and split=%s:", len(overlap), fa, fb)
                for v in sorted(overlap):
                    _log("error", "  %s", v)
                    rows.append({"video_id": v, "leakage_between": f"{fa},{fb}", "leakage": True})

    for f in splits_present:
        exclusive = fold_vids[f] - set().union(*(fold_vids[g] for g in splits_present if g != f))
        for v in sorted(exclusive):
            rows.append({"video_id": v, "leakage_between": str(f), "leakage": False})

    out_dir = outputs_root() / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "leakage_audit.csv"
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    _log("info", "Leakage audit written to %s", audit_path)

    # --- Subclass coverage check ---
    subclass_folds = df.groupby("subclass")["split"].apply(set)
    expected_folds = set(splits_present)
    any_missing = False
    for subclass, present in subclass_folds.items():
        missing = expected_folds - present
        if missing:
            any_missing = True
            _log("error", "FAIL: subclass '%s' absent from split(s) %s", subclass, sorted(missing))

    # --- Summary counts ---
    expected_ratio = {"train": 0.70, "val": 0.15, "test": 0.15}
    total = len(df)
    any_ratio_bad = False
    for split in splits_present:
        n_imgs = (df["split"] == split).sum()
        n_vids = len(fold_vids[split])
        ratio = n_imgs / total if total else 0.0
        _log("info", "%s: %d images, %d videos (%.2f%%)", split, n_imgs, n_vids, ratio * 100)
        if split in expected_ratio and abs(ratio - expected_ratio[split]) > 0.05:
            any_ratio_bad = True
            _log(
                "error",
                "FAIL: split '%s' ratio %.2f%% is outside ±5 percentage points of %.2f%%",
                split,
                ratio * 100,
                expected_ratio[split] * 100,
            )

    # Subclass counts per split
    counts = df.groupby(["subclass", "split"]).size().unstack(fill_value=0)
    _log("info", "Subclass counts per split:\n%s", counts.to_string())

    ok = not any_leakage and not any_missing and not any_ratio_bad
    if ok:
        _log("info", "PASS: no leakage, all subclasses present in all splits, ratios near 70/15/15.")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split integrity audit")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("verify_splits", logs_root())

    if args.manifest is None:
        from kvasir_triage.paths import manifest_path
        args.manifest = manifest_path()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    ok = audit(args.manifest, log=log)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
