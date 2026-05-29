"""Verify split integrity: no video leakage across folds, all subclasses in all folds."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvasir_triage.data.splits import infer_video_id
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, outputs_root


def audit(manifest_path: Path, log=None) -> bool:
    def _log(level, msg, *args):
        if log:
            getattr(log, level)(msg, *args)
        else:
            print(msg % args if args else msg)

    df = pd.read_csv(manifest_path)

    if "stem" not in df.columns:
        df["stem"] = df["filename"].apply(lambda f: Path(f).stem)

    df["video_id"] = df["stem"].apply(infer_video_id)

    folds_present = sorted(df["fold"].unique())
    _log("info", "Folds in manifest: %s", folds_present)

    # --- Video-level leakage check across all fold pairs ---
    fold_vids = {f: set(df[df["fold"] == f]["video_id"]) for f in folds_present}

    rows = []
    any_leakage = False
    for i, fa in enumerate(folds_present):
        for fb in folds_present[i + 1 :]:
            overlap = fold_vids[fa] & fold_vids[fb]
            if overlap:
                any_leakage = True
                _log("error", "FAIL: %d videos appear in both fold=%d and fold=%d:", len(overlap), fa, fb)
                for v in sorted(overlap):
                    _log("error", "  %s", v)
                    rows.append({"video_id": v, "leakage_between": f"{fa},{fb}", "leakage": True})

    for f in folds_present:
        exclusive = fold_vids[f] - set().union(*(fold_vids[g] for g in folds_present if g != f))
        for v in sorted(exclusive):
            rows.append({"video_id": v, "leakage_between": str(f), "leakage": False})

    out_dir = outputs_root() / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "leakage_audit.csv"
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    _log("info", "Leakage audit written to %s", audit_path)

    # --- Subclass coverage check ---
    subclass_folds = df.groupby("subclass")["fold"].apply(set)
    expected_folds = set(folds_present)
    any_missing = False
    for subclass, present in subclass_folds.items():
        missing = expected_folds - present
        if missing:
            any_missing = True
            _log("error", "FAIL: subclass '%s' absent from fold(s) %s", subclass, sorted(missing))

    # --- Summary counts ---
    fold_labels = {0: "train", 1: "val", 2: "test"}
    for f in folds_present:
        fold_label = fold_labels.get(f, f"fold_{f}")
        n_imgs = (df["fold"] == f).sum()
        n_vids = len(fold_vids[f])
        _log("info", "%s: %d images, %d videos", fold_label, n_imgs, n_vids)

    # Subclass counts per fold
    counts = df.groupby(["subclass", "fold"]).size().unstack(fill_value=0)
    _log("info", "Subclass counts per fold:\n%s", counts.to_string())

    ok = not any_leakage and not any_missing
    if ok:
        _log("info", "PASS: no leakage, all subclasses present in all folds.")
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
