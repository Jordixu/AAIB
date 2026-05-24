"""Verify that we do not have frames from the same video in different splits."""
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

    fold0_vids = set(df[df["fold"] == 0]["video_id"])
    fold1_vids = set(df[df["fold"] == 1]["video_id"])
    overlap = fold0_vids & fold1_vids

    out_dir = outputs_root() / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "leakage_audit.csv"

    rows = []
    for vid in sorted(overlap):
        rows.append({"video_id": vid, "folds": "0,1", "leakage": True})
    for vid in sorted(fold0_vids - overlap):
        rows.append({"video_id": vid, "folds": "0", "leakage": False})
    for vid in sorted(fold1_vids - overlap):
        rows.append({"video_id": vid, "folds": "1", "leakage": False})

    pd.DataFrame(rows).to_csv(audit_path, index=False)
    _log("info", "Leakage audit written to %s", audit_path)

    if overlap:
        _log("error", "FAIL: %d video IDs appear in both folds:", len(overlap))
        for v in sorted(overlap):
            _log("error", "  %s", v)
        return False

    _log("info", "PASS: no patient overlap. Fold 0: %d videos, Fold 1: %d videos.", len(fold0_vids), len(fold1_vids))
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Patient-leakage audit")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("verify_splits", logs_root())

    if args.manifest is None:
        from kvasir_triage.paths import manifest_path
        p = manifest_path()
        if not p.exists():
            p = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
        args.manifest = p

    ok = audit(args.manifest, log=log)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
