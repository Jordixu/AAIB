"""Phase 6 interpretability audit: Grad-CAM on True Positives."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from kvasir_triage.config import load_config
from kvasir_triage.data.dataloaders import make_dataloaders
from kvasir_triage.eval.gradcam import run_gradcam_audit
from kvasir_triage.models.registry import build_model
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, manifest_path, outputs_root, norm_stats_path
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import load_json, save_json


def _validate_shards_available(df: pd.DataFrame, train_fold: int) -> None:
    """Fail early when the manifest exists but the image shards are missing."""
    required = {"fold", "shard_rel_path", "local_idx", "binary_label", "subclass"}
    missing_cols = sorted(required.difference(df.columns))
    if missing_cols:
        raise RuntimeError(f"Manifest is missing required columns: {missing_cols}")

    val_fold = 1 - train_fold
    val_df = df[df["fold"] == val_fold]
    if val_df.empty:
        raise RuntimeError(f"No validation rows found for fold {val_fold}.")

    missing = []
    for rel in val_df["shard_rel_path"].drop_duplicates():
        path = outputs_root() / rel
        if not path.exists():
            missing.append(path)
        if len(missing) >= 5:
            break

    if missing:
        sample = "\n  ".join(str(p) for p in missing)
        raise RuntimeError(
            "Grad-CAM needs the preprocessed NPZ image shards, but they were not found.\n"
            f"First missing shard paths:\n  {sample}\n"
            "Run preprocessing again or copy the outputs/preprocessed/shards directory "
            "next to outputs/preprocessed/manifest.csv."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    log = setup_logging("gradcam_audit", logs_root())

    run_dir = args.run_dir
    with open(run_dir / "config.snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    # Checkpoint loading replaces the weights, so avoid downloading ImageNet weights here.
    model_cfg = {**cfg, "pretrained": False}
    model = build_model(cfg["model"], model_cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mf = manifest_path()
    if not mf.exists():
        mf = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
    df = pd.read_csv(mf)
    _validate_shards_available(df, cfg.get("fold", 0))

    loader_cfg = {
        **cfg,
        "batch_size": args.batch_size,
        "in_memory": False,
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": device.type == "cuda",
    }
    _, val_loader = make_dataloaders(df, cfg.get("fold", 0), loader_cfg)

    norm_stats = load_json(norm_stats_path())
    train_fold = cfg.get("fold", 0)
    mean = norm_stats[f"fold_{train_fold}"]["mean"]
    std = norm_stats[f"fold_{train_fold}"]["std"]

    out_dir = run_dir / "gradcam"
    csv_path = run_gradcam_audit(
        model,
        cfg["model"],
        val_loader,
        out_dir,
        device,
        args.max_samples,
        mean=mean,
        std=std,
        threshold=args.threshold,
    )

    # compute summary stats
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    masses = [float(r["activation_mass"]) for r in rows if r["activation_mass"] not in ("", "None")]
    if masses:
        stats = {"mean_activation_mass": float(np.mean(masses)), "n_samples": len(masses)}
        save_json(stats, run_dir / "gradcam" / "activation_mass_stats.json")
        log.info("Mean activation mass: %.3f (n=%d)", stats["mean_activation_mass"], stats["n_samples"])
    else:
        log.warning("No activation mass computed (bbox masks are not available in the current dataset).")
    if not rows:
        log.warning("No true positives found at threshold %.3f; no thumbnails were generated.", args.threshold)

    log.info("Grad-CAM results: %s", out_dir)


if __name__ == "__main__":
    main()
