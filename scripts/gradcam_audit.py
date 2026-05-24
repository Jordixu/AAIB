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
from kvasir_triage.paths import logs_root, manifest_path, outputs_root
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    log = setup_logging("gradcam_audit", logs_root())

    run_dir = args.run_dir
    with open(run_dir / "config.snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    model = build_model(cfg["model"], cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mf = manifest_path()
    if not mf.exists():
        mf = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
    df = pd.read_csv(mf)

    _, val_loader = make_dataloaders(df, cfg.get("fold", 0), {**cfg, "num_workers": 0})

    out_dir = run_dir / "gradcam"
    csv_path = run_gradcam_audit(model, cfg["model"], val_loader, out_dir, device, args.max_samples)

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
        log.warning("No activation mass computed (no bbox annotations in val set).")

    log.info("Grad-CAM results: %s", out_dir)


if __name__ == "__main__":
    main()
