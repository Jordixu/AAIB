"""Train one config on the fixed train split and validate on the val split."""
import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import yaml

from kvasir_triage.config import load_config
from kvasir_triage.data.dataloaders import make_dataloaders
from kvasir_triage.models.registry import build_model
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, manifest_path, outputs_root, runs_root
from kvasir_triage.seeding import set_all_seeds
from kvasir_triage.tracking.hardware import snapshot_hardware
from kvasir_triage.tracking.jsonl_logger import JsonlLogger
from kvasir_triage.tracking.wandb_logger import WandbLogger
from kvasir_triage.training.trainer import train
from kvasir_triage.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=[0],
        help="Legacy alias for the training split. Only fold 0 / split='train' is valid.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["fold"] = args.fold
    cfg["train_split"] = "train"
    cfg["val_split"] = "val"
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.max_batches is not None:
        cfg["max_batches"] = args.max_batches

    set_all_seeds(cfg.get("seed", 42))

    log = setup_logging("train", logs_root())

    run_id = args.run_id or f"{Path(args.config).stem}_fold{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # snapshot config
    with open(run_dir / "config.snapshot.yaml", "w") as f:
        yaml.dump(cfg, f)

    # hardware info
    hw = snapshot_hardware()
    save_json(hw, run_dir / "hardware.json")

    mf = manifest_path()
    if not mf.exists():
        raise FileNotFoundError(
            f"Manifest not found at {mf}. "
            "Run preprocessing first: sbatch scripts/slurm/preprocess.sbatch"
        )
    df = pd.read_csv(mf)

    train_loader, val_loader = make_dataloaders(df, args.fold, cfg)

    model = build_model(cfg["model"], cfg)

    jsonl_log = JsonlLogger(run_dir)
    wandb_log = WandbLogger(cfg, run_name=run_id)
    loggers = [jsonl_log, wandb_log]

    resume_path = run_dir / "checkpoints" / "last.pt" if args.resume else None

    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
        loggers=loggers,
        resume_path=resume_path,
    )

    save_json(
        {**result, "run_id": run_id, "fold": args.fold, "train_split": "train", "val_split": "val"},
        run_dir / "train_result.json",
    )
    wandb_log.finish()
    log.info("Done. Best PR-AUC: %.4f  (%d epochs)", result["best_pr_auc"], result["epochs_run"])
    log.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
