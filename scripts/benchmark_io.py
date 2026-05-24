"""Measure DataLoader throughput: in_memory vs shard mode, varying workers."""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from kvasir_triage.data.dataloaders import make_dataloaders
from kvasir_triage.log_setup import setup_logging
from kvasir_triage.paths import logs_root, manifest_path, outputs_root


def time_one_epoch(loader) -> tuple[float, int]:
    t0 = time.perf_counter()
    n = 0
    for imgs, labels, _ in loader:
        n += imgs.size(0)
    return time.perf_counter() - t0, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Only test a small subset of configs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log = setup_logging("benchmark_io", logs_root())

    mf = manifest_path()
    if not mf.exists():
        mf = outputs_root().parent / "outputs_phase1_2_3" / "preprocessed_manifest.csv"
    df = pd.read_csv(mf)

    configs = [
        {"in_memory": False, "num_workers": 0, "persistent_workers": False},
        {"in_memory": False, "num_workers": 2, "persistent_workers": True},
        {"in_memory": False, "num_workers": 4, "persistent_workers": True},
    ]
    if not args.quick:
        configs += [
            {"in_memory": False, "num_workers": 8, "persistent_workers": True},
            {"in_memory": True, "num_workers": 0, "persistent_workers": False},
            {"in_memory": True, "num_workers": 4, "persistent_workers": True},
        ]

    results = []
    for c in configs:
        mode = "in_memory" if c["in_memory"] else "shard"
        nw = c["num_workers"]
        pw = c["persistent_workers"]
        log.info("Testing %s, num_workers=%d, persistent=%s ...", mode, nw, pw)

        base_cfg = {**c, "batch_size": 32, "pin_memory": False}
        try:
            train_loader, _ = make_dataloaders(df, args.fold, base_cfg)
            for imgs, _, _ in train_loader:
                break
            elapsed, n = time_one_epoch(train_loader)
            sps = round(n / elapsed, 1)
        except Exception as e:
            log.error("FAILED: %s", e)
            sps = -1

        results.append({
            "mode": mode,
            "num_workers": nw,
            "persistent_workers": pw,
            "samples_per_sec": sps,
        })
        log.info("  -> %s samples/sec", sps)

    out_path = args.out or (outputs_root() / "io_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    log.info("Benchmark written to %s", out_path)


if __name__ == "__main__":
    main()
