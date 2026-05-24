"""Download pretrained weights for all models that need them.

Run this once on a machine with internet access, then upload the
weights/ directory to the cluster alongside the rest of the project.

The sbatch training scripts set TORCH_HOME=$SLURM_SUBMIT_DIR/weights,
so torchvision will find the cached files without hitting the network.

Usage:
    python scripts/download_weights.py
"""
import os
from pathlib import Path

# Must be set before importing torchvision so the cache lands in weights/
weights_dir = Path(__file__).parent.parent / "weights"
os.environ["TORCH_HOME"] = str(weights_dir)

from torchvision.models import (  # noqa: E402
    densenet161,
    DenseNet161_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

MODELS = [
    ("DenseNet161", densenet161, DenseNet161_Weights.IMAGENET1K_V1),
    ("EfficientNet-B0", efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
]

if __name__ == "__main__":
    cache_dir = weights_dir / "hub" / "checkpoints"
    print(f"Downloading to: {cache_dir}\n")
    for name, builder, weights in MODELS:
        print(f"  {name} ... ", end="", flush=True)
        builder(weights=weights)
        print("done")
    files = list(cache_dir.glob("*.pth"))
    print(f"\nCached files ({len(files)}):")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
    print("\nNext: upload the weights/ directory to the cluster.")
