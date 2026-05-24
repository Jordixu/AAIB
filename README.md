# Kvasir-Capsule VCE Triage

Binary triage filter (Normal vs Suspicious) for Video Capsule Endoscopy frames.

See [docs/USAGE.md](docs/USAGE.md) for how to run everything, and [docs/PROJECT.md](docs/PROJECT.md) for background and decisions.

## Quick start

```bash
conda env create -f environment.yml
conda activate kvasir-triage
pip install -e .

# fix manifest paths after checkout on a new machine
python scripts/repair_manifest.py

# verify no patient leakage
python scripts/verify_splits.py

# smoke test (~2 min on CPU)
python scripts/smoke_test.py --epochs 2

# benchmark data loading on this machine
python scripts/benchmark_io.py --quick

# train one fold
python scripts/train.py --config configs/efficientnet_b0.yaml --fold 0
```
