"""One-shot builder for notebooks/run_demo.ipynb. Delete after use."""
import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "run_demo.ipynb"


def md(src: str) -> dict:
    lines = src.splitlines(keepends=False)
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }


def code(src: str) -> dict:
    lines = src.splitlines(keepends=False)
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ---------------------------------------------------------------------------
# Cell 0 - title / intro (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
# AAIB - Capsule endoscopy triage demo

This notebook is a **runnable, traceable demo** of the binary triage pipeline
described in the report: classifying capsule-endoscopy frames as
**normal vs abnormal**, building on the Kvasir-Capsule subclass labels.

It is meant to run end-to-end on a fresh Colab runtime (or a local clone) using
only what ships with the repository:

1. Load a small **sample of the labelled data** (`data/demo_sample/`) and
   reproduce the **same preprocessing and stratified-split logic** used for
   the reported run.
2. Build the **same `EfficientNetV2-M` model**, loss, optimizer, and
   augmentation pipeline as the real training script.
3. **Load the saved checkpoint** from the reported 21-epoch run and run
   **one additional live epoch** on the demo sample - enough to demonstrate
   the actual training/evaluation code path without needing hours of compute.
4. Display the **saved training curves, calibrated thresholds, and test-set
   metrics from the real run**, so the reported numbers are shown
   side-by-side with what this demo can reproduce on a laptop.

See section 10 for exactly how the pieces here map back to the report, and
what is illustrative-of-the-pipeline versus a reproduction-of-the-result."""))

# ---------------------------------------------------------------------------
# Cell 1 - setup heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 1. Setup

On Colab this clones the repository (shallow) and installs its dependencies;
locally it just locates the repo root. Either way, `PROJECT_ROOT` ends up
pointing at a checkout that has `src/kvasir_triage`, `data/demo_sample/`, and
`outputs/` available."""))

# ---------------------------------------------------------------------------
# Cell 2 - setup code
# ---------------------------------------------------------------------------
cells.append(code("""\
import os
import sys
from pathlib import Path

CLONE_DIR = Path("/content/AAIB")
REPO_URL = "https://github.com/Jordixu/AAIB.git"

IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def find_project_root() -> Path:
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "kvasir_triage").is_dir():
            return candidate
    return CLONE_DIR


if IN_COLAB:
    if not (CLONE_DIR / "src" / "kvasir_triage").is_dir():
        os.system(f"git clone --depth 1 {REPO_URL} {CLONE_DIR}")
    os.system(f"pip install -q -r {CLONE_DIR}/requirements.txt")
    PROJECT_ROOT = CLONE_DIR
else:
    PROJECT_ROOT = find_project_root()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)
print("PROJECT_ROOT =", PROJECT_ROOT)

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from kvasir_triage.data.labels import DIR_NAME_MAP, is_excluded
from kvasir_triage.data.preprocess import preprocess_image_to_uint8
from kvasir_triage.data.splits import build_subclass_stratified_splits
from kvasir_triage.data.transforms import AlbumentationsWrapper, make_transforms_for_fold
from kvasir_triage.eval.metrics import compute_metrics, per_subclass_recall
from kvasir_triage.losses.focal import BinaryFocalLoss
from kvasir_triage.models.registry import build_model
from kvasir_triage.training.checkpoint import load_checkpoint
from kvasir_triage.utils import load_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", DEVICE)"""))

# ---------------------------------------------------------------------------
# Cell 3 - data heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 2. The demo data sample

`data/demo_sample/` ships with the repo (it is the one exception to the
`data/*` gitignore rule). It was produced by
[`scripts/build_demo_sample.py`](../scripts/build_demo_sample.py), which:

- scans `data/labelled_images/` the same way `scripts/preprocess.py` does
  (mapping raw directory names through `DIR_NAME_MAP` and dropping
  `is_excluded` subclasses),
- for every subclass, picks frames from up to 6 source videos (the
  best-represented ones), spreading the picked frames across each video's
  frame range,
- copies ~485 frames total into `data/demo_sample/labelled_images/`
  (mirroring the original directory layout) plus a filtered
  `metadata_sample.csv`.

Picking frames from multiple videos per subclass matters: the real splitter
assigns whole `(video_id, subclass)` groups to a single split, so a
single-video subclass could never appear in train, val, *and* test at once."""))

# ---------------------------------------------------------------------------
# Cell 4 - load sample manifest
# ---------------------------------------------------------------------------
cells.append(code("""\
DEMO_DIR = PROJECT_ROOT / "data" / "demo_sample"
DEMO_IMAGES_DIR = DEMO_DIR / "labelled_images"

rows = []
for class_dir in sorted(DEMO_IMAGES_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    subclass = DIR_NAME_MAP.get(class_dir.name, class_dir.name)
    if is_excluded(subclass):
        continue
    for path in sorted(class_dir.rglob("*.jpg")):
        stem = path.stem
        rows.append({
            "path": path,
            "stem": stem,
            "subclass": subclass,
            "video_id": stem.split("_")[0],
            "binary_label": 0 if subclass == "Normal clean mucosa" else 1,
        })

image_df = pd.DataFrame(rows)
print(f"{len(image_df)} frames, {image_df['subclass'].nunique()} subclasses, "
      f"{image_df['video_id'].nunique()} source videos")
image_df.groupby("subclass").agg(
    frames=("stem", "size"),
    videos=("video_id", "nunique"),
    binary_label=("binary_label", "first"),
)"""))

# ---------------------------------------------------------------------------
# Cell 5 - splits heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 3. Stratified train/val/test splits

`build_subclass_stratified_splits` (the function the real pipeline uses, see
[`src/kvasir_triage/data/splits.py`](../src/kvasir_triage/data/splits.py))
assigns whole `(video_id, subclass)` groups to a single split - this keeps all
frames from one video and one subclass together, which avoids
within-class-within-video leakage between train and val/test. Frames from the
same video but a *different* subclass can still land in a different split.

We call it with the exact same arguments as `scripts/preprocess.py`, just on
our much smaller `image_df`."""))

# ---------------------------------------------------------------------------
# Cell 6 - splits code
# ---------------------------------------------------------------------------
cells.append(code("""\
SEED = 42

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    manifest = build_subclass_stratified_splits(image_df, seed=SEED)

print(manifest.groupby(["split", "binary_label"]).size().unstack(fill_value=0))
print()
print(manifest["split"].value_counts())"""))

# ---------------------------------------------------------------------------
# Cell 7 - preprocessing heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 4. Preprocessing

`preprocess_image_to_uint8` (used by `scripts/preprocess.py` to build the
training shards) crops the circular endoscope field of view, masks out the
black border, and resizes to a fixed square - the same transform every frame
goes through before it ever reaches a model. We run it here on a few sample
frames just to show what the network actually sees."""))

# ---------------------------------------------------------------------------
# Cell 8 - preprocessing code
# ---------------------------------------------------------------------------
cells.append(code("""\
sample_rows = manifest.sample(n=4, random_state=SEED)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for col, row in enumerate(sample_rows.itertuples()):
    raw = plt.imread(row.path)
    processed = preprocess_image_to_uint8(raw)
    axes[0, col].imshow(raw)
    axes[0, col].set_title(f"{row.subclass}\\nraw", fontsize=8)
    axes[1, col].imshow(processed)
    axes[1, col].set_title("preprocessed", fontsize=8)
    for ax in (axes[0, col], axes[1, col]):
        ax.axis("off")
fig.suptitle("Raw frames vs. preprocessed (cropped + masked + resized)", fontsize=11)
fig.tight_layout()
plt.show()"""))

# ---------------------------------------------------------------------------
# Cell 9 - augmentation heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 5. Augmentation and datasets

`build_train_transform` / `build_eval_transform` build the same
torchvision pipelines used for training and evaluation (random flips/affine
jitter for train, deterministic resize+normalize for eval), parameterized by
the dataset's saved normalization statistics
(`outputs/preprocessed/normalization_stats.json`, which ships with the repo).

`FrameDataset` wraps a split of the manifest, loading and preprocessing each
frame on the fly and returning `(image_tensor, binary_label)` pairs - exactly
the dataset class `scripts/train.py` instantiates for each split."""))

# ---------------------------------------------------------------------------
# Cell 10 - augmentation code
# ---------------------------------------------------------------------------
cells.append(code("""\
norm_stats = load_json(PROJECT_ROOT / "outputs" / "preprocessed" / "normalization_stats.json")
mean, std = norm_stats["mean"], norm_stats["std"]
print("normalization stats:", norm_stats)

IMAGE_SIZE = 224
train_tf = build_train_transform(IMAGE_SIZE, mean, std)
eval_tf = build_eval_transform(IMAGE_SIZE, mean, std)

train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

train_ds = FrameDataset(train_df, transform=train_tf)
val_ds = FrameDataset(val_df, transform=eval_tf)
test_ds = FrameDataset(test_df, transform=eval_tf)

print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

images, labels = next(iter(train_loader))
print("batch:", images.shape, labels.shape, images.dtype, labels.dtype)"""))

# ---------------------------------------------------------------------------
# Cell 11 - augmented samples heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
### A look at the augmented training batch

Unnormalizing a few training-loader samples shows the random flips, rotation,
and color jitter `build_train_transform` applies - the same augmentation the
real run trained under."""))

# ---------------------------------------------------------------------------
# Cell 12 - augmented samples code
# ---------------------------------------------------------------------------
cells.append(code("""\
mean_t = torch.tensor(mean).view(3, 1, 1)
std_t = torch.tensor(std).view(3, 1, 1)


def unnormalize(img: torch.Tensor) -> np.ndarray:
    arr = (img * std_t + mean_t).clamp(0, 1)
    return arr.permute(1, 2, 0).numpy()


fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, img, label in zip(axes, images, labels):
    ax.imshow(unnormalize(img))
    ax.set_title("abnormal" if label.item() else "normal", fontsize=9)
    ax.axis("off")
fig.suptitle("Augmented training batch (random flip / rotation / jitter)", fontsize=11)
fig.tight_layout()
plt.show()"""))

# ---------------------------------------------------------------------------
# Cell 13 - model heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 6. Model, loss, optimizer

`build_model("efficientnet_v2_m", ...)` constructs the same torchvision
backbone with a binary classification head that `scripts/train.py` builds for
this architecture. The reported run used `BinaryFocalLoss` (focal loss with a
configurable `alpha`/`gamma`, useful for the class imbalance between normal
and abnormal frames) and an `AdamW` optimizer with a cosine LR schedule."""))

# ---------------------------------------------------------------------------
# Cell 14 - model code
# ---------------------------------------------------------------------------
cells.append(code("""\
MODEL_NAME = "efficientnet_v2_m"
DISPLAY_NAME = "EfficientNetV2-M"

model = build_model(MODEL_NAME, pretrained=False, num_classes=1).to(DEVICE)

criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

n_params = sum(p.numel() for p in model.parameters())
print(f"{DISPLAY_NAME}: {n_params:,} parameters")"""))

# ---------------------------------------------------------------------------
# Cell 15 - checkpoint heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 7. Loading the reported checkpoint

The reported numbers come from a 21-epoch run saved under
`outputs/runs/efficientnet_v2_m_full_fold0/`. That directory is gitignored
(checkpoints are large), so it is only present on a machine that already has
the full `outputs/` tree - e.g. the original training machine, or a Colab
session where you've mounted/copied it in.

If it's not found, we fall back to the `pretrained=False` (random-init) model
built above - the rest of the notebook still runs, it just starts the live
epoch from scratch instead of from the converged weights."""))

# ---------------------------------------------------------------------------
# Cell 16 - checkpoint code
# ---------------------------------------------------------------------------
cells.append(code("""\
RUN_DIR = PROJECT_ROOT / "outputs" / "runs" / f"{MODEL_NAME}_full_fold0"
CHECKPOINT_PATH = RUN_DIR / "checkpoint_best.pt"

loaded_from_checkpoint = False
if CHECKPOINT_PATH.exists():
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    loaded_from_checkpoint = True
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    print(
        f"No checkpoint found at {CHECKPOINT_PATH} - "
        "outputs/runs/ is gitignored and not present in this checkout.\\n"
        "Continuing with the randomly-initialized model from section 6; "
        "the live epoch below still exercises the same code path, just "
        "without the converged starting weights."
    )"""))

# ---------------------------------------------------------------------------
# Cell 17 - live training heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 8. One live epoch

This is the actual training/evaluation loop from `scripts/train.py`,
condensed to a single epoch over the demo sample - just enough to demonstrate
the real code path (forward pass, focal loss, backward pass, optimizer step,
metric computation) on hardware that doesn't have hours to spare.

A note on `BinaryFocalLoss`: it expects the **raw** `[B, 1]` logits and does
the squeeze internally, so we pass `raw_logits` straight to `criterion` and
only flatten a detached copy afterwards for `compute_metrics`."""))

# ---------------------------------------------------------------------------
# Cell 18 - live training code
# ---------------------------------------------------------------------------
cells.append(code("""\
def run_epoch(loader, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()

            raw_logits = model(images)
            loss = criterion(raw_logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(raw_logits.squeeze(1).detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_labels).numpy()
    metrics = compute_metrics(targets, logits)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


print("Before the live epoch (using the loaded weights as-is):")
before_metrics = run_epoch(val_loader, train=False)
print({k: round(v, 4) for k, v in before_metrics.items()})

print("\\nRunning one live training epoch over the demo train split ...")
train_metrics = run_epoch(train_loader, train=True)
scheduler.step()
print({k: round(v, 4) for k, v in train_metrics.items()})

print("\\nAfter the live epoch:")
after_metrics = run_epoch(val_loader, train=False)
print({k: round(v, 4) for k, v in after_metrics.items()})

pd.DataFrame(
    [before_metrics, train_metrics, after_metrics],
    index=["val (before)", "train (live epoch)", "val (after)"],
).round(4)"""))

# ---------------------------------------------------------------------------
# Cell 19 - reported curves heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
## 9. The actual reported results

Everything above ran live, on the small demo sample, in a few minutes. The
numbers below are different: they are the **saved artifacts from the real
21-epoch run** referenced in the report, loaded directly from
`outputs/runs/efficientnet_v2_m_full_fold0/metrics.jsonl` (training curves) and
`outputs/test_evaluation/efficientnet_v2_m_full_test_results.json`
(held-out test metrics, the latter ships with the repo and is always
available).

If `metrics.jsonl` isn't present (gitignored, same as the checkpoint), the
curve plot is skipped and only the test results - which *do* ship with the
repo - are shown."""))

# ---------------------------------------------------------------------------
# Cell 20 - load metrics heading -> actually code: load + plot curves
# ---------------------------------------------------------------------------
cells.append(code("""\
def load_metrics(run_dir) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


metrics_df = load_metrics(RUN_DIR if RUN_DIR.exists() else None)

if metrics_df.empty:
    print(
        f"{RUN_DIR / 'metrics.jsonl'} not found - outputs/runs/ is gitignored.\\n"
        "Skipping the live curve plot; the test-set results below still load "
        "from the tracked outputs/test_evaluation/ directory."
    )
else:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, group in metrics_df.groupby("split"):
        axes[0].plot(group["epoch"], group["loss"], marker="o", label=split, markersize=3)
        axes[1].plot(group["epoch"], group["roc_auc"], marker="o", label=split, markersize=3)
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="ROC AUC", xlabel="epoch", ylabel="roc auc")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{DISPLAY_NAME} - reported run, 21 epochs (outputs/runs/.../metrics.jsonl)", fontsize=11)
    fig.tight_layout()
    plt.show()"""))

# ---------------------------------------------------------------------------
# Cell 21 - test results heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
### Held-out test set metrics

`outputs/test_evaluation/efficientnet_v2_m_full_test_results.json` is the
exact file the report's test-set table draws from. It includes the overall
binary metrics, the calibrated decision threshold, the confusion matrix, and
per-subclass recall (how well the model catches each abnormal finding type,
not just "abnormal" as a whole)."""))

# ---------------------------------------------------------------------------
# Cell 22 - test results code
# ---------------------------------------------------------------------------
cells.append(code("""\
TEST_RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "test_evaluation" / f"{MODEL_NAME}_full_test_results.json"
)
test_results = load_json(TEST_RESULTS_PATH)

overall = {k: v for k, v in test_results.items()
           if isinstance(v, (int, float)) and k not in ("n_models_ensembled",)}
print(f"Loaded {TEST_RESULTS_PATH.name}")
if "n_models_ensembled" in test_results:
    print(f"(reflects an ensemble of {test_results['n_models_ensembled']} model(s) / seeds)")
pd.Series(overall, name="value").to_frame().round(4)"""))

# ---------------------------------------------------------------------------
# Cell 23 - confusion matrix + threshold comparison code (markdown header)
# ---------------------------------------------------------------------------
cells.append(md("""\
### Confusion matrix and the BCE -> Focal threshold story

The report describes how the team moved from a plain BCE loss to a
recall-anchored focal-loss + calibrated-threshold setup, specifically to push
recall on abnormal frames up (missing an abnormal frame is the costly error in
a triage setting). The saved test-results JSON carries both the confusion
matrix at the calibrated threshold and the threshold value itself, which we
plot here next to a default 0.5 cutoff for context."""))

# ---------------------------------------------------------------------------
# Cell 24 - confusion matrix code
# ---------------------------------------------------------------------------
cells.append(code("""\
cm = np.array(test_results["confusion_matrix"])
threshold = test_results.get("threshold", 0.5)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(cm, cmap="Blues")
axes[0].set(
    title=f"Confusion matrix @ threshold={threshold:.3f}",
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["normal", "abnormal"], yticklabels=["normal", "abnormal"],
    xlabel="predicted", ylabel="actual",
)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
fig.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].bar(["default 0.5", f"calibrated {threshold:.3f}"],
            [0.5, threshold], color=["#9aa0a6", "#1a73e8"])
axes[1].set(title="Decision threshold: default vs. calibrated (BCE\\u2192Focal)",
            ylabel="probability cutoff")
axes[1].grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()"""))

# ---------------------------------------------------------------------------
# Cell 25 - per-subclass recall heading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("""\
### Per-subclass recall on the test set

`subclass_recall` in the saved results breaks the abnormal-class recall down
by finding type - this is the figure the report uses to argue the model isn't
just good at the easy/common conditions."""))

# ---------------------------------------------------------------------------
# Cell 26 -> wait, originally cell 25 was the per-subclass recall code, cell 26 the limitations.
# Re-map: append per-subclass recall code now (this becomes index 25 after the md at 24... let's just append in order)
# ---------------------------------------------------------------------------
cells.append(code("""\
subclass_recall = test_results.get("subclass_recall", {})
recall_series = pd.Series(subclass_recall).sort_values()

fig, ax = plt.subplots(figsize=(8, 0.4 * len(recall_series) + 1.5))
colors = ["#d93025" if v < 0.5 else "#1a73e8" for v in recall_series.values]
ax.barh(recall_series.index, recall_series.values, color=colors)
ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", label="recall = 0.5")
ax.set(xlim=(0, 1), xlabel="recall", title=f"{DISPLAY_NAME} - per-subclass recall (test set)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
plt.show()"""))

# ---------------------------------------------------------------------------
# Cell - limitations (markdown), final cell
# ---------------------------------------------------------------------------
cells.append(md("""\
## 10. Limitations and how this maps to the report

- **Sample size.** `data/demo_sample/` has about 485 frames from a handful of
  source videos per subclass, against the full `data/labelled_images/` set
  used for the reported run. Splits, class balance, and the live-epoch metrics
  above are all proportionally smaller and noisier than the real thing - they
  are illustrative of the *pipeline*, not a reproduction of the *result*.
- **One epoch, not 21.** The reported numbers come from a 21-epoch run on a
  GPU cluster (`outputs/runs/efficientnet_v2_m_full_fold0/`). Running that here
  would take hours on a CPU runtime, so this notebook instead loads the
  resulting checkpoint and runs one additional live epoch to demonstrate the
  same code path, then separately displays the saved curves and test metrics
  from the real run (section 9).
- **Gitignored run artifacts.** `outputs/runs/` (checkpoints, `metrics.jsonl`,
  `threshold_calibration.json`) is gitignored because of size; only the
  `outputs/test_evaluation/*_test_results.json` files and
  `outputs/preprocessed/normalization_stats.json` ship with the repo. On a
  machine that has the full `outputs/runs/` tree (e.g. the original training
  machine), sections 7 to 9 above pick those files up automatically and the
  comparisons become directly traceable end-to-end.
- **No ensembling here.** The reported test results
  (`efficientnet_v2_m_full_test_results.json`) may reflect more than one
  seed/model ensembled (`n_models_ensembled`); the live-training demo above
  uses a single model.

For the full picture - all models, all folds, cross-validation summaries, and
the report-quality figures - see `notebooks/results_comparison.ipynb` and the
per-model `notebooks/results_<model>.ipynb` notebooks, which read from the
complete (non-demo) `outputs/` tree."""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Wrote {len(cells)} cells to {NB_PATH}")
