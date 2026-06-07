# KVASIR-CAPSULE VCE TRIAGE

**System Purpose:** Binary triage filter (Normal vs Suspicious) for Video Capsule Endoscopy (VCE) frames based on the Kvasir-Capsule dataset.
**Clinical Role:** Triage aid to prioritize pathologist review time. Not a standalone diagnostic tool.
**Package Contents:** Pipeline source code, trained-model artifacts, metrics, reproducible notebooks, and a Colab-runnable demo.

---

## 1. PACKAGE CONTENTS

```text
.
├── src/kvasir_triage/         # Installable library: core pipeline code
│   ├── data/                  # Dataloaders, label maps, bbox/mask helpers, transforms, splits, sampler
│   ├── models/                # Backbone wrappers & registry (DenseNet-161, EfficientNet-B0/V2-S/M/L, MobileNetV3-Large)
│   ├── losses/                # Binary focal loss
│   ├── training/              # Trainer loop, optimizer/scheduler, checkpointing, early stopping
│   ├── eval/                  # Metrics, threshold calibration, CV summaries, Grad-CAM audit, path A/B/C diagnosis
│   ├── tracking/              # JSONL run logger, hardware snapshot, optional W&B logger
│   └── *.py                   # config, paths, seeding, utils, log_setup
├── scripts/                   # Executable pipeline stages
│   └── slurm/                 # Batch submission scripts for cluster training
├── configs/                   # YAML configurations (base.yaml + per-model overrides)
├── tests/                     # Pytest suite (labels, splits, models, metrics, Grad-CAM, etc.)
├── notebooks/                 # EDA, results comparison, and Colab demo (run_demo.ipynb)
├── data/                      # Subsets and metadata (demo_sample/ for the demo notebook)
├── outputs/                   # Generated artifacts (EDA figures, shards, runs, test JSONs)
└── pyproject.toml / env*.yaml # Metadata and conda environment specs
```
---

## 2. PIPELINE EXECUTION (Code -> Results)

Linear pipeline. Each script reads from and writes to `outputs/`:

1. **`scripts/preprocess.py`**
   * Inputs: `data/labelled_images/`, `data/metadata.csv`
   * Operations: Subclass-stratified/video-grouped splits; resize/corner-mask to 336x336.
   * Outputs: `outputs/preprocessed/` (manifest.csv, config, stats, train/val/test NPZ shards).
2. **`scripts/train_cv.py`** (wraps `train.py`)
   * Operations: Trains models per `configs/*.yaml` (e.g., `--config configs/efficientnet_v2_m.yaml`).
   * Outputs: `outputs/runs/<run_id>_fold0/` (checkpoints, metrics.jsonl, config/hardware snapshots).
3. **`scripts/evaluate.py`**
   * Operations: Validation split inference. Calibrates threshold via recall-anchored search (Target: 0.90 Recall, Min Precision: 0.10).
   * Outputs: `threshold_calibration.json`, `eval_predictions_val.{csv,npz}`, `inference_speed.json`, `suspicious_frames_best.csv`.
4. **`scripts/gradcam_audit.py`**
   * Operations: Grad-CAM on false negatives. Checks activation mass inside lesion bounding boxes.
   * Outputs: `<run_dir>/gradcam/` (`activation_mass.csv`, `activation_mass_stats.json`, thumbnail crops).
5. **`scripts/evaluate_test.py`**
   * Operations: Scores held-out test split using validation-calibrated threshold.
   * Outputs: `outputs/test_evaluation/<model>_test_results.json`.
6. **`scripts/build_cv_summary.py` & `diagnose.py`**
   * Operations: Aggregates multi-fold metrics. Assigns diagnosis Path A/B/C. (Note: Unused for headline single-run comparisons).

---

## 3. OUTPUTS SUPPORTING THE REPORT

Artifacts supporting the evaluation (no rerunning required):

* **`outputs/eda/`**: 16 PNGs (class balance, bbox geometry, augmentations, etc.) and `eda_summary.json`.
* **`outputs/preprocessed/`**: `manifest.csv` (split, shard, subclass, label, bbox area per frame) and NPZ shards.
* **`outputs/test_evaluation/*.json`**: Held-out test-set evaluations per architecture.
* **`outputs/runs/<model>_full_fold0/`**: Directories for all 7 tested architectures containing metrics, checkpoints, validation predictions, and Grad-CAM audits.

### Headline Comparison (Held-out test split, n=4,995)

Optimized for **PR-AUC**. EfficientNetV2-M is selected for the demo. Here is a summary of the results (on the test set):

| model | test PR-AUC | test MCC | test NPV @ 0.5 | precision @ calibrated threshold | recall @ calibrated threshold |
| --- | --- | --- | --- | --- | --- |
| densenet161 (pretrained) | 0.510 | 0.400 | 0.893 | 0.188 | 0.967 |
| densenet161 (scratch) | 0.154 | -0.016 | 0.857 | 0.140 | 0.908 |
| efficientnet_b0 | 0.532 | 0.469 | 0.916 | 0.167 | 0.909 |
| efficientnet_v2_l | 0.555 | 0.452 | 0.915 | 0.166 | 0.886 |
| **efficientnet_v2_m** | **0.687** | 0.486 | 0.899 | 0.202 | 0.922 |
| efficientnet_v2_s | 0.682 | 0.615 | 0.928 | 0.160 | 0.977 |
| mobilenet_v3_large | 0.554 | 0.492 | 0.916 | 0.165 | 0.950 |

**EfficientNetV2-M Calibrated Metrics:** * Threshold: 0.0161 (calibrated on Val for 0.90 recall).
* Absolute Stats: 0.92 recall, 0.97 NPV (641 True Positives, 54 False Negatives).
* Subclass Recall: 1.00 (Angiectasia, Erosion, Ulcer), 0.90 (Lymphangiectasia), 0.88 (Erythema), 0.83 (Foreign body).

---

## 4. DATA SAMPLE AND MANIFEST

Full 16 GB preprocessed shards are excluded from the repo. Bundled assets include:
* `outputs/preprocessed/manifest.csv` (46,715 kept frames, of the entire dataset as evidence).
* `data/demo_sample/` (Balanced mini-set for the demo notebook, including the images and the associated metadata).

The sample includes up to 60 frames per class (extracted from up to 6 videos, 10 frames per video).

> [!WARNING]
> Some class were excluded beforehand due to unsufficient data, see the report.
---

## 5. FLOW OF ASSETS TO RESULTS

```
[ data/labelled_images/ + data/metadata.csv ]
        | scripts/preprocess.py
[ outputs/preprocessed/ (manifest, stats, shards) ]
        | scripts/train_cv.py
[ outputs/runs/<model>/ (checkpoints, metrics, configs) ]
        | scripts/evaluate.py
[ outputs/runs/<model>/ (calibrated threshold, val predictions) ]
        | scripts/gradcam_audit.py
[ outputs/runs/<model>/gradcam/ (activation mass CSV, thumbnails) ]
        | scripts/evaluate_test.py
[ outputs/test_evaluation/<model>_test_results.json ]
        | notebooks/results_<model>.ipynb & results_comparison.ipynb
[ Final figures, tables, and reported text ]
```
---

## 6. DEMO NOTEBOOK (`run_demo.ipynb`)

> [!NOTE]
> Even though the notebook is optimized for Colab, it actually can run on CPU (takes less than 5 minutes). If you want to run it on Colab, read the header of the notebook to know the files that should be updated to the server.

Self-contained Colab walkthrough:
1. **Preprocess:** Runs dataset pipeline (`src/kvasir_triage/data/`) on `data/demo_sample/` (resize, mask, normalize, augment, split).
2. **Train:** Builds `MobileNetV3-Large`, runs one training epoch to demonstrate the loop.
3. **Evaluate:** Loads archived checkpoint (`outputs/runs/mobilenet_v3_large_full_fold0/best.pt`), plots curves, and reproduces test metrics, confusion matrices, and subclass recall from archived JSONs.

---

## 7. KNOWN LIMITATIONS

* **Low Precision by Design:** Anchoring to 0.90 recall yields ~0.20 test precision (4 out of 5 flags are false positives). Acceptable for triage sorting, invalid for standalone diagnosis.
* **Excluded Classes:** 4 subclasses are explicitly outside the model's operating envelope (`operating_envelope.json`).
* **Uneven Subclass Recall:** Rarer anomalies (Foreign body, Lymphangiectasia) consistently exhibit lower recall than common ones (Ulcer, Angiectasia).
* **Empty `cv_summary/`:** Headline models use single full runs. Multi-seed cross-validation scripts exist but were not executed for the baseline comparisons.
* **Preprocessing Snapshot Mismatch:** Bundled manifest uses a newer split (val=7,581 / test=6,284). Archived runs use an older pass of the exact same strategy with different seed assignments (val=8,933 / test=4,995). Archived files are internally consistent.
* **I/O Thrashing:** `BalancedBatchSampler` can cause high disk reads if working sets outgrow the LRU shard cache; requires tuning `lru_size`.
* **Single-Run Variance:** Architectures were trained once (1 fold, 1 seed). Sampling noise across architectures is not averaged out.
