"""Phase 7 path classifier: Path A (success), B (Clever Hans), C (data-bound)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvasir_triage.utils import load_json, save_json


# Thresholds that define the three paths
PR_AUC_THRESHOLD = 0.85
NPV_THRESHOLD = 0.95
ACTIVATION_MASS_THRESHOLD = 0.5  # fraction of CAM in bbox to be considered "attending"


def classify_path(cv_metrics: dict, activation_mass_stats: dict | None = None) -> str:
    """Return 'A', 'B', or 'C'."""
    agg = cv_metrics.get("aggregate", {})
    mean_prauc = agg.get("pr_auc", {}).get("mean", 0.0)
    mean_npv = agg.get("npv", {}).get("mean", 0.0)

    performance_ok = mean_prauc >= PR_AUC_THRESHOLD and mean_npv >= NPV_THRESHOLD

    if not performance_ok:
        return "C"  # data-bound

    if activation_mass_stats:
        mean_mass = activation_mass_stats.get("mean_activation_mass", 1.0)
        if mean_mass < ACTIVATION_MASS_THRESHOLD:
            return "B"  # Clever Hans - attending outside lesion

    return "A"  # success


def build_operating_envelope(
    path: str,
    cv_metrics: dict,
    excluded_classes: list[str],
) -> dict[str, Any]:
    return {
        "phase_7_path": path,
        "cv_metrics_summary": cv_metrics.get("aggregate", {}),
        "excluded_classes": excluded_classes,
        "operating_envelope": {
            "applicable_subclasses": [
                c for c in [
                    "Angiectasia", "Blood - fresh", "Erosion", "Erythema",
                    "Foreign body", "Lymphangiectasia", "Polyp", "Ulcer",
                ]
            ],
            "excluded_with_rationale": {
                "Ampulla of Vater": "Excluded: too rare (<0.1% of dataset), unreliable class statistics.",
                "Blood - hematin": "Excluded: visually similar to Normal mucosa; confirmed low count.",
            },
            "input_requirements": {
                "image_size": "336x336",
                "format": "uint8 RGB",
                "corner_mask_applied": True,
            },
        },
    }


def run_diagnosis(cv_summary_dir: Path, excluded_classes: list[str] | None = None) -> dict:
    cv_path = cv_summary_dir / "cv_metrics.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"cv_metrics.json not found in {cv_summary_dir}")

    cv_metrics = load_json(cv_path)
    mass_path = cv_summary_dir / "activation_mass_stats.json"
    mass_stats = load_json(mass_path) if mass_path.exists() else None

    path = classify_path(cv_metrics, mass_stats)
    envelope = build_operating_envelope(
        path,
        cv_metrics,
        excluded_classes or ["Ampulla of Vater", "Blood - hematin"],
    )
    save_json(envelope, cv_summary_dir / "operating_envelope.json")
    return envelope
