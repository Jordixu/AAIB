"""Threshold calibration: recall-anchored (default) and NPV-anchored."""
from __future__ import annotations

import numpy as np


def find_threshold_by_recall(
    logits: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.90,
    min_precision: float = 0.10,
    n_coarse: int = 99,
    n_fine: int = 1000,
) -> dict:
    """Find the maximum threshold t such that recall(t) >= target_recall.

    min_precision acts as a regularizer against class collapse: if the only way
    to achieve target_recall is to predict everything as positive (driving
    precision below min_precision), the dual constraint blocks that solution
    and status reflects which constraint is binding.

    Returns a dict with:
        threshold  – selected threshold (or None if target_recall is unreachable)
        recall     – recall at selected threshold
        precision  – precision at selected threshold
        status     – "ok" | "precision_not_achievable" | "recall_not_achievable"
    """
    probs = 1 / (1 + np.exp(-logits))

    def metrics_at(t):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        return recall, precision

    thresholds = np.linspace(0.01, 0.99, n_coarse)
    all_metrics = [metrics_at(t) for t in thresholds]
    recalls = np.array([m[0] for m in all_metrics])
    precisions = np.array([m[1] for m in all_metrics])

    dual_valid = (recalls >= target_recall) & (precisions >= min_precision)

    if dual_valid.any():
        boundary_idx = np.where(dual_valid)[0].max()
        lo = thresholds[max(boundary_idx - 1, 0)]
        hi = thresholds[min(boundary_idx + 1, len(thresholds) - 1)]
        fine_ts = np.linspace(lo, hi, n_fine)
        fine_metrics = [metrics_at(t) for t in fine_ts]
        fine_recalls = np.array([m[0] for m in fine_metrics])
        fine_precisions = np.array([m[1] for m in fine_metrics])
        fine_valid = (fine_recalls >= target_recall) & (fine_precisions >= min_precision)
        t_opt = float(fine_ts[fine_valid].max()) if fine_valid.any() else float(thresholds[boundary_idx])
        r, p = metrics_at(t_opt)
        return {
            "threshold": t_opt,
            "recall": r,
            "precision": p,
            "target_recall": target_recall,
            "min_precision": min_precision,
            "status": "ok",
        }

    # Recall-only feasible — precision floor is blocking the dual constraint.
    # Return the most selective threshold that still hits target_recall, and flag it.
    recall_only = recalls >= target_recall
    if recall_only.any():
        t_opt = float(thresholds[np.where(recall_only)[0].max()])
        r, p = metrics_at(t_opt)
        return {
            "threshold": t_opt,
            "recall": r,
            "precision": p,
            "target_recall": target_recall,
            "min_precision": min_precision,
            "status": "precision_not_achievable",
        }

    # Neither constraint is satisfiable — return the threshold with highest recall.
    best_idx = int(recalls.argmax())
    r, p = metrics_at(float(thresholds[best_idx]))
    return {
        "threshold": None,
        "recall": r,
        "precision": p,
        "target_recall": target_recall,
        "min_precision": min_precision,
        "status": "recall_not_achievable",
    }


def find_safe_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    target_npv: float = 0.95,
    n_coarse: int = 99,
    n_fine: int = 1000,
) -> dict:
    """Find the maximum threshold t such that NPV(t) >= target_npv.

    Useful when you need to trust the negative predictions (e.g. "this frame
    is safe to skip"). For reducing false negatives, prefer find_threshold_by_recall.

    Returns a dict with t_safe (or None if not achievable), the NPV at t_safe,
    and the closest NPV found when target is not reachable.
    """
    probs = 1 / (1 + np.exp(-logits))

    def npv_at(t):
        preds = (probs >= t).astype(int)
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    thresholds = np.linspace(0.01, 0.99, n_coarse)
    npvs = np.array([npv_at(t) for t in thresholds])

    valid_mask = npvs >= target_npv
    if valid_mask.any():
        boundary_idx = np.where(valid_mask)[0].max()
        lo = thresholds[max(boundary_idx - 1, 0)]
        hi = thresholds[min(boundary_idx + 1, len(thresholds) - 1)]
        fine_ts = np.linspace(lo, hi, n_fine)
        fine_npvs = np.array([npv_at(t) for t in fine_ts])
        fine_valid = fine_npvs >= target_npv
        t_safe = float(fine_ts[fine_valid].max()) if fine_valid.any() else float(thresholds[boundary_idx])
        return {"t_safe": t_safe, "npv_at_t_safe": npv_at(t_safe), "target_npv": target_npv}

    best_idx = npvs.argmax()
    return {
        "t_safe": None,
        "npv_at_t_safe": None,
        "closest_npv": float(npvs[best_idx]),
        "closest_threshold": float(thresholds[best_idx]),
        "target_npv": target_npv,
    }
