"""NPV-anchored threshold calibration."""
from __future__ import annotations

import numpy as np


def find_safe_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    target_npv: float = 0.95,
    n_coarse: int = 99,
    n_fine: int = 1000,
) -> dict:
    """Find the maximum threshold t such that NPV(t) >= target_npv.

    Returns a dict with t_safe (or None if not achievable), the NPV at t_safe,
    and the closest NPV found when target is not reachable.
    """
    probs = 1 / (1 + np.exp(-logits))

    def npv_at(t):
        preds = (probs >= t).astype(int)
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    # coarse sweep
    thresholds = np.linspace(0.01, 0.99, n_coarse)
    npvs = np.array([npv_at(t) for t in thresholds])

    valid_mask = npvs >= target_npv
    if valid_mask.any():
        # find boundary between valid and invalid
        boundary_idx = np.where(valid_mask)[0].max()
        # fine sweep around boundary
        lo = thresholds[max(boundary_idx - 1, 0)]
        hi = thresholds[min(boundary_idx + 1, len(thresholds) - 1)]
        fine_ts = np.linspace(lo, hi, n_fine)
        fine_npvs = np.array([npv_at(t) for t in fine_ts])
        fine_valid = fine_npvs >= target_npv
        t_safe = float(fine_ts[fine_valid].max()) if fine_valid.any() else float(thresholds[boundary_idx])
        npv_at_safe = npv_at(t_safe)
        return {"t_safe": t_safe, "npv_at_t_safe": npv_at_safe, "target_npv": target_npv}

    # not reachable - return the best we can get
    best_idx = npvs.argmax()
    return {
        "t_safe": None,
        "npv_at_t_safe": None,
        "closest_npv": float(npvs[best_idx]),
        "closest_threshold": float(thresholds[best_idx]),
        "target_npv": target_npv,
    }
