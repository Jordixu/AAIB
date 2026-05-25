"""PR-AUC, MCC, NPV, and per-subclass recall."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = expit(logits)
    preds = (probs >= threshold).astype(int)

    pr_auc = float(average_precision_score(labels, probs))
    mcc = float(matthews_corrcoef(labels, preds))

    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())

    return {
        "pr_auc": pr_auc,
        "mcc": mcc,
        "npv": float(npv),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def per_subclass_recall(
    logits: np.ndarray,
    labels: np.ndarray,
    subclasses: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Recall per subclass (only for Suspicious=1 subclasses)."""
    probs = expit(logits)
    preds = (probs >= threshold).astype(int)
    subclasses = np.array(subclasses)
    result: dict[str, float] = {}
    for sub in np.unique(subclasses):
        mask = subclasses == sub
        sub_labels = labels[mask]
        if sub_labels.sum() == 0:
            continue  # not a Suspicious subclass
        sub_preds = preds[mask]
        recall = float(sub_preds[sub_labels == 1].mean())
        result[sub] = recall
    return result
