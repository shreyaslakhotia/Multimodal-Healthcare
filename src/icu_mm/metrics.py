"""Validation threshold selection and locked-test metric computation."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def select_threshold(y_validation, probabilities, recall_target=None):
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Empty validation set")
    precision, recall = precision[:-1], recall[:-1]
    if recall_target is not None:
        if not 0 < recall_target <= 1:
            raise ValueError("Recall target must be in (0, 1]")
        candidates = thresholds[recall >= recall_target]
        if not len(candidates):
            raise ValueError("No threshold meets validation recall target")
        return float(candidates.max())
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return float(thresholds[f1 == f1.max()].max())


def score(y_true, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Expected finite probabilities in [0,1]")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Both outcome classes required for AUROC")
    tn, fp, fn, tp = confusion_matrix(y_true, probabilities >= threshold, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y_true, probabilities)),
            "auprc": float(average_precision_score(y_true, probabilities)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.,
            "recall": float(tp / (tp + fn)),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
            "specificity": float(tn / (tn + fp)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n": len(y_true), "positives": int(np.sum(y_true))}
