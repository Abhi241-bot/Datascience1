"""Imbalance-aware evaluation metrics.

For a ~0.17%-positive problem, accuracy is meaningless — we report PR-AUC,
precision, recall, F1, and the confusion matrix.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def compute_metrics(y_true, y_proba, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Return the imbalance-aware metric suite for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype="float64")
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"  PR-AUC    : {metrics['pr_auc']:.4f}   <-- primary metric",
        f"  ROC-AUC   : {metrics['roc_auc']:.4f}",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1        : {metrics['f1']:.4f}",
        f"  Confusion : TP={metrics['true_positives']} FP={metrics['false_positives']} "
        f"FN={metrics['false_negatives']} TN={metrics['true_negatives']}",
    ]
    return "\n".join(lines)
