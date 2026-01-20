"""
Evaluation metrics for protein localization.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with accuracy, macro_f1, mcc
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
    }


def compute_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary mapping class name to metrics dict
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    per_class = {}
    for i in range(num_classes):
        per_class[class_names[i]] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return per_class


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    return confusion_matrix(y_true, y_pred)
