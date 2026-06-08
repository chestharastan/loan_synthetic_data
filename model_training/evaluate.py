"""Evaluation helpers shared by the training entry point.

Produces a structured metrics dict (suitable for JSON export) and a readable
text report for the console.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_classifier(
    y_true,
    y_pred,
    labels: list | None = None,
    display_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable metrics dict for a classifier.

    ``labels`` fixes the class ordering (used for the confusion matrix); the
    optional ``display_names`` provides human-readable names (e.g. mapping the
    binary ``0/1`` approval target to ``Rejected/Approved``).
    """
    names = display_names if display_names is not None else labels
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(n) for n in names] if names is not None
        else sorted(np.unique(y_true).tolist()),
        "per_class": {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def text_report(
    name: str,
    y_true,
    y_pred,
    labels: list | None = None,
    display_names: list[str] | None = None,
) -> str:
    """Human-readable confusion matrix + classification report block."""
    names = display_names if display_names is not None else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true, y_pred, labels=labels, target_names=names, zero_division=0
    )
    return (
        f"\n=== {name} ===\n"
        f"Confusion matrix:\n{cm}\n\n"
        f"{rep}"
    )
