"""Validation metrics with explicit positive class (1 = Churn Yes)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_CLASS = 1


def positive_scores(model, X) -> np.ndarray:
    """Probability of class 1, taken from the column matching classes_."""
    column = list(model.classes_).index(POSITIVE_CLASS)
    return model.predict_proba(X)[:, column]


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Ranking metrics use scores; threshold metrics use score >= threshold."""
    scores = positive_scores(model, X)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
