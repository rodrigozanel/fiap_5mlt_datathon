"""Model evaluation: metrics, reports, and plots."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    metrics = {
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }
    if auc is not None:
        metrics["auc_roc"] = auc

    return metrics


def get_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Return a formatted classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(
        y_test,
        y_pred,
        target_names=["Sem Defasagem", "Com Defasagem"],
        zero_division=0,
    )


def get_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    """Return the confusion matrix."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def compare_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare multiple models on the same test set."""
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
    return pd.DataFrame(results).set_index("model").sort_values(
        "f1_weighted", ascending=False
    )


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Log metrics to console."""
    for name, value in metrics.items():
        logger.info(f"{prefix}{name}: {value:.4f}")
