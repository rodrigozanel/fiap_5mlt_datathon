"""Model training, hyperparameter tuning, and serialization."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessing_pipeline
from src.utils import MODEL_DIR, get_logger

logger = get_logger(__name__)


def _get_model(model_type: str) -> Any:
    """Return a classifier instance by name."""
    if model_type == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=200, verbose=-1, random_state=42
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "lgbm",
) -> Pipeline:
    """Train a full pipeline (preprocessing + classifier)."""
    preprocessor = build_preprocessing_pipeline()
    classifier = _get_model(model_type)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.info(f"Training {model_type} on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    return pipeline


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "lgbm",
    cv: int = 5,
) -> Pipeline:
    """Run grid search to find best hyperparameters."""
    preprocessor = build_preprocessing_pipeline()
    classifier = _get_model(model_type)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    param_grids = {
        "lgbm": {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 5, 7, -1],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__num_leaves": [15, 31, 63],
        },
        "xgb": {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
        },
        "rf": {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "lr": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
    }

    param_grid = param_grids.get(model_type, {})
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    logger.info(f"Tuning {model_type} with {cv}-fold CV...")
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_weighted",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")

    return search.best_estimator_


def save_model(model: Pipeline, path: str | Path | None = None) -> Path:
    """Serialize model pipeline to disk."""
    if path is None:
        path = MODEL_DIR / "model.joblib"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(path: str | Path | None = None) -> Pipeline:
    """Load serialized model pipeline."""
    if path is None:
        path = MODEL_DIR / "model.joblib"
    path = Path(path)
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
