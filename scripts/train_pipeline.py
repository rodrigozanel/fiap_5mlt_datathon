#!/usr/bin/env python3
"""End-to-end training pipeline: load data, train models, evaluate, and save the best."""

import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from pathlib import Path

import mlflow
import pandas as pd

from src.evaluate import (
    compare_models,
    evaluate_model,
    get_classification_report,
    get_confusion_matrix,
)
from src.preprocessing import prepare_dataset
from src.train import save_model, train_model
from src.utils import DATA_DIR, get_logger

logger = get_logger("train_pipeline")

# ── Configuration ──────────────────────────────────────────────────────────

DATA_FILE = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
SPLIT_STRATEGY = "stratified"  # "stratified" or "temporal"
MODEL_TYPES = ["xgb", "rf", "lr"]  # Always available
USE_FEATURE_STORE = os.getenv("USE_FEATURE_STORE", "false").lower() == "true"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = "passos-magicos-defasagem"

FEATURE_STORE_DIR = Path(__file__).resolve().parent.parent / "feature_store"


def _try_lgbm() -> bool:
    """Check if LightGBM is available."""
    try:
        import lightgbm  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _get_model_params(pipeline) -> dict:
    """Extract classifier parameters from a sklearn pipeline."""
    classifier = pipeline.named_steps["classifier"]
    return {k: v for k, v in classifier.get_params().items() if v is not None}


def main() -> None:
    print("=== Passos Magicos - Training Pipeline ===\n")

    # ── 0. Setup MLflow ─────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT}\n")

    # ── 1. Load data ────────────────────────────────────────────────
    if USE_FEATURE_STORE:
        print("Loading features from Feast Feature Store...")
        from feast import FeatureStore
        from sklearn.model_selection import train_test_split

        FeatureStore(repo_path=str(FEATURE_STORE_DIR))  # validate config
        parquet_path = FEATURE_STORE_DIR / "data" / "student_features.parquet"
        df = pd.read_parquet(parquet_path)

        # Drop Feast metadata columns
        drop = ["student_id", "event_timestamp"]
        df = df.drop(columns=[c for c in drop if c in df.columns])

        y = df.pop("target")
        X = df

        if SPLIT_STRATEGY == "temporal":
            train_mask = X["ano"].isin([2022, 2023])
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        print(f"  Source: Feature Store ({parquet_path})")
    else:
        xlsx_path = DATA_DIR / "raw" / DATA_FILE
        if not xlsx_path.exists():
            print(f"ERROR: Dataset not found at {xlsx_path}")
            print(f"Copy the XLSX file to: {xlsx_path}")
            print(f"  cp '<download-dir>/{DATA_FILE}' {xlsx_path}")
            raise SystemExit(1)

        print(f"Loading data from {xlsx_path}...")
        X_train, X_test, y_train, y_test = prepare_dataset(
            xlsx_path, strategy=SPLIT_STRATEGY
        )
    print(
        f"Dataset split: {len(X_train)} train, {len(X_test)} test, "
        f"{X_train.shape[1]} features"
    )
    print(f"Target distribution (train): {dict(y_train.value_counts())}")
    print(f"Target distribution (test):  {dict(y_test.value_counts())}\n")

    # ── 3. Train candidate models ─────────────────────────────────────
    model_types = MODEL_TYPES.copy()
    if _try_lgbm():
        model_types.insert(0, "lgbm")
        print("LightGBM available - including in comparison\n")
    else:
        print("LightGBM not available (missing libomp?) - skipping\n")

    trained_models = {}
    for mt in model_types:
        print(f"Training {mt}...", end=" ", flush=True)
        pipeline = train_model(X_train, y_train, model_type=mt)
        trained_models[mt] = pipeline
        print("done")

    # ── 4. Evaluate, compare, and log to MLflow ───────────────────────
    print("\n=== Model Comparison ===")
    comparison = compare_models(trained_models, X_test, y_test)
    print(comparison.to_string())

    # Select the best by F1 weighted
    best_name = comparison.index[0]

    # Log each model as a separate MLflow run
    for model_name, pipeline in trained_models.items():
        metrics = evaluate_model(pipeline, X_test, y_test)
        cm = get_confusion_matrix(pipeline, X_test, y_test)
        is_best = model_name == best_name

        with mlflow.start_run(run_name=model_name):
            # Dataset info
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "split_strategy": SPLIT_STRATEGY,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "n_features": X_train.shape[1],
                    "features": ", ".join(X_train.columns.tolist()),
                }
            )

            # Model hyperparameters
            model_params = _get_model_params(pipeline)
            mlflow.log_params({f"hp_{k}": v for k, v in model_params.items()})

            # Metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(
                {
                    "confusion_tn": int(cm[0][0]),
                    "confusion_fp": int(cm[0][1]),
                    "confusion_fn": int(cm[1][0]),
                    "confusion_tp": int(cm[1][1]),
                }
            )

            # Tags
            mlflow.set_tag("is_best", str(is_best))
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("feature_source", "feast" if USE_FEATURE_STORE else "inline")

            # Log the sklearn pipeline as an MLflow artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"  MLflow: logged {model_name}" + (" (best)" if is_best else ""))

    best_model = trained_models[best_name]
    best_metrics = evaluate_model(best_model, X_test, y_test)

    print(f"\nBest model: {best_name}")
    print(f"  F1 (weighted): {best_metrics['f1_weighted']:.4f}")
    print(f"  Accuracy:      {best_metrics['accuracy']:.4f}")
    if "auc_roc" in best_metrics:
        print(f"  AUC-ROC:       {best_metrics['auc_roc']:.4f}")

    print(f"\n=== Classification Report ({best_name}) ===")
    print(get_classification_report(best_model, X_test, y_test))

    # ── 5. Save the best model ────────────────────────────────────────
    model_path = save_model(best_model)
    print(f"\nModel saved to {model_path}")
    print("You can now start the API: poetry run uvicorn app.main:app --port 8000")


if __name__ == "__main__":
    main()
