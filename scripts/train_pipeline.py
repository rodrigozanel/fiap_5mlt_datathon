#!/usr/bin/env python3
"""End-to-end training pipeline: load data, train models, evaluate, and save the best."""

from pathlib import Path

import pandas as pd

from src.evaluate import compare_models, evaluate_model, get_classification_report
from src.preprocessing import prepare_dataset
from src.train import save_model, train_model
from src.utils import DATA_DIR, get_logger

logger = get_logger("train_pipeline")

# ── Configuration ──────────────────────────────────────────────────────────

DATA_FILE = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
SPLIT_STRATEGY = "stratified"  # "stratified" or "temporal"
MODEL_TYPES = ["xgb", "rf", "lr"]  # Always available


def _try_lgbm() -> bool:
    """Check if LightGBM is available."""
    try:
        import lightgbm  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def main() -> None:
    print("=== Passos Magicos - Training Pipeline ===\n")

    # ── 1. Locate data ────────────────────────────────────────────────
    xlsx_path = DATA_DIR / "raw" / DATA_FILE
    if not xlsx_path.exists():
        print(f"ERROR: Dataset not found at {xlsx_path}")
        print(f"Copy the XLSX file to: {xlsx_path}")
        print(f"  cp '<download-dir>/{DATA_FILE}' {xlsx_path}")
        raise SystemExit(1)

    # ── 2. Load, preprocess, and split ────────────────────────────────
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

    # ── 4. Evaluate and compare ───────────────────────────────────────
    print("\n=== Model Comparison ===")
    comparison = compare_models(trained_models, X_test, y_test)
    print(comparison.to_string())

    # Select the best by F1 weighted
    best_name = comparison.index[0]
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
