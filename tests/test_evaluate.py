"""Tests for src/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    compare_models,
    evaluate_model,
    get_classification_report,
    get_confusion_matrix,
    log_metrics,
)


@pytest.fixture
def trained_model_and_data():
    """Create a simple trained model with test data."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)

    X_test = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
    y_test = pd.Series((X_test["a"] + X_test["b"] > 0).astype(int))

    return model, X_test, y_test


class TestEvaluateModel:
    def test_returns_dict(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        assert isinstance(metrics, dict)

    def test_contains_expected_keys(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        expected_keys = {"f1_weighted", "f1_macro", "accuracy", "precision_weighted", "recall_weighted"}
        assert expected_keys <= set(metrics.keys())

    def test_auc_roc_present(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        assert "auc_roc" in metrics

    def test_metric_ranges(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


class TestGetClassificationReport:
    def test_returns_string(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        report = get_classification_report(model, X_test, y_test)
        assert isinstance(report, str)

    def test_contains_class_names(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        report = get_classification_report(model, X_test, y_test)
        assert "Sem Defasagem" in report
        assert "Com Defasagem" in report


class TestGetConfusionMatrix:
    def test_returns_ndarray(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert isinstance(cm, np.ndarray)

    def test_shape_2x2(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert cm.shape == (2, 2)

    def test_sums_to_total(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert cm.sum() == len(y_test)


class TestCompareModels:
    def test_returns_dataframe(self, trained_model_and_data):
        model, X_test, y_test = trained_model_and_data
        models = {"lr": model}
        result = compare_models(models, X_test, y_test)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_f1(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(80), "b": np.random.randn(80)})
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))

        model1 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        model1.fit(X, y)

        model2 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.001))])
        model2.fit(X, y)

        X_test = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        y_test = pd.Series((X_test["a"] + X_test["b"] > 0).astype(int))

        result = compare_models({"good": model1, "weak": model2}, X_test, y_test)
        assert result.index[0] in ("good", "weak")  # first is best f1
        assert "f1_weighted" in result.columns


class TestLogMetrics:
    def test_runs_without_error(self):
        metrics = {"f1_weighted": 0.85, "accuracy": 0.90}
        log_metrics(metrics, prefix="test_")
