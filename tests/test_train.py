"""Tests for src/train.py."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import _get_model, load_model, save_model, train_model


def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _xgb_available() -> bool:
    try:
        from xgboost import XGBClassifier  # noqa: F401
        XGBClassifier()  # triggers libomp load
        return True
    except (ImportError, OSError, Exception):
        return False


@pytest.fixture
def training_data():
    """Generate simple training data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "inde": np.random.uniform(2, 10, n),
            "iaa": np.random.uniform(2, 10, n),
            "ieg": np.random.uniform(2, 10, n),
            "ips": np.random.uniform(2, 10, n),
            "ida": np.random.uniform(2, 10, n),
            "ipp": np.random.uniform(2, 10, n),
            "ipv": np.random.uniform(2, 10, n),
            "ian": np.random.uniform(2, 10, n),
            "nota_mat": np.random.uniform(2, 10, n),
            "nota_por": np.random.uniform(2, 10, n),
            "nota_ing": np.random.uniform(2, 10, n),
            "idade": np.random.randint(8, 20, n),
            "anos_na_pm": np.random.randint(0, 8, n),
            "fase_num": np.random.randint(0, 8, n),
            "media_notas": np.random.uniform(2, 10, n),
            "nota_min": np.random.uniform(2, 10, n),
            "genero_encoded": np.random.choice([0, 1], n),
            "pedra_encoded": np.random.choice([1, 2, 3, 4], n),
            "ponto_virada": np.random.choice([0, 1], n),
            "indicado_bolsa": np.random.choice([0, 1], n),
            "indicadores_baixos": np.random.randint(0, 7, n),
            "ano": np.random.choice([2022, 2023, 2024], n),
        }
    ), pd.Series(np.random.choice([0, 1], n))


class TestGetModel:
    @pytest.mark.skipif(
        not _lgbm_available(), reason="LightGBM not available (missing libomp?)"
    )
    def test_lgbm(self):
        model = _get_model("lgbm")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    @pytest.mark.skipif(
        not _xgb_available(), reason="XGBoost not available (missing libomp?)"
    )
    def test_xgb(self):
        model = _get_model("xgb")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_rf(self):
        model = _get_model("rf")
        assert hasattr(model, "fit")

    def test_lr(self):
        model = _get_model("lr")
        assert hasattr(model, "fit")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_model("unknown_model")


class TestTrainModel:
    def test_returns_pipeline(self, training_data):
        X, y = training_data
        pipeline = train_model(X, y, model_type="lr")
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_can_predict(self, training_data):
        X, y = training_data
        pipeline = train_model(X, y, model_type="lr")
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        assert set(preds) <= {0, 1}

    def test_pipeline_has_predict_proba(self, training_data):
        X, y = training_data
        pipeline = train_model(X, y, model_type="lr")
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)


class TestSaveLoadModel:
    def test_roundtrip(self, training_data):
        X, y = training_data
        pipeline = train_model(X, y, model_type="lr")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            save_model(pipeline, path)
            assert path.exists()

            loaded = load_model(path)
            original_preds = pipeline.predict(X)
            loaded_preds = loaded.predict(X)
            np.testing.assert_array_equal(original_preds, loaded_preds)
