"""Tests for the FastAPI application."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.main import app
from app.routes import _classify_risk, _prepare_input, set_model
from app.schemas import StudentInput


@pytest.fixture(autouse=True)
def _reset_model():
    """Reset model state before each test."""
    set_model(None)
    yield
    set_model(None)


@pytest.fixture
def client():
    """Test client without model loaded."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def dummy_model():
    """A trained pipeline that accepts the feature-engineered columns.

    Column order must match what _prepare_input produces.
    """
    np.random.seed(42)
    n = 50
    # Order matches _prepare_input output (original cols, then engineered)
    feature_cols = [
        "inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian",
        "nota_mat", "nota_por", "nota_ing", "idade",
        "ponto_virada", "indicado_bolsa", "ano",
        "media_notas", "nota_min",
        "anos_na_pm", "fase_num", "pedra_encoded", "genero_encoded",
        "indicadores_baixos",
    ]
    X = pd.DataFrame(
        {col: np.random.uniform(0, 10, n) for col in feature_cols}
    )
    y = pd.Series(np.random.choice([0, 1], n))

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)
    return model


@pytest.fixture
def client_with_model(dummy_model):
    """Test client with model loaded."""
    set_model(dummy_model)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_token(client):
    """Valid JWT token from login endpoint."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "passos-magicos"},
    )
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers with valid token."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_without_model(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False

    def test_health_with_model(self, client_with_model):
        data = client_with_model.get("/api/v1/health").json()
        assert data["model_loaded"] is True


class TestAuth:
    def test_login_valid_credentials(self, client):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "passos-magicos"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password(self, client):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_invalid_username(self, client):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "hacker", "password": "passos-magicos"},
        )
        assert response.status_code == 401


class TestPredict:
    def test_predict_no_auth_returns_401(self, client_with_model, sample_student_input):
        response = client_with_model.post("/api/v1/predict", json=sample_student_input)
        assert response.status_code == 401

    def test_predict_invalid_token_returns_401(self, client_with_model, sample_student_input):
        headers = {"Authorization": "Bearer invalidtoken"}
        response = client_with_model.post(
            "/api/v1/predict", json=sample_student_input, headers=headers
        )
        assert response.status_code == 401

    def test_predict_no_model_returns_503(self, client, auth_headers, sample_student_input):
        response = client.post(
            "/api/v1/predict", json=sample_student_input, headers=auth_headers
        )
        assert response.status_code == 503

    def test_predict_returns_200(self, client_with_model, auth_headers, sample_student_input):
        response = client_with_model.post(
            "/api/v1/predict", json=sample_student_input, headers=auth_headers
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client_with_model, auth_headers, sample_student_input):
        data = client_with_model.post(
            "/api/v1/predict", json=sample_student_input, headers=auth_headers
        ).json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ("baixo", "medio", "alto")

    def test_predict_invalid_input_returns_422(self, client_with_model, auth_headers):
        response = client_with_model.post(
            "/api/v1/predict", json={"fase": "invalid"}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_predict_missing_required_field(self, client_with_model, auth_headers):
        response = client_with_model.post(
            "/api/v1/predict", json={"fase": 3}, headers=auth_headers
        )
        assert response.status_code == 422


class TestClassifyRisk:
    def test_baixo(self):
        assert _classify_risk(0.1) == "baixo"
        assert _classify_risk(0.29) == "baixo"

    def test_medio(self):
        assert _classify_risk(0.3) == "medio"
        assert _classify_risk(0.59) == "medio"

    def test_alto(self):
        assert _classify_risk(0.6) == "alto"
        assert _classify_risk(0.99) == "alto"


class TestPrepareInput:
    def test_returns_dataframe(self, sample_student_input):
        student = StudentInput(**sample_student_input)
        df = _prepare_input(student)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_drops_raw_columns(self, sample_student_input):
        student = StudentInput(**sample_student_input)
        df = _prepare_input(student)
        for col in ("fase", "pedra", "genero", "ano_ingresso"):
            assert col not in df.columns

    def test_has_engineered_features(self, sample_student_input):
        student = StudentInput(**sample_student_input)
        df = _prepare_input(student)
        assert "media_notas" in df.columns
        assert "pedra_encoded" in df.columns
        assert "genero_encoded" in df.columns
