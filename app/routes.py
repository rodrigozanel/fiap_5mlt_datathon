"""API routes for prediction, health check, and authentication."""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.auth import create_access_token, get_current_user, verify_credentials
from app.schemas import (
    HealthResponse,
    LoginRequest,
    PredictionOutput,
    StudentInput,
    TokenResponse,
)
from monitoring.logger import log_prediction
from src.feature_engineering import (
    create_academic_features,
    create_context_features,
    create_engagement_features,
    encode_gender,
    encode_pedra,
)
from src.preprocessing import PEDRA_ORDER
from src.utils import get_logger

logger = get_logger("api.routes")

router = APIRouter(prefix="/api/v1")

# Model is set by main.py on startup
_model = None


def set_model(model):
    """Set the loaded model (called from main.py startup)."""
    global _model
    _model = model


def _prepare_input(student: StudentInput) -> pd.DataFrame:
    """Convert StudentInput to a DataFrame matching the training features."""
    data = {
        "inde": student.inde,
        "iaa": student.iaa,
        "ieg": student.ieg,
        "ips": student.ips,
        "ida": student.ida,
        "ipp": student.ipp,
        "ipv": student.ipv,
        "ian": student.ian,
        "nota_mat": student.nota_mat,
        "nota_por": student.nota_por,
        "nota_ing": student.nota_ing,
        "idade": student.idade,
        "ponto_virada": int(student.atingiu_pv),
        "indicado_bolsa": int(student.indicado_bolsa),
        "fase": student.fase,
        "pedra": student.pedra,
        "genero": student.genero,
        "ano_ingresso": student.ano_ingresso,
        "ano": 2024,
    }
    df = pd.DataFrame([data])

    # Apply feature engineering
    df = create_academic_features(df)
    df = create_context_features(df)
    df = create_engagement_features(df)

    # Drop raw columns that were encoded
    drop_cols = ["fase", "pedra", "genero", "ano_ingresso"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def _classify_risk(probability: float) -> str:
    """Map probability to risk level."""
    if probability < 0.3:
        return "baixo"
    elif probability < 0.6:
        return "medio"
    return "alto"


@router.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Authenticate and return a JWT bearer token."""
    if not verify_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(body.username)
    return TokenResponse(access_token=token)


@router.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput, _: str = Depends(get_current_user)):
    """Predict student defasagem risk. Requires Bearer token."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    input_df = _prepare_input(student)
    probability = float(_model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)
    risk_level = _classify_risk(probability)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"prediction={prediction} probability={probability:.4f} "
        f"risk={risk_level} latency={latency_ms:.1f}ms"
    )

    log_prediction(
        input_data=student.model_dump(),
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        latency_ms=latency_ms,
    )

    return PredictionOutput(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
    )
