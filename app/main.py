"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.routes import router, set_model
from src.utils import MODEL_DIR, get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.joblib")))

    if model_path.exists():
        import joblib

        model = joblib.load(model_path)
        set_model(model)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. /predict will return 503.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Passos Magicos - Predicao de Defasagem Escolar",
    description="API para predicao de risco de defasagem escolar de estudantes da Associacao Passos Magicos",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
