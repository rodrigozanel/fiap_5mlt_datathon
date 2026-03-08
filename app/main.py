"""FastAPI application entry point."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from opentelemetry.trace import StatusCode

from app.routes import router, set_model
from app.telemetry import get_tracer, init_telemetry, record_model_load
from src.utils import MODEL_DIR, get_logger

logger = get_logger("api.main")
tracer = get_tracer("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    init_telemetry()

    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.joblib")))

    with tracer.start_as_current_span("model.load") as span:
        span.set_attribute("model.path", str(model_path))
        if model_path.exists():
            import joblib

            start = time.perf_counter()
            model = joblib.load(model_path)
            load_ms = (time.perf_counter() - start) * 1000

            set_model(model)
            span.set_attribute("model.load_time_ms", round(load_ms, 2))
            span.set_attribute("model.loaded", True)
            record_model_load(load_ms)
            logger.info(f"Model loaded from {model_path} in {load_ms:.0f}ms")
        else:
            span.set_attribute("model.loaded", False)
            span.set_status(StatusCode.ERROR, "Model file not found")
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
