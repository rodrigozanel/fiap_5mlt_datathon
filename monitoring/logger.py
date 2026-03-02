"""Structured JSON logging for prediction monitoring."""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "prediction_data"):
            log_entry.update(record.prediction_data)
        return json.dumps(log_entry, default=str)


def get_prediction_logger() -> logging.Logger:
    """Return a logger that writes prediction logs to a rotating JSON file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("predictions")
    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        LOGS_DIR / "predictions.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_prediction(
    input_data: dict[str, Any],
    prediction: int,
    probability: float,
    risk_level: str,
    latency_ms: float,
) -> None:
    """Log a single prediction event."""
    logger = get_prediction_logger()
    record = logger.makeRecord(
        name="predictions",
        level=logging.INFO,
        fn="",
        lno=0,
        msg="prediction",
        args=(),
        exc_info=None,
    )
    record.prediction_data = {
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "latency_ms": round(latency_ms, 2),
    }
    logger.handle(record)
