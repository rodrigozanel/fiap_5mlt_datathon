"""OpenTelemetry instrumentation for custom spans and metrics.

When the app runs via `opentelemetry-instrument`, the tracer and meter
providers are already configured by the auto-instrumentation agent.
This module simply creates custom tracers, meters, and instruments
on top of whatever provider is active (auto or manual).
"""

from opentelemetry import metrics, trace


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    return metrics.get_meter(name)


# -- Pre-built instruments --------------------------------------------------

_meter = None
_prediction_counter = None
_risk_counter = None
_probability_histogram = None
_latency_histogram = None
_login_counter = None
_login_failure_counter = None
_model_load_histogram = None


def _ensure_meter():
    global _meter, _prediction_counter, _risk_counter, _probability_histogram
    global _latency_histogram, _login_counter, _login_failure_counter
    global _model_load_histogram
    if _meter is not None:
        return
    _meter = get_meter("passos-magicos")

    _prediction_counter = _meter.create_counter(
        "predictions.total",
        description="Total number of predictions served",
    )
    _risk_counter = _meter.create_counter(
        "predictions.risk_level",
        description="Predictions broken down by risk level",
    )
    _probability_histogram = _meter.create_histogram(
        "predictions.probability",
        description="Distribution of predicted probabilities",
        unit="1",
    )
    _latency_histogram = _meter.create_histogram(
        "predictions.latency_ms",
        description="End-to-end prediction latency",
        unit="ms",
    )
    _login_counter = _meter.create_counter(
        "auth.login.total",
        description="Total login attempts",
    )
    _login_failure_counter = _meter.create_counter(
        "auth.login.failures",
        description="Failed login attempts",
    )
    _model_load_histogram = _meter.create_histogram(
        "model.load_time_ms",
        description="Time to load the ML model from disk",
        unit="ms",
    )


def record_prediction(probability: float, risk_level: str, latency_ms: float):
    """Record prediction metrics."""
    _ensure_meter()
    _prediction_counter.add(1)
    _risk_counter.add(1, {"risk_level": risk_level})
    _probability_histogram.record(probability)
    _latency_histogram.record(latency_ms)


def record_login(success: bool):
    """Record login attempt metrics."""
    _ensure_meter()
    _login_counter.add(1, {"success": str(success)})
    if not success:
        _login_failure_counter.add(1)


def record_model_load(duration_ms: float):
    """Record model loading time."""
    _ensure_meter()
    _model_load_histogram.record(duration_ms)
