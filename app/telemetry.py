"""OpenTelemetry instrumentation for traces and metrics."""

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False

SERVICE = "passos-magicos-ml"


def init_telemetry():
    """Initialize OpenTelemetry tracer and meter providers."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    resource = Resource(attributes={SERVICE_NAME: SERVICE})

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(), export_interval_millis=15000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)


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