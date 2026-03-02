FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (libgomp needed by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenTelemetry auto-instrumentation
RUN pip install opentelemetry-distro opentelemetry-exporter-otlp \
    && opentelemetry-bootstrap --action=install

# Copy application code
COPY app/ app/
COPY src/ src/
COPY scripts/ scripts/
COPY monitoring/ monitoring/

# Create directories
RUN mkdir -p logs data/raw data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["opentelemetry-instrument", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
