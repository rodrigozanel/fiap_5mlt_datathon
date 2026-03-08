FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY server.py .
COPY static/ static/

RUN mkdir -p data

EXPOSE 3000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]
