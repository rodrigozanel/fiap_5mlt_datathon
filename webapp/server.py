"""Lightweight web UI server for Passos Magicos API."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Passos Magicos - Web UI")

HISTORY_FILE = Path("/app/data/history.jsonl")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/train", response_class=HTMLResponse)
def train_page():
    return (STATIC_DIR / "train.html").read_text()


@app.get("/history")
def get_history():
    if not HISTORY_FILE.exists():
        return []
    records = []
    for line in HISTORY_FILE.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-100:]  # last 100


@app.post("/history")
async def save_history(request: Request):
    entry = await request.json()
    entry["saved_at"] = datetime.now(timezone.utc).isoformat()
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return {"status": "ok"}