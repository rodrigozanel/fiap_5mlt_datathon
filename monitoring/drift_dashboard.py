"""Streamlit dashboard for data drift monitoring."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.log"

st.set_page_config(page_title="Passos Magicos - Drift Monitor", layout="wide")
st.title("Passos Magicos - Monitoramento de Drift")


def load_predictions() -> pd.DataFrame:
    """Load prediction logs from JSON lines file."""
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()

    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                flat = {
                    "timestamp": entry.get("timestamp"),
                    "prediction": entry.get("prediction"),
                    "probability": entry.get("probability"),
                    "risk_level": entry.get("risk_level"),
                    "latency_ms": entry.get("latency_ms"),
                }
                input_data = entry.get("input", {})
                for k, v in input_data.items():
                    flat[f"input_{k}"] = v
                records.append(flat)
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --- Load data ---
df = load_predictions()

if df.empty:
    st.warning("Nenhuma predicao registrada ainda. Faca requests ao endpoint /predict.")
    st.stop()

st.sidebar.header("Filtros")
st.sidebar.metric("Total de Predicoes", len(df))

# --- Overview metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predicoes", len(df))
col2.metric("% Risco Alto", f"{(df['risk_level'] == 'alto').mean():.1%}")
col3.metric("Probabilidade Media", f"{df['probability'].mean():.3f}")
col4.metric("Latencia Media (ms)", f"{df['latency_ms'].mean():.1f}")

st.divider()

# --- Prediction distribution ---
st.subheader("Distribuicao de Predicoes")
col_left, col_right = st.columns(2)

with col_left:
    st.bar_chart(df["risk_level"].value_counts())

with col_right:
    st.bar_chart(df["prediction"].value_counts())

st.divider()

# --- Feature distributions ---
st.subheader("Distribuicao de Features (Dados de Producao)")

numeric_input_cols = [c for c in df.columns if c.startswith("input_") and df[c].dtype in ("float64", "int64")]

if numeric_input_cols:
    selected_feature = st.selectbox("Feature", numeric_input_cols)
    st.line_chart(df[selected_feature])

    st.subheader("Estatisticas Descritivas")
    st.dataframe(df[numeric_input_cols].describe().T)
else:
    st.info("Nenhuma feature numerica encontrada nos logs.")

st.divider()

# --- Probability over time ---
st.subheader("Probabilidade ao Longo do Tempo")
if "timestamp" in df.columns:
    chart_data = df.set_index("timestamp")[["probability"]]
    st.line_chart(chart_data)

# --- Latency monitoring ---
st.subheader("Latencia por Predicao")
if "timestamp" in df.columns:
    latency_data = df.set_index("timestamp")[["latency_ms"]]
    st.line_chart(latency_data)

# --- Raw logs ---
with st.expander("Logs Brutos (ultimas 50)"):
    st.dataframe(df.tail(50))
