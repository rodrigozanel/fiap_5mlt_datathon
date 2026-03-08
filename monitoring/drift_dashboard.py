"""Streamlit dashboard for data drift monitoring."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.log"
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "data" / "student_features.parquet"

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

numeric_input_cols = [
    c
    for c in df.columns
    if c.startswith("input_") and df[c].dtype in ("float64", "int64")
]

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

# --- Feature Store ---
st.divider()
st.subheader("Feature Store (Feast)")

if FEATURE_STORE_PARQUET.exists():
    df_fs = pd.read_parquet(FEATURE_STORE_PARQUET)
    meta_cols = {"student_id", "event_timestamp", "target"}
    feature_cols = [c for c in df_fs.columns if c not in meta_cols]

    # Metadata from parquet file
    parquet_mtime = pd.Timestamp.fromtimestamp(
        FEATURE_STORE_PARQUET.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Try to get registry info
    registry_info = {}
    try:
        from feast import FeatureStore as _FS

        fs_path = BASE_DIR / "feature_store"
        store = _FS(repo_path=str(fs_path))
        views = store.list_feature_views()
        if views:
            fv = views[0]
            registry_info["Feature View"] = fv.name
            registry_info["Entidade"] = ", ".join(e.name for e in fv.entities)
            registry_info["TTL"] = str(fv.ttl)
    except Exception:
        pass

    fs_col1, fs_col2, fs_col3, fs_col4 = st.columns(4)
    fs_col1.metric("Registros no Store", len(df_fs))
    fs_col2.metric("Features", len(feature_cols))
    if "target" in df_fs.columns:
        fs_col3.metric("% Defasagem (treino)", f"{df_fs['target'].mean():.1%}")
    fs_col4.metric("Ultima Geracao", parquet_mtime)

    if registry_info:
        st.caption(
            f"**View:** {registry_info.get('Feature View', '–')} | "
            f"**Entidade:** {registry_info.get('Entidade', '–')} | "
            f"**TTL:** {registry_info.get('TTL', '–')} | "
            f"**Provider:** local (SQLite)"
        )

    if "ano" in df_fs.columns:
        anos = sorted(df_fs["ano"].unique())
        st.caption(f"**Anos no dataset:** {', '.join(str(int(a)) for a in anos)}")
        with st.expander("Distribuicao por Ano"):
            ano_stats = df_fs.groupby("ano").agg(
                registros=("ano", "size"),
                defasagem=("target", "mean")
                if "target" in df_fs.columns
                else ("ano", "size"),
            )
            if "target" in df_fs.columns:
                ano_stats = (
                    df_fs.groupby("ano")
                    .agg(
                        registros=("ano", "size"),
                        pct_defasagem=("target", "mean"),
                    )
                    .round(3)
                )
                ano_stats["pct_defasagem"] = ano_stats["pct_defasagem"].map(
                    "{:.1%}".format
                )
            else:
                ano_stats = df_fs.groupby("ano").size().to_frame("registros")
            st.dataframe(ano_stats)

    with st.expander("Estatisticas do Feature Store"):
        st.dataframe(df_fs[feature_cols].describe().T.round(3))

    # Compare training vs production distributions
    input_to_feature = {f"input_{c}": c for c in feature_cols}
    comparable = [ic for ic in input_to_feature if ic in df.columns]

    if comparable:
        st.subheader("Treino vs Producao")
        selected = st.selectbox("Feature (comparacao)", comparable, key="fs_cmp")
        fs_feat = input_to_feature[selected]

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.caption("Treino (Feature Store)")
            st.bar_chart(df_fs[fs_feat].dropna().value_counts().sort_index().head(30))
            st.dataframe(df_fs[[fs_feat]].describe().T.round(3))
        with cmp_col2:
            st.caption("Producao (Predictions)")
            st.bar_chart(df[selected].dropna().value_counts().sort_index().head(30))
            st.dataframe(df[[selected]].describe().T.round(3))
else:
    st.info(
        "Feature Store nao encontrado. "
        "Execute: docker compose --profile feast run --rm materialize"
    )

# --- Raw logs ---
with st.expander("Logs Brutos (ultimas 50)"):
    st.dataframe(df.tail(50))
