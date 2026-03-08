#!/usr/bin/env python3
"""Materialize features into the Feast offline/online stores.

This script:
1. Loads raw data from XLSX
2. Applies preprocessing and feature engineering
3. Saves as parquet (Feast offline source)
4. Applies Feast definitions and materializes to online store
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import combine_datasets, create_target, handle_missing, load_data
from src.utils import DATA_DIR, get_logger

logger = get_logger("materialize_features")

DATA_FILE = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
FEATURE_STORE_DIR = Path(__file__).resolve().parent.parent / "feature_store"
PARQUET_PATH = FEATURE_STORE_DIR / "data" / "student_features.parquet"


def main():
    print("=== Feature Materialization ===\n")

    # 1. Load and process data
    xlsx_path = DATA_DIR / "raw" / DATA_FILE
    if not xlsx_path.exists():
        print(f"ERROR: Dataset not found at {xlsx_path}")
        raise SystemExit(1)

    print(f"Loading data from {xlsx_path}...")
    sheets = load_data(xlsx_path)
    combined = combine_datasets(sheets)
    combined = create_target(combined)
    combined = engineer_features(combined)
    combined = handle_missing(combined)

    print(f"Processed dataset: {len(combined)} rows, {combined.shape[1]} columns")

    # 2. Prepare for Feast
    # Create a unique student_id (RA + year) for the entity key
    if "ra" in combined.columns:
        combined["student_id"] = (
            combined["ra"].astype(str) + "_" + combined["ano"].astype(int).astype(str)
        )
    else:
        combined["student_id"] = [f"student_{i}" for i in range(len(combined))]

    # Feast requires an event_timestamp
    combined["event_timestamp"] = pd.Timestamp(datetime.now(timezone.utc))

    # Keep only feature columns + entity + timestamp + target
    feature_cols = [
        "student_id",
        "event_timestamp",
        # Raw indicators
        "inde",
        "iaa",
        "ieg",
        "ips",
        "ida",
        "ipp",
        "ipv",
        "ian",
        "nota_mat",
        "nota_por",
        "nota_ing",
        "idade",
        "ponto_virada",
        "indicado_bolsa",
        "ano",
        # Engineered
        "media_notas",
        "nota_min",
        "anos_na_pm",
        "fase_num",
        "pedra_encoded",
        "genero_encoded",
        "indicadores_baixos",
        # Target (for training reads)
        "target",
    ]
    available = [c for c in feature_cols if c in combined.columns]
    df_features = combined[available].copy()

    # Ensure correct types
    int_cols = [
        "ponto_virada",
        "indicado_bolsa",
        "ano",
        "pedra_encoded",
        "genero_encoded",
        "indicadores_baixos",
        "target",
    ]
    for col in int_cols:
        if col in df_features.columns:
            df_features[col] = df_features[col].fillna(0).astype(int)

    # 3. Save as parquet (offline store source)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(PARQUET_PATH, index=False)
    print(f"Parquet saved: {PARQUET_PATH} ({len(df_features)} rows)")

    # 4. Apply Feast definitions and materialize
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(FEATURE_STORE_DIR))
    print("Applying Feast definitions...")
    store.apply(
        [
            __import__("importlib").import_module("feature_store.definitions").student,
            __import__("importlib")
            .import_module("feature_store.definitions")
            .student_features_view,
        ]
    )

    print("Materializing to online store...")
    store.materialize(
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime.now(timezone.utc),
    )

    print(f"\nDone! Features materialized for {len(df_features)} students.")
    print(f"  Offline store: {PARQUET_PATH}")
    print(f"  Online store:  {FEATURE_STORE_DIR / 'data' / 'online_store.db'}")
    print(f"  Registry:      {FEATURE_STORE_DIR / 'data' / 'registry.db'}")


if __name__ == "__main__":
    main()
