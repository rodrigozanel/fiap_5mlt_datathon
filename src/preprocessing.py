"""Data loading, standardization, and preprocessing pipeline."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── Column mappings per year ────────────────────────────────────────────────

_COMMON_RENAME = {
    "Fase": "fase",
    "Turma": "turma",
    "Gênero": "genero",
    "Ano ingresso": "ano_ingresso",
    "Instituição de ensino": "instituicao_ensino",
    "IAA": "iaa",
    "IEG": "ieg",
    "IPS": "ips",
    "IDA": "ida",
    "IPV": "ipv",
    "IAN": "ian",
    "Rec Psicologia": "rec_psicologia",
    "RA": "ra",
}

COLUMN_MAP_2022 = {
    **_COMMON_RENAME,
    "Idade 22": "idade",
    "INDE 22": "inde",
    "Pedra 22": "pedra",
    "Matem": "nota_mat",
    "Portug": "nota_por",
    "Inglês": "nota_ing",
    "Atingiu PV": "ponto_virada",
    "Indicado": "indicado_bolsa",
    "Fase ideal": "fase_ideal",
    "Defas": "defasagem",
}

COLUMN_MAP_2023 = {
    **_COMMON_RENAME,
    "Idade": "idade",
    "INDE 2023": "inde",
    "Pedra 2023": "pedra",
    "Mat": "nota_mat",
    "Por": "nota_por",
    "Ing": "nota_ing",
    "IPP": "ipp",
    "Atingiu PV": "ponto_virada",
    "Indicado": "indicado_bolsa",
    "Fase Ideal": "fase_ideal",
    "Defasagem": "defasagem",
}

COLUMN_MAP_2024 = {
    **_COMMON_RENAME,
    "Idade": "idade",
    "INDE 2024": "inde",
    "Pedra 2024": "pedra",
    "Mat": "nota_mat",
    "Por": "nota_por",
    "Ing": "nota_ing",
    "IPP": "ipp",
    "Atingiu PV": "ponto_virada",
    "Indicado": "indicado_bolsa",
    "Fase Ideal": "fase_ideal",
    "Defasagem": "defasagem",
}

SHEET_CONFIG = {
    "PEDE2022": {"map": COLUMN_MAP_2022, "year": 2022},
    "PEDE2023": {"map": COLUMN_MAP_2023, "year": 2023},
    "PEDE2024": {"map": COLUMN_MAP_2024, "year": 2024},
}

# Columns we keep after renaming
KEEP_COLUMNS = [
    "ra",
    "fase",
    "idade",
    "genero",
    "ano_ingresso",
    "inde",
    "pedra",
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
    "ponto_virada",
    "indicado_bolsa",
    "defasagem",
    "ano",
]

PEDRA_ORDER = ["Quartzo", "Ágata", "Ametista", "Topázio"]

NUMERIC_FEATURES = [
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
    "anos_na_pm",
    "fase_num",
    "media_notas",
    "nota_min",
]

CATEGORICAL_FEATURES = ["genero_encoded", "pedra_encoded"]
BOOLEAN_FEATURES = ["ponto_virada", "indicado_bolsa"]


# ── Loading ─────────────────────────────────────────────────────────────────


def load_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load all PEDE sheets from the XLSX file.

    Returns dict with keys 'PEDE2022', 'PEDE2023', 'PEDE2024'.
    """
    path = Path(path)
    sheets = {}
    for sheet_name in SHEET_CONFIG:
        sheets[sheet_name] = pd.read_excel(path, sheet_name=sheet_name)
    return sheets


# ── Standardisation ─────────────────────────────────────────────────────────


def _normalise_boolean(series: pd.Series) -> pd.Series:
    """Convert various boolean representations to 0/1."""
    mapping = {
        "Sim": 1,
        "Não": 0,
        "sim": 1,
        "não": 0,
        "SIM": 1,
        "NÃO": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
        1.0: 1,
        0.0: 0,
    }
    return series.map(mapping).astype("Int64")  # nullable int


def _normalise_gender(series: pd.Series) -> pd.Series:
    """Normalise gender values across years."""
    mapping = {
        "Menina": "Feminino",
        "Menino": "Masculino",
        "Feminino": "Feminino",
        "Masculino": "Masculino",
    }
    return series.map(mapping)


def _extract_fase_num(series: pd.Series) -> pd.Series:
    """Extract numeric phase from Fase column (e.g. '7' -> 7, 'ALFA' -> 0)."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str.isdigit():
            return int(val_str)
        # ALFA is the introductory phase
        if "ALFA" in val_str.upper():
            return 0
        # Try to extract first digit
        for ch in val_str:
            if ch.isdigit():
                return int(ch)
        return np.nan

    return series.apply(_parse)


def standardize_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Rename and select columns for a single sheet."""
    cfg = SHEET_CONFIG[sheet_name]
    col_map = cfg["map"]
    year = cfg["year"]

    # Rename only columns that exist
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    result = df.rename(columns=rename)
    result["ano"] = year

    # If IPP not in renamed columns, fill with NaN
    if "ipp" not in result.columns:
        result["ipp"] = np.nan

    # Keep only relevant columns (that exist)
    available = [c for c in KEEP_COLUMNS if c in result.columns]
    result = result[available].copy()

    # Normalise booleans
    if "ponto_virada" in result.columns:
        result["ponto_virada"] = _normalise_boolean(result["ponto_virada"])
    if "indicado_bolsa" in result.columns:
        result["indicado_bolsa"] = _normalise_boolean(result["indicado_bolsa"])

    # Normalise gender
    if "genero" in result.columns:
        result["genero"] = _normalise_gender(result["genero"])

    return result


def combine_datasets(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Standardise and stack all sheets into one DataFrame."""
    frames = []
    for sheet_name, df in sheets.items():
        frames.append(standardize_columns(df, sheet_name))

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ── Target creation ─────────────────────────────────────────────────────────


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: 1 if defasagem > 0, else 0.

    Drops rows where defasagem is NaN and removes the original column.
    """
    result = df.copy()
    result = result.dropna(subset=["defasagem"])
    result["target"] = (result["defasagem"] > 0).astype(int)
    result = result.drop(columns=["defasagem"])
    return result


# ── Missing value handling ──────────────────────────────────────────────────


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    result = df.copy()

    # Coerce expected numeric model features that may come as strings
    # (e.g. "idade" or "inde" parsed from Excel as object dtype).
    numeric_candidates = NUMERIC_FEATURES + BOOLEAN_FEATURES + ["ano", "target"]
    for col in numeric_candidates:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Numeric columns: fill with median
    num_cols = result.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    # Categorical columns: fill with 'desconhecido'
    cat_cols = result.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna("desconhecido")

    return result


# ── Preprocessing pipeline ──────────────────────────────────────────────────


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for the feature set."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )
    # Preserve transformed feature names so downstream estimators
    # (notably LightGBM) see consistent named inputs on fit/predict.
    preprocessor.set_output(transform="pandas")
    return preprocessor


# ── Train/test split ────────────────────────────────────────────────────────


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    strategy: str = "stratified",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    strategy:
      - 'stratified': random stratified split
      - 'temporal': train on 2022+2023, test on 2024
    """
    if strategy == "temporal":
        train_mask = df["ano"].isin([2022, 2023])
        test_mask = df["ano"] == 2024
        train_df = df[train_mask]
        test_df = df[test_mask]
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

    return X_train, X_test, y_train, y_test


# ── Full pipeline helper ────────────────────────────────────────────────────


def prepare_dataset(
    xlsx_path: str | Path,
    strategy: str = "stratified",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """End-to-end: load → standardise → combine → target → split."""
    from src.feature_engineering import engineer_features

    sheets = load_data(xlsx_path)
    combined = combine_datasets(sheets)
    combined = create_target(combined)
    combined = engineer_features(combined)
    combined = handle_missing(combined)

    X_train, X_test, y_train, y_test = split_data(
        combined, strategy=strategy
    )
    return X_train, X_test, y_train, y_test
