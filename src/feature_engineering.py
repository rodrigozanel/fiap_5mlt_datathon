"""Feature engineering: derived features, encoding, and selection."""

import numpy as np
import pandas as pd

from src.preprocessing import PEDRA_ORDER, _extract_fase_num


def encode_pedra(series: pd.Series) -> pd.Series:
    """Encode PEDRA as ordinal: Quartzo=1, Ágata=2, Ametista=3, Topázio=4."""
    mapping = {p: i + 1 for i, p in enumerate(PEDRA_ORDER)}
    return series.map(mapping).fillna(0).astype(int)


def encode_gender(series: pd.Series) -> pd.Series:
    """Encode gender: Feminino=0, Masculino=1."""
    return series.map({"Feminino": 0, "Masculino": 1}).fillna(-1).astype(int)


def create_academic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from subject grades."""
    result = df.copy()
    nota_cols = ["nota_mat", "nota_por", "nota_ing"]
    existing = [c for c in nota_cols if c in result.columns]

    if existing:
        result["media_notas"] = result[existing].mean(axis=1, skipna=True)
        result["nota_min"] = result[existing].min(axis=1, skipna=True)
    else:
        result["media_notas"] = np.nan
        result["nota_min"] = np.nan

    return result


def create_context_features(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """Create contextual features."""
    result = df.copy()

    # Years in Passos Magicos
    if "ano_ingresso" in result.columns and "ano" in result.columns:
        result["anos_na_pm"] = result["ano"] - result["ano_ingresso"]
        result["anos_na_pm"] = result["anos_na_pm"].clip(lower=0)
    else:
        result["anos_na_pm"] = np.nan

    # Numeric phase
    if "fase" in result.columns:
        result["fase_num"] = _extract_fase_num(result["fase"])
    else:
        result["fase_num"] = np.nan

    # Encoded categoricals
    if "pedra" in result.columns:
        result["pedra_encoded"] = encode_pedra(result["pedra"])
    else:
        result["pedra_encoded"] = 0

    if "genero" in result.columns:
        result["genero_encoded"] = encode_gender(result["genero"])
    else:
        result["genero_encoded"] = -1

    return result


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features related to student engagement and support."""
    result = df.copy()

    indicator_cols = ["iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian"]
    existing = [c for c in indicator_cols if c in result.columns]

    if existing:
        medians = result[existing].median()
        below = result[existing].lt(medians)
        result["indicadores_baixos"] = below.sum(axis=1)
    else:
        result["indicadores_baixos"] = 0

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    result = df.copy()
    result = create_academic_features(result)
    result = create_context_features(result)
    result = create_engagement_features(result)

    # Drop columns not needed for modeling
    drop_cols = [
        "ra",
        "fase",
        "turma",
        "genero",
        "pedra",
        "instituicao_ensino",
        "rec_psicologia",
        "ano_ingresso",
        "fase_ideal",
    ]
    to_drop = [c for c in drop_cols if c in result.columns]
    result = result.drop(columns=to_drop)

    return result


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "importance",
    threshold: float = 0.01,
) -> list[str]:
    """Select features based on importance from a quick LightGBM fit."""
    if method == "importance":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=100, verbose=-1, random_state=42
        )
        model.fit(X, y)
        importances = pd.Series(
            model.feature_importances_, index=X.columns
        )
        selected = importances[importances >= threshold * importances.sum()]
        return sorted(selected.index.tolist())
    else:
        return X.columns.tolist()
