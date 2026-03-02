"""Tests for src/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    create_academic_features,
    create_context_features,
    create_engagement_features,
    encode_gender,
    encode_pedra,
    engineer_features,
)


class TestEncodePedra:
    def test_all_pedras(self):
        s = pd.Series(["Quartzo", "Ágata", "Ametista", "Topázio"])
        result = encode_pedra(s)
        assert list(result) == [1, 2, 3, 4]

    def test_unknown_pedra(self):
        s = pd.Series(["Unknown", np.nan])
        result = encode_pedra(s)
        assert list(result) == [0, 0]


class TestEncodeGender:
    def test_encode(self):
        s = pd.Series(["Feminino", "Masculino", "Feminino"])
        result = encode_gender(s)
        assert list(result) == [0, 1, 0]

    def test_unknown_gender(self):
        s = pd.Series([np.nan, "Other"])
        result = encode_gender(s)
        assert list(result) == [-1, -1]


class TestCreateAcademicFeatures:
    def test_media_notas(self):
        df = pd.DataFrame({"nota_mat": [6.0], "nota_por": [8.0], "nota_ing": [4.0]})
        result = create_academic_features(df)
        assert "media_notas" in result.columns
        assert abs(result["media_notas"].iloc[0] - 6.0) < 0.01

    def test_nota_min(self):
        df = pd.DataFrame({"nota_mat": [6.0], "nota_por": [8.0], "nota_ing": [4.0]})
        result = create_academic_features(df)
        assert result["nota_min"].iloc[0] == 4.0

    def test_handles_missing_notas(self):
        df = pd.DataFrame({"nota_mat": [6.0], "nota_por": [np.nan], "nota_ing": [4.0]})
        result = create_academic_features(df)
        assert abs(result["media_notas"].iloc[0] - 5.0) < 0.01

    def test_no_nota_columns(self):
        df = pd.DataFrame({"x": [1]})
        result = create_academic_features(df)
        assert "media_notas" in result.columns
        assert pd.isna(result["media_notas"].iloc[0])


class TestCreateContextFeatures:
    def test_anos_na_pm(self):
        df = pd.DataFrame({"ano_ingresso": [2020], "ano": [2024]})
        result = create_context_features(df)
        assert result["anos_na_pm"].iloc[0] == 4

    def test_anos_na_pm_clipped(self):
        df = pd.DataFrame({"ano_ingresso": [2025], "ano": [2024]})
        result = create_context_features(df)
        assert result["anos_na_pm"].iloc[0] == 0

    def test_fase_num(self):
        df = pd.DataFrame({"fase": [3]})
        result = create_context_features(df)
        assert result["fase_num"].iloc[0] == 3

    def test_fase_num_alfa(self):
        df = pd.DataFrame({"fase": ["ALFA"]})
        result = create_context_features(df)
        assert result["fase_num"].iloc[0] == 0

    def test_pedra_encoded(self):
        df = pd.DataFrame({"pedra": ["Ametista"]})
        result = create_context_features(df)
        assert result["pedra_encoded"].iloc[0] == 3

    def test_genero_encoded(self):
        df = pd.DataFrame({"genero": ["Feminino"]})
        result = create_context_features(df)
        assert result["genero_encoded"].iloc[0] == 0


class TestCreateEngagementFeatures:
    def test_indicadores_baixos(self):
        df = pd.DataFrame(
            {
                "iaa": [1.0, 9.0],
                "ieg": [1.0, 9.0],
                "ips": [1.0, 9.0],
                "ida": [1.0, 9.0],
                "ipv": [1.0, 9.0],
                "ian": [1.0, 9.0],
            }
        )
        result = create_engagement_features(df)
        assert "indicadores_baixos" in result.columns
        # First row has all low values, second all high
        assert result["indicadores_baixos"].iloc[0] == 6
        assert result["indicadores_baixos"].iloc[1] == 0


class TestEngineerFeatures:
    def test_drops_raw_columns(self):
        df = pd.DataFrame(
            {
                "ra": ["RA-1"],
                "fase": [3],
                "genero": ["Feminino"],
                "pedra": ["Ágata"],
                "ano_ingresso": [2020],
                "ano": [2024],
                "nota_mat": [6.0],
                "nota_por": [7.0],
                "nota_ing": [5.0],
                "inde": [6.5],
                "iaa": [7.0],
                "ieg": [6.0],
                "ips": [7.0],
                "ida": [6.0],
                "ipv": [6.0],
                "ian": [5.0],
            }
        )
        result = engineer_features(df)
        assert "ra" not in result.columns
        assert "fase" not in result.columns
        assert "genero" not in result.columns
        assert "pedra" not in result.columns
        assert "genero_encoded" in result.columns
        assert "pedra_encoded" in result.columns
        assert "media_notas" in result.columns
