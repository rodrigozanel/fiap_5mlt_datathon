"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    KEEP_COLUMNS,
    SHEET_CONFIG,
    _extract_fase_num,
    _normalise_boolean,
    _normalise_gender,
    build_preprocessing_pipeline,
    combine_datasets,
    create_target,
    handle_missing,
    split_data,
    standardize_columns,
)


class TestNormaliseBoolean:
    def test_sim_nao(self):
        s = pd.Series(["Sim", "Não", "Sim"])
        result = _normalise_boolean(s)
        assert list(result) == [1, 0, 1]

    def test_numeric(self):
        s = pd.Series([1, 0, 1.0, 0.0])
        result = _normalise_boolean(s)
        assert list(result) == [1, 0, 1, 0]

    def test_nan_handling(self):
        s = pd.Series(["Sim", None, "Não"])
        result = _normalise_boolean(s)
        assert result.iloc[0] == 1
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 0


class TestNormaliseGender:
    def test_menina_menino(self):
        s = pd.Series(["Menina", "Menino", "Menina"])
        result = _normalise_gender(s)
        assert list(result) == ["Feminino", "Masculino", "Feminino"]

    def test_already_normalised(self):
        s = pd.Series(["Feminino", "Masculino"])
        result = _normalise_gender(s)
        assert list(result) == ["Feminino", "Masculino"]


class TestExtractFaseNum:
    def test_numeric_string(self):
        s = pd.Series(["3", "5", "7"])
        result = _extract_fase_num(s)
        assert list(result) == [3, 5, 7]

    def test_alfa(self):
        s = pd.Series(["ALFA", "alfa"])
        result = _extract_fase_num(s)
        assert list(result) == [0, 0]

    def test_integer_input(self):
        s = pd.Series([3, 5, 7])
        result = _extract_fase_num(s)
        assert list(result) == [3, 5, 7]

    def test_nan(self):
        s = pd.Series([np.nan])
        result = _extract_fase_num(s)
        assert pd.isna(result.iloc[0])


class TestStandardizeColumns:
    def test_renames_2022_columns(self, sample_raw_2022):
        result = standardize_columns(sample_raw_2022, "PEDE2022")
        assert "defasagem" in result.columns
        assert "idade" in result.columns
        assert "inde" in result.columns
        assert "ano" in result.columns
        assert result["ano"].iloc[0] == 2022

    def test_gender_normalised(self, sample_raw_2022):
        result = standardize_columns(sample_raw_2022, "PEDE2022")
        assert set(result["genero"].dropna().unique()) <= {"Feminino", "Masculino"}

    def test_boolean_normalised(self, sample_raw_2022):
        result = standardize_columns(sample_raw_2022, "PEDE2022")
        assert set(result["ponto_virada"].dropna().unique()) <= {0, 1}

    def test_ipp_filled_when_missing(self, sample_raw_2022):
        result = standardize_columns(sample_raw_2022, "PEDE2022")
        assert "ipp" in result.columns


class TestCombineDatasets:
    def test_combines_multiple_sheets(self, sample_raw_2022):
        sheets = {"PEDE2022": sample_raw_2022}
        result = combine_datasets(sheets)
        assert len(result) == len(sample_raw_2022)
        assert "ano" in result.columns


class TestCreateTarget:
    def test_binary_target_created(self):
        df = pd.DataFrame({"defasagem": [0, 1, 2, -1, 0], "x": [1, 2, 3, 4, 5]})
        result = create_target(df)
        assert "target" in result.columns
        assert "defasagem" not in result.columns
        assert list(result["target"]) == [0, 1, 1, 0, 0]

    def test_drops_nan_defasagem(self):
        df = pd.DataFrame({"defasagem": [0, np.nan, 1], "x": [1, 2, 3]})
        result = create_target(df)
        assert len(result) == 2


class TestHandleMissing:
    def test_no_nan_after_handling(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": ["x", None, "z"],
            }
        )
        result = handle_missing(df)
        assert result.isna().sum().sum() == 0

    def test_numeric_filled_with_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = handle_missing(df)
        assert result["a"].iloc[1] == 2.0

    def test_categorical_filled(self):
        df = pd.DataFrame({"b": ["x", None, "z"]})
        result = handle_missing(df)
        assert result["b"].iloc[1] == "desconhecido"


class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self):
        pipeline = build_preprocessing_pipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "transform")


class TestSplitData:
    def test_stratified_split(self, sample_standardized_df):
        X_train, X_test, y_train, y_test = split_data(
            sample_standardized_df, strategy="stratified", test_size=0.4
        )
        assert len(X_train) + len(X_test) == len(sample_standardized_df)
        assert "target" not in X_train.columns

    def test_temporal_split(self):
        df = pd.DataFrame(
            {
                "ano": [2022, 2022, 2023, 2023, 2024, 2024],
                "x": [1, 2, 3, 4, 5, 6],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        X_train, X_test, y_train, y_test = split_data(df, strategy="temporal")
        assert len(X_train) == 4
        assert len(X_test) == 2
        assert all(X_test["ano"] == 2024)
