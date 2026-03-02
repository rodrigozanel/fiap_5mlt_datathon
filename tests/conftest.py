"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_2022():
    """DataFrame simulating raw PEDE2022 data."""
    return pd.DataFrame(
        {
            "RA": ["RA-1", "RA-2", "RA-3", "RA-4", "RA-5"],
            "Fase": [3, 5, 2, 7, 1],
            "Turma": ["A", "B", "A", "A", "C"],
            "Nome": ["A1", "A2", "A3", "A4", "A5"],
            "Ano nasc": [2008, 2006, 2010, 2003, 2012],
            "Idade 22": [14, 16, 12, 19, 10],
            "Gênero": ["Menina", "Menino", "Menina", "Menino", "Menina"],
            "Ano ingresso": [2020, 2019, 2021, 2016, 2022],
            "Instituição de ensino": ["Pub", "Pub", "Priv", "Pub", "Pub"],
            "Pedra 22": ["Ágata", "Ametista", "Quartzo", "Topázio", "Quartzo"],
            "INDE 22": [6.5, 8.1, 4.2, 9.0, 3.5],
            "IAA": [7.2, 8.0, 5.0, 9.1, 4.0],
            "IEG": [6.8, 7.5, 4.5, 8.5, 3.0],
            "IPS": [7.0, 7.8, 5.5, 8.0, 4.5],
            "IDA": [5.9, 7.0, 3.5, 8.2, 3.0],
            "Matem": [6.0, 7.5, 3.0, 8.0, 2.5],
            "Portug": [7.0, 8.0, 4.0, 8.5, 3.5],
            "Inglês": [5.5, 6.5, 3.5, 7.0, 2.0],
            "IPV": [6.0, 7.0, 4.0, 8.5, 3.0],
            "IAN": [5, 8, 3, 9, 2],
            "Indicado": ["Não", "Sim", "Não", "Sim", "Não"],
            "Atingiu PV": ["Não", "Sim", "Não", "Sim", "Não"],
            "Fase ideal": ["Fase 4", "Fase 5", "Fase 4", "Fase 8", "Fase 2"],
            "Defas": [1, 0, 2, -1, 1],
        }
    )


@pytest.fixture
def sample_standardized_df():
    """DataFrame after standardization and target creation."""
    return pd.DataFrame(
        {
            "inde": [6.5, 8.1, 4.2, 9.0, 3.5],
            "iaa": [7.2, 8.0, 5.0, 9.1, 4.0],
            "ieg": [6.8, 7.5, 4.5, 8.5, 3.0],
            "ips": [7.0, 7.8, 5.5, 8.0, 4.5],
            "ida": [5.9, 7.0, 3.5, 8.2, 3.0],
            "ipp": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "ipv": [6.0, 7.0, 4.0, 8.5, 3.0],
            "ian": [5.0, 8.0, 3.0, 9.0, 2.0],
            "nota_mat": [6.0, 7.5, 3.0, 8.0, 2.5],
            "nota_por": [7.0, 8.0, 4.0, 8.5, 3.5],
            "nota_ing": [5.5, 6.5, 3.5, 7.0, 2.0],
            "idade": [14, 16, 12, 19, 10],
            "ponto_virada": [0, 1, 0, 1, 0],
            "indicado_bolsa": [0, 1, 0, 1, 0],
            "ano": [2022, 2022, 2022, 2022, 2022],
            "target": [1, 0, 1, 0, 1],
            # Feature engineering outputs
            "media_notas": [6.17, 7.33, 3.50, 7.83, 2.67],
            "nota_min": [5.5, 6.5, 3.0, 7.0, 2.0],
            "anos_na_pm": [2, 3, 1, 6, 0],
            "fase_num": [3.0, 5.0, 2.0, 7.0, 1.0],
            "pedra_encoded": [2, 3, 1, 4, 1],
            "genero_encoded": [0, 1, 0, 1, 0],
            "indicadores_baixos": [3, 1, 5, 0, 6],
        }
    )


@pytest.fixture
def sample_student_input():
    """Sample valid student input for API testing."""
    return {
        "fase": 3,
        "idade": 14,
        "genero": "Feminino",
        "ano_ingresso": 2020,
        "inde": 6.5,
        "pedra": "Ágata",
        "iaa": 7.2,
        "ieg": 6.8,
        "ips": 7.0,
        "ida": 5.9,
        "ipp": 6.5,
        "ipv": 6.0,
        "ian": 5.5,
        "nota_mat": 6.0,
        "nota_por": 7.0,
        "nota_ing": 5.5,
        "atingiu_pv": False,
        "indicado_bolsa": False,
    }
