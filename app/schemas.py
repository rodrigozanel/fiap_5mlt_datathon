"""Pydantic schemas for API request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """Input schema for student prediction."""

    fase: int = Field(ge=0, le=8, description="Fase do aluno na Passos Magicos (0=ALFA)")
    idade: int = Field(ge=5, le=30, description="Idade do aluno")
    genero: str = Field(description="Genero: Feminino ou Masculino")
    ano_ingresso: int = Field(ge=2010, le=2026, description="Ano de ingresso na PM")
    inde: float = Field(ge=0, le=12, description="Indice de Desenvolvimento Educacional")
    pedra: str = Field(description="Classificacao PEDRA: Quartzo, Ágata, Ametista ou Topázio")
    iaa: float = Field(ge=0, le=12, description="Indicador Auto Avaliacao")
    ieg: float = Field(ge=0, le=12, description="Indicador Engajamento")
    ips: float = Field(ge=0, le=12, description="Indicador Psicossocial")
    ida: float = Field(ge=0, le=12, description="Indicador Aprendizagem")
    ipp: float = Field(default=0.0, ge=0, le=12, description="Indicador Psicopedagogico")
    ipv: float = Field(ge=0, le=12, description="Indicador Ponto de Virada")
    ian: float = Field(ge=0, le=12, description="Indicador Adequacao ao Nivel")
    nota_mat: Optional[float] = Field(default=None, ge=0, le=12)
    nota_por: Optional[float] = Field(default=None, ge=0, le=12)
    nota_ing: Optional[float] = Field(default=None, ge=0, le=12)
    atingiu_pv: bool = Field(default=False, description="Atingiu Ponto de Virada")
    indicado_bolsa: bool = Field(default=False, description="Indicado para bolsa")

    model_config = {"json_schema_extra": {
        "examples": [
            {
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
        ]
    }}


class PredictionOutput(BaseModel):
    """Output schema for prediction."""

    prediction: int = Field(description="0=sem risco, 1=com risco de defasagem")
    probability: float = Field(description="Probabilidade de defasagem (0.0-1.0)")
    risk_level: str = Field(description="Nivel de risco: baixo, medio, alto")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class LoginRequest(BaseModel):
    """Login request body."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
