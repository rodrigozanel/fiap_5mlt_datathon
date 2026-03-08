"""Feast feature definitions for Passos Magicos."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# -- Entity: a student identified by their RA (registro academico) + year --
student = Entity(
    name="student",
    join_keys=["student_id"],
    description="Student identified by RA + year combination",
)

# -- Source: parquet file with pre-computed features --
student_features_source = FileSource(
    path="/app/feature_store/data/student_features.parquet",
    timestamp_field="event_timestamp",
)

# -- Feature View: academic indicators and engineered features --
student_features_view = FeatureView(
    name="student_features",
    entities=[student],
    ttl=timedelta(days=365),
    schema=[
        # Raw indicators
        Field(name="inde", dtype=Float64),
        Field(name="iaa", dtype=Float64),
        Field(name="ieg", dtype=Float64),
        Field(name="ips", dtype=Float64),
        Field(name="ida", dtype=Float64),
        Field(name="ipp", dtype=Float64),
        Field(name="ipv", dtype=Float64),
        Field(name="ian", dtype=Float64),
        Field(name="nota_mat", dtype=Float64),
        Field(name="nota_por", dtype=Float64),
        Field(name="nota_ing", dtype=Float64),
        Field(name="idade", dtype=Float64),
        Field(name="ponto_virada", dtype=Int64),
        Field(name="indicado_bolsa", dtype=Int64),
        Field(name="ano", dtype=Int64),
        # Engineered features
        Field(name="media_notas", dtype=Float64),
        Field(name="nota_min", dtype=Float64),
        Field(name="anos_na_pm", dtype=Float64),
        Field(name="fase_num", dtype=Float64),
        Field(name="pedra_encoded", dtype=Int64),
        Field(name="genero_encoded", dtype=Int64),
        Field(name="indicadores_baixos", dtype=Int64),
    ],
    source=student_features_source,
    online=True,
)
