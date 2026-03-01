# Tech Spec - Especificacao Tecnica

## Passos Magicos - Pipeline MLOps de Predicao de Defasagem Escolar

---

## 1. Arquitetura Geral

```
+------------------+     +------------------+     +------------------+
|   Dados Brutos   | --> |   Pipeline ML    | --> |  Modelo (.joblib)|
|  (XLSX 3 abas)   |     | preproc + feat   |     |                  |
+------------------+     |  eng + training  |     +--------+---------+
                         +------------------+              |
                                                           v
+------------------+     +------------------+     +------------------+
|    Cliente       | --> |   FastAPI        | --> |  Predict Engine  |
|  (curl/Postman)  |     |  /predict        |     |  (carrega model) |
+------------------+     |  /health         |     +------------------+
                         +------------------+              |
                                                           v
                         +------------------+     +------------------+
                         |  Streamlit       | <-- |   Logs (JSON)    |
                         |  Drift Dashboard |     |  predictions.log |
                         +------------------+     +------------------+
```

---

## 2. Stack Tecnologica

| Componente | Tecnologia | Versao | Justificativa |
|-----------|-----------|--------|--------------|
| Linguagem | Python | 3.13 | Ja configurado no pyproject.toml |
| ML Framework | scikit-learn | latest | Pipelines, preprocessors, modelos |
| Boosting | LightGBM | latest | Melhor performance em dados tabulares |
| Dados | pandas, numpy | latest | Manipulacao de dados |
| API | FastAPI + uvicorn | latest | Async, Swagger auto, Pydantic validation |
| Serializacao | joblib | latest | Mais eficiente que pickle para arrays numpy |
| Testes | pytest + pytest-cov | latest | Standard, cobertura integrada |
| Container | Docker | latest | Isolamento e reprodutibilidade |
| Monitoring | evidently + streamlit | latest | Drift detection + dashboard |
| Observabilidade | OpenTelemetry + SigNoz | latest | Traces, metricas, logs distribuidos |
| Logging | logging (stdlib) | builtin | Sem dependencia extra |

---

## 3. Estrutura do Projeto

```
passos-magicos-ml/
|
|-- app/                          # API
|   |-- __init__.py
|   |-- main.py                   # FastAPI app, startup, middleware
|   |-- routes.py                 # POST /predict, GET /health
|   |-- schemas.py                # Pydantic models (StudentInput, PredictionOutput)
|   |-- model/
|       |-- model.joblib          # Modelo treinado serializado
|       |-- preprocessor.joblib   # Pipeline de preprocessing serializada
|
|-- src/                          # Pipeline ML
|   |-- __init__.py
|   |-- preprocessing.py          # Carga, limpeza, padronizacao, split
|   |-- feature_engineering.py    # Criacao e selecao de features
|   |-- train.py                  # Treinamento, tuning, salvamento
|   |-- evaluate.py               # Metricas, plots, report
|   |-- utils.py                  # Logger config, constantes, helpers
|
|-- tests/                        # Testes unitarios
|   |-- __init__.py
|   |-- conftest.py               # Fixtures compartilhadas
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   |-- test_train.py
|   |-- test_evaluate.py
|   |-- test_api.py
|
|-- monitoring/                   # Monitoramento
|   |-- drift_dashboard.py        # Streamlit app para drift
|   |-- logger.py                 # Logging config
|
|-- notebooks/
|   |-- eda.ipynb                 # Exploracao de dados
|
|-- data/                         # Dados (gitignore em producao)
|   |-- raw/                      # Dados brutos do XLSX
|   |-- processed/                # Dados processados
|
|-- docs/                         # Documentacao
|   |-- PRD.md
|   |-- TECHSPEC.md
|
|-- signoz/                       # Configuracao SigNoz + OTel
|   |-- otel-collector-config.yaml
|   |-- otel-collector-opamp-config.yaml
|   |-- prometheus.yml
|   |-- clickhouse/
|       |-- config.xml
|       |-- users.xml
|       |-- custom-function.xml
|       |-- cluster.xml
|       |-- user_scripts/
|
|-- Dockerfile
|-- docker-compose.yml            # API + Streamlit + SigNoz stack
|-- requirements.txt
|-- pyproject.toml
|-- README.md
|-- .gitignore
```

---

## 4. Pipeline de Dados

### 4.1 Carga e Padronizacao (`src/preprocessing.py`)

**Entrada:** `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` com 3 abas

| Aba | Linhas | Colunas | Observacoes |
|-----|--------|---------|-------------|
| PEDE2022 | 861 | 42 | Coluna target: "Defas", notas: "Matem"/"Portug"/"Ingles" |
| PEDE2023 | 1014 | 48 | Coluna target: "Defasagem", notas: "Mat"/"Por"/"Ing", tem IPP |
| PEDE2024 | 1156 | 50 | Coluna target: "Defasagem", tem "Ativo/Inativo", "Escola" |

**Mapeamento de colunas entre anos:**

```python
COLUMN_MAP_2022 = {
    "Defas": "defasagem",
    "Idade 22": "idade",
    "Matem": "nota_mat",
    "Portug": "nota_por",
    "Inglês": "nota_ing",
    "INDE 22": "inde",
    "Pedra 22": "pedra",
    "Atingiu PV": "ponto_virada",
    "Fase ideal": "fase_ideal",
    "Indicado": "indicado_bolsa",
    "Fase": "fase",
    "Turma": "turma",
    "Gênero": "genero",
    "Ano ingresso": "ano_ingresso",
    "IAA": "iaa", "IEG": "ieg", "IPS": "ips",
    "IDA": "ida", "IPV": "ipv", "IAN": "ian",
    "Rec Psicologia": "rec_psicologia",
}

COLUMN_MAP_2023 = {
    "Defasagem": "defasagem",
    "Idade": "idade",
    "Mat": "nota_mat",
    "Por": "nota_por",
    "Ing": "nota_ing",
    "INDE 2023": "inde",
    "Pedra 2023": "pedra",
    "Atingiu PV": "ponto_virada",
    "Fase Ideal": "fase_ideal",
    "Indicado": "indicado_bolsa",
    "IPP": "ipp",
    # ... demais campos iguais ao 2022
}

COLUMN_MAP_2024 = {
    # Similar ao 2023, com "INDE 2024", "Pedra 2024"
    # Campos extras: "Escola", "Ativo/ Inativo"
}
```

**Processo:**
1. Carregar cada aba do XLSX
2. Renomear colunas usando os maps acima
3. Adicionar coluna `ano` (2022, 2023, 2024)
4. Selecionar apenas colunas comuns relevantes
5. Empilhar verticalmente (pd.concat) -> ~3031 registros

**Colunas a REMOVER:**
- `Nome` / `Nome Anonimizado` (identificador)
- `RA` (ID, nao feature - mas manter para cruzamento antes de dropar)
- `Avaliador1-6` (nomes dos avaliadores)
- `Rec Av1-4` / `Rec Psicologia` (texto livre - MVP sem NLP)
- `Destaque IEG/IDA/IPV` (texto livre)
- `Fase Ideal` (usado para calcular o target - DATA LEAKAGE)
- `Cg`, `Cf`, `Ct` (rankings derivados do INDE)
- `Pedra 20/21/22/23` (historico - usar apenas pedra do ano corrente)
- `INDE 22/23` (historico - usar apenas INDE do ano corrente)
- `Nº Av` (numero de avaliacoes)

### 4.2 Variavel-Alvo

```python
# Defasagem original: numerico (0, 1, 2, 3...)
# Target binario (confirmado): 1 se aluno tem defasagem, 0 se nao
df["target"] = (df["defasagem"] > 0).astype(int)

# Remover coluna defasagem original (evitar leakage)
df = df.drop(columns=["defasagem"])
```

### 4.3 Feature Engineering (`src/feature_engineering.py`)

**Features numericas diretas:**

| Feature | Origem | Tipo |
|---------|--------|------|
| inde | INDE do ano | float |
| iaa | Indicador Auto Avaliacao | float |
| ieg | Indicador Engajamento | float |
| ips | Indicador Psicossocial | float |
| ida | Indicador Aprendizagem | float |
| ipp | Indicador Psicopedagogico | float |
| ipv | Indicador Ponto de Virada | float |
| ian | Indicador Adequacao ao Nivel | float |
| nota_mat | Nota de Matematica | float |
| nota_por | Nota de Portugues | float |
| nota_ing | Nota de Ingles | float |
| idade | Idade do aluno | int |

**Features derivadas:**

| Feature | Calculo | Justificativa |
|---------|---------|--------------|
| media_notas | (mat + por + ing) / 3 | Performance academica geral |
| nota_min | min(mat, por, ing) | Identifica ponto fraco |
| variancia_notas | var(mat, por, ing) | Dispersao entre disciplinas |
| anos_na_pm | ano_atual - ano_ingresso | Tempo no programa |
| fase_num | Extrair numero da Fase | Nivel numerico |
| pedra_encoded | Quartzo=1, Agata=2, Ametista=3, Topazio=4 | Ordinal encoding |
| indicadores_baixos | count(indicadores < mediana) | Quantidade de areas fracas |

**Features categoricas/booleanas:**

| Feature | Encoding |
|---------|---------|
| genero | LabelEncoder (Masculino=0, Feminino=1) |
| ponto_virada | Binario 0/1 |
| indicado_bolsa | Binario 0/1 |

**Features a NAO usar (risco de leakage):**
- `fase_ideal` - diretamente usado para calcular defasagem
- `INDE` vs sub-indicadores - INDE e a ponderacao dos sub-indicadores. Testar usar um ou outro, nao ambos

### 4.4 Preprocessing Pipeline (sklearn)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

numeric_features = [
    "inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian",
    "nota_mat", "nota_por", "nota_ing", "media_notas",
    "nota_min", "idade", "anos_na_pm", "fase_num"
]

categorical_features = ["genero", "pedra_encoded"]
boolean_features = ["ponto_virada", "indicado_bolsa"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
    ("bool", "passthrough", boolean_features)
])
```

### 4.5 Split de Dados

**Estrategia principal: Split temporal**
- Treino: PEDE2022 + PEDE2023 (~1875 registros)
- Teste: PEDE2024 (~1156 registros)
- Simula cenario real: prever defasagem futura com dados passados

**Estrategia alternativa: Split estratificado**
- 80% treino / 20% teste, estratificado pelo target
- Mais dados para treino, mas menos realista

---

## 5. Treinamento (`src/train.py`)

### Modelos Candidatos

| Modelo | Vantagem | Desvantagem |
|--------|----------|-------------|
| **LightGBM** (favorito) | Melhor performance tabular, rapido | Mais hiperparametros |
| Random Forest | Robusto, poucos hiperparametros | Menos preciso |
| Logistic Regression | Interpretavel, baseline | Relacoes lineares apenas |

### Validacao

- **StratifiedKFold** com `n_splits=5`
- **Metrica principal:** F1-Score (weighted) - equilibra precision e recall para classes desbalanceadas
- **Metricas secundarias:** Accuracy, AUC-ROC, Precision, Recall

### Hyperparameter Tuning

```python
# LightGBM - Grid Search
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 50],
}
```

### Salvamento

```python
import joblib

# Salvar pipeline completa (preprocessor + modelo)
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", best_model)
])
joblib.dump(full_pipeline, "app/model/model.joblib")
```

---

## 6. Avaliacao (`src/evaluate.py`)

### Metricas Reportadas

```python
from sklearn.metrics import (
    f1_score,           # METRICA PRINCIPAL
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
```

### Plots Gerados

1. **Confusion Matrix** - heatmap com contagens
2. **ROC Curve** - AUC-ROC score
3. **Feature Importance** - top 15 features (LightGBM built-in)
4. **Precision-Recall Curve** - util para classes desbalanceadas

### Justificativa para Producao

Documentar no README:
- F1-Score >= 0.70 (threshold minimo aceitavel)
- Cross-validation mostra estabilidade (baixo desvio padrao entre folds)
- Analise de erros: tipos de alunos mais dificeis de classificar
- Comparacao com baseline (modelo mais simples)

---

## 7. API (`app/`)

### Endpoints

| Metodo | Path | Descricao | Request Body | Response |
|--------|------|-----------|-------------|----------|
| GET | `/health` | Health check | - | `{"status": "healthy"}` |
| POST | `/predict` | Predicao individual | `StudentInput` | `PredictionOutput` |
| GET | `/docs` | Swagger UI (auto FastAPI) | - | HTML |

### Schemas Pydantic (`app/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional

class StudentInput(BaseModel):
    fase: int = Field(ge=1, le=8, description="Fase do aluno na Passos Magicos")
    idade: int = Field(ge=6, le=25, description="Idade do aluno")
    genero: str = Field(pattern="^(Masculino|Feminino)$")
    ano_ingresso: int = Field(ge=2010, le=2024)
    inde: float = Field(ge=0, le=10, description="Indice de Desenvolvimento Educacional")
    pedra: str = Field(pattern="^(Quartzo|Ágata|Ametista|Topázio)$")
    iaa: float = Field(ge=0, le=10, description="Indicador Auto Avaliacao")
    ieg: float = Field(ge=0, le=10, description="Indicador Engajamento")
    ips: float = Field(ge=0, le=10, description="Indicador Psicossocial")
    ida: float = Field(ge=0, le=10, description="Indicador Aprendizagem")
    ipp: float = Field(ge=0, le=10, description="Indicador Psicopedagogico")
    ipv: float = Field(ge=0, le=10, description="Indicador Ponto de Virada")
    ian: float = Field(ge=0, le=10, description="Indicador Adequacao ao Nivel")
    nota_mat: Optional[float] = Field(default=None, ge=0, le=10)
    nota_por: Optional[float] = Field(default=None, ge=0, le=10)
    nota_ing: Optional[float] = Field(default=None, ge=0, le=10)
    atingiu_pv: bool = False
    indicado_bolsa: bool = False

class PredictionOutput(BaseModel):
    prediction: int      # 0 (sem risco) ou 1 (com risco)
    probability: float   # 0.0 a 1.0
    risk_level: str      # "baixo", "medio", "alto"
```

### Fluxo de Predicao

```python
# app/routes.py - logica do /predict
@router.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    # 1. Converter input em DataFrame
    input_df = pd.DataFrame([student.model_dump()])

    # 2. Aplicar feature engineering (media_notas, anos_na_pm, etc.)
    input_df = create_derived_features(input_df)

    # 3. Predict com pipeline completa (preprocessor + modelo)
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.5)

    # 4. Classificar nivel de risco
    if probability < 0.3:
        risk_level = "baixo"
    elif probability < 0.6:
        risk_level = "medio"
    else:
        risk_level = "alto"

    # 5. Logar predicao
    log_prediction(student, prediction, probability)

    return PredictionOutput(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level
    )
```

### Exemplo de Chamada

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "atingiu_pv": false,
    "indicado_bolsa": false
  }'

# Resposta:
# {"prediction": 1, "probability": 0.73, "risk_level": "alto"}
```

---

## 8. Docker

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar OpenTelemetry auto-instrumentation
RUN pip install opentelemetry-distro opentelemetry-exporter-otlp
RUN opentelemetry-bootstrap --action=install

COPY app/ app/
COPY src/ src/
COPY monitoring/ monitoring/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Usar opentelemetry-instrument para auto-instrumentacao
CMD ["opentelemetry-instrument", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (API + Dashboard + SigNoz Stack)

```yaml
x-logging: &default-logging
  options:
    max-size: 50m
    max-file: "3"

x-clickhouse-defaults: &clickhouse-defaults
  restart: unless-stopped
  logging: *default-logging
  image: clickhouse/clickhouse-server:24.8.5-alpine
  tty: true
  healthcheck:
    test: ["CMD", "wget", "--spider", "-q", "0.0.0.0:8123/ping"]
    interval: 30s
    timeout: 5s
    retries: 3
  ulimits:
    nproc: 65535
    nofile:
      soft: 262144
      hard: 262144

services:
  # ==========================================================================
  # Passos Magicos ML API
  # ==========================================================================
  app:
    build: .
    container_name: passos-magicos-api
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./app/model:/app/app/model
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
      # OpenTelemetry configuration for SigNoz
      - OTEL_SERVICE_NAME=passos-magicos-ml
      - OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
      - OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://signoz-otel-collector:4318}
      - OTEL_LOG_LEVEL=info
    command: >
      opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000
    networks:
      - app-network
    restart: unless-stopped
    depends_on:
      signoz-otel-collector:
        condition: service_started

  # Drift monitoring dashboard
  dashboard:
    build: .
    container_name: passos-magicos-dashboard
    command: streamlit run monitoring/drift_dashboard.py --server.port 8501
    ports:
      - "8501:8501"
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network
    depends_on:
      - app

  # ==========================================================================
  # SigNoz Observability Stack
  # ==========================================================================

  signoz-zookeeper:
    image: zookeeper:3.9
    container_name: signoz-zookeeper
    restart: unless-stopped
    logging: *default-logging
    volumes:
      - signoz-zookeeper-data:/data
      - signoz-zookeeper-datalog:/datalog
    environment:
      - ZOO_MY_ID=1
      - ZOO_4LW_COMMANDS_WHITELIST=ruok,stat
    healthcheck:
      test: ["CMD-SHELL", "echo ruok | nc localhost 2181 | grep imok"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - app-network

  signoz-init-clickhouse:
    <<: *clickhouse-defaults
    container_name: signoz-init-clickhouse
    command:
      - bash
      - -c
      - |
        version="v0.0.1"
        node_os=$$(uname -s | tr '[:upper:]' '[:lower:]')
        node_arch=$$(uname -m | sed s/aarch64/arm64/ | sed s/x86_64/amd64/)
        cd /tmp
        wget -O histogram-quantile.tar.gz "https://github.com/SigNoz/signoz/releases/download/histogram-quantile%2F$${version}/histogram-quantile_$${node_os}_$${node_arch}.tar.gz"
        tar -xvzf histogram-quantile.tar.gz
        mv histogram-quantile /var/lib/clickhouse/user_scripts/histogramQuantile
    restart: on-failure
    volumes:
      - ./signoz/clickhouse/user_scripts:/var/lib/clickhouse/user_scripts/
    networks:
      - app-network

  signoz-clickhouse:
    <<: *clickhouse-defaults
    container_name: signoz-clickhouse
    depends_on:
      signoz-init-clickhouse:
        condition: service_completed_successfully
      signoz-zookeeper:
        condition: service_healthy
    volumes:
      - ./signoz/clickhouse/config.xml:/etc/clickhouse-server/config.xml
      - ./signoz/clickhouse/users.xml:/etc/clickhouse-server/users.xml
      - ./signoz/clickhouse/custom-function.xml:/etc/clickhouse-server/custom-function.xml
      - ./signoz/clickhouse/user_scripts:/var/lib/clickhouse/user_scripts/
      - ./signoz/clickhouse/cluster.xml:/etc/clickhouse-server/config.d/cluster.xml
      - signoz-clickhouse-data:/var/lib/clickhouse/
    environment:
      - CLICKHOUSE_SKIP_USER_SETUP=1
    networks:
      - app-network

  signoz-schema-migrator-sync:
    image: signoz/signoz-schema-migrator:${SIGNOZ_OTELCOL_TAG:-v0.129.8}
    container_name: signoz-schema-migrator-sync
    restart: on-failure
    logging: *default-logging
    command: [sync, --dsn=tcp://signoz-clickhouse:9000, --up=]
    depends_on:
      signoz-clickhouse:
        condition: service_healthy
    networks:
      - app-network

  signoz-schema-migrator-async:
    image: signoz/signoz-schema-migrator:${SIGNOZ_OTELCOL_TAG:-v0.129.8}
    container_name: signoz-schema-migrator-async
    restart: on-failure
    logging: *default-logging
    command: [async, --dsn=tcp://signoz-clickhouse:9000, --up=]
    depends_on:
      signoz-clickhouse:
        condition: service_healthy
      signoz-schema-migrator-sync:
        condition: service_completed_successfully
    networks:
      - app-network

  signoz:
    image: signoz/signoz:${SIGNOZ_VERSION:-v0.101.0}
    container_name: signoz
    restart: unless-stopped
    logging: *default-logging
    command: [--config=/root/config/prometheus.yml]
    ports:
      - "8080:8080"  # SigNoz UI
    volumes:
      - ./signoz/prometheus.yml:/root/config/prometheus.yml
      - ./signoz/dashboards:/root/config/dashboards
      - signoz-sqlite-data:/var/lib/signoz/
    environment:
      - SIGNOZ_ALERTMANAGER_PROVIDER=signoz
      - SIGNOZ_TELEMETRYSTORE_CLICKHOUSE_DSN=tcp://signoz-clickhouse:9000
      - SIGNOZ_SQLSTORE_SQLITE_PATH=/var/lib/signoz/signoz.db
      - DASHBOARDS_PATH=/root/config/dashboards
      - STORAGE=clickhouse
      - GODEBUG=netdns=go
      - TELEMETRY_ENABLED=true
      - DEPLOYMENT_TYPE=docker-standalone-amd
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "localhost:8080/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    depends_on:
      signoz-clickhouse:
        condition: service_healthy
      signoz-schema-migrator-sync:
        condition: service_completed_successfully
    networks:
      - app-network

  signoz-otel-collector:
    image: signoz/signoz-otel-collector:${SIGNOZ_OTELCOL_TAG:-v0.129.8}
    container_name: signoz-otel-collector
    restart: unless-stopped
    logging: *default-logging
    command:
      - --config=/etc/otel-collector-config.yaml
      - --feature-gates=-pkg.translator.prometheus.NormalizeName
    volumes:
      - ./signoz/otel-collector-config.yaml:/etc/otel-collector-config.yaml
    environment:
      - OTEL_RESOURCE_ATTRIBUTES=host.name=signoz-host,os.type=linux
      - LOW_CARDINAL_EXCEPTION_GROUPING=false
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    depends_on:
      signoz:
        condition: service_healthy
    networks:
      - app-network

volumes:
  signoz-zookeeper-data:
  signoz-zookeeper-datalog:
  signoz-clickhouse-data:
  signoz-sqlite-data:

networks:
  app-network:
    driver: bridge
```

### Comandos

```bash
# Subir tudo (API + Dashboard + SigNoz)
docker-compose up -d

# Verificar servicos
docker-compose ps
curl http://localhost:8000/health

# Acessar
# API:       http://localhost:8000
# Swagger:   http://localhost:8000/docs
# SigNoz:    http://localhost:8080
# Dashboard: http://localhost:8501

# Logs
docker-compose logs app
docker-compose logs signoz-otel-collector

# Parar
docker-compose down

# Parar e limpar volumes
docker-compose down -v
```

---

## 9. Observabilidade (OpenTelemetry + SigNoz)

### Arquitetura de Observabilidade

```
+-------------------+        +---------------------+       +------------------+
| FastAPI App       | -----> | OTel Collector      | ----> | ClickHouse       |
| (auto-instrument) | OTLP   | (signoz-otel-       | TCP   | (signoz-         |
| + custom spans    | HTTP   |  collector:4318)    |       |  clickhouse:9000)|
+-------------------+        +---------------------+       +--------+---------+
                                                                     |
                                                            +--------v---------+
                                                            | SigNoz UI        |
                                                            | (localhost:8080)  |
                                                            | Traces, Metrics,  |
                                                            | Logs, Dashboards  |
                                                            +------------------+
```

### Instrumentacao da API

**Auto-instrumentacao** (via `opentelemetry-instrument` no CMD do Docker):
- FastAPI: traces automaticos para cada request HTTP
- HTTP clients: traces de chamadas externas

**Instrumentacao manual** (spans customizados no pipeline de predicao):

```python
from opentelemetry import trace

tracer = trace.get_tracer("prediction", "1.0.0")

@router.post("/predict")
def predict(student: StudentInput):
    with tracer.start_as_current_span("prediction.pipeline") as span:
        span.set_attribute("student.fase", student.fase)
        span.set_attribute("student.idade", student.idade)

        with tracer.start_as_current_span("prediction.validate_input"):
            # validacao dos dados
            ...

        with tracer.start_as_current_span("prediction.preprocess"):
            # preprocessamento
            ...

        with tracer.start_as_current_span("prediction.model_inference"):
            probability = model.predict_proba(input_df)[0][1]
            span.set_attribute("prediction.probability", probability)
            span.set_attribute("prediction.risk_level", risk_level)

        return PredictionOutput(...)
```

### Configuracao do OTel Collector

Arquivo: `signoz/otel-collector-config.yaml`

**Receivers:** OTLP (gRPC :4317, HTTP :4318) + Prometheus scraping
**Processors:** Batch (10k spans, 10s timeout) + SpanMetrics (latency histograms)
**Exporters:** ClickHouse (traces, metrics, logs em databases separados)
**Pipelines:**
- Traces: otlp -> spanmetrics + batch -> clickhouse
- Metrics: otlp -> batch -> clickhouse
- Logs: otlp -> batch -> clickhouse

### Metricas Customizadas (Prometheus)

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("http_requests_total", "Total requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["method", "path"])
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["risk_level"])
PREDICTION_LATENCY = Histogram("prediction_duration_seconds", "Prediction latency")
```

### Portas e Acesso

| Servico | Porta | URL |
|---------|-------|-----|
| API (FastAPI) | 8000 | http://localhost:8000 |
| SigNoz UI | 8080 | http://localhost:8080 |
| Drift Dashboard | 8501 | http://localhost:8501 |
| OTel Collector gRPC | 4317 | - |
| OTel Collector HTTP | 4318 | - |

### Dependencias OTel (no Dockerfile)

```
opentelemetry-distro
opentelemetry-exporter-otlp
# opentelemetry-bootstrap instala automaticamente:
#   opentelemetry-instrumentation-fastapi
#   opentelemetry-instrumentation-requests
#   opentelemetry-instrumentation-httpx
```

### Arquivos de Configuracao SigNoz (copiados de fiap_5mlt_tc4)

```
signoz/
|-- otel-collector-config.yaml       # Config do collector (receivers, processors, exporters)
|-- otel-collector-opamp-config.yaml # OpAmp connection to SigNoz
|-- prometheus.yml                   # Prometheus remote read config
|-- clickhouse/
    |-- config.xml                   # ClickHouse server config
    |-- users.xml                    # ClickHouse users/permissions
    |-- custom-function.xml          # histogramQuantile UDF
    |-- cluster.xml                  # ZooKeeper + cluster config
    |-- user_scripts/
        |-- histogramQuantile        # Binary for histogram quantiles
```

---

## 10. Monitoramento

### Logging (`monitoring/logger.py`)

```python
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

def setup_logger():
    logger = logging.getLogger("predictions")
    handler = RotatingFileHandler(
        "logs/predictions.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    logger.addHandler(handler)
    return logger

def log_prediction(input_data, prediction, probability, latency_ms):
    logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "latency_ms": latency_ms,
    }))
```

### Drift Dashboard (`monitoring/drift_dashboard.py`)

```python
import streamlit as st
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

st.title("Monitoramento de Drift - Passos Magicos ML")

# Carregar dados de referencia (treino)
reference_data = pd.read_csv("data/processed/train.csv")

# Carregar dados de producao (logs parseados)
production_data = parse_prediction_logs("logs/predictions.log")

# Gerar report de drift
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=production_data)

# Exibir no Streamlit
st.components.v1.html(report.get_html(), height=800, scrolling=True)
```

**Metricas monitoradas:**
- PSI (Population Stability Index) por feature
- KS-test para features numericas
- Chi-square para features categoricas
- Prediction drift (distribuicao das predicoes)

**Acesso:** `http://localhost:8501`

---

## 10. Testes

### Estrutura

```
tests/
|-- conftest.py               # Fixtures compartilhadas
|-- test_preprocessing.py     # 8+ testes
|-- test_feature_engineering.py  # 6+ testes
|-- test_train.py             # 4+ testes
|-- test_evaluate.py          # 4+ testes
|-- test_api.py               # 5+ testes
```

### Fixtures (`tests/conftest.py`)

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_raw_data():
    """DataFrame simulando dados brutos de uma aba do XLSX."""
    return pd.DataFrame({
        "Fase": [3, 5, 2],
        "Idade": [14, 16, 12],
        "Gênero": ["Feminino", "Masculino", "Feminino"],
        "Ano ingresso": [2020, 2019, 2021],
        "INDE 2023": [6.5, 8.1, 4.2],
        "Pedra 2023": ["Ágata", "Ametista", "Quartzo"],
        "IAA": [7.2, 8.0, 5.0],
        "IEG": [6.8, 7.5, 4.5],
        # ... demais campos
        "Defasagem": [1, 0, 2],
    })

@pytest.fixture
def trained_model():
    """Carrega modelo treinado para testes."""
    import joblib
    return joblib.load("app/model/model.joblib")

@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
```

### Testes por Modulo

| Modulo | Teste | O que valida |
|--------|-------|-------------|
| preprocessing | `test_load_data` | Carrega XLSX sem erro, retorna dict com 3 DataFrames |
| preprocessing | `test_standardize_columns` | Colunas renomeadas corretamente |
| preprocessing | `test_combine_datasets` | Concat funciona, coluna 'ano' existe |
| preprocessing | `test_create_target` | Target binario, sem NaN no target |
| preprocessing | `test_handle_missing` | Sem NaN apos imputacao |
| preprocessing | `test_split_data` | Proporcoes corretas, estratificacao mantida |
| feature_eng | `test_media_notas` | (mat+por+ing)/3 calculado corretamente |
| feature_eng | `test_pedra_encoding` | Quartzo=1, Agata=2, Ametista=3, Topazio=4 |
| feature_eng | `test_anos_na_pm` | ano_atual - ano_ingresso correto |
| train | `test_train_returns_model` | Retorna objeto com .predict e .predict_proba |
| train | `test_save_load_roundtrip` | Salvar e carregar preserva predicoes |
| evaluate | `test_metrics_range` | F1 entre 0 e 1, accuracy entre 0 e 1 |
| evaluate | `test_confusion_matrix_shape` | Matriz 2x2 |
| api | `test_health_endpoint` | GET /health retorna 200 |
| api | `test_predict_valid_input` | POST /predict retorna 200 com campos corretos |
| api | `test_predict_invalid_input` | Input invalido retorna 422 |
| api | `test_predict_response_schema` | Resposta tem prediction, probability, risk_level |

### Comando para Rodar

```bash
# Rodar todos os testes com cobertura
pytest tests/ -v --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80

# Rodar testes especificos
pytest tests/test_api.py -v

# Gerar report HTML de cobertura
pytest tests/ --cov=src --cov=app --cov-report=html
```

---

## 12. Dependencias (`requirements.txt`)

```
# ML
scikit-learn>=1.4
lightgbm>=4.0
pandas>=2.0
numpy>=1.26
joblib>=1.3

# API
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0

# Observabilidade (OTel + Prometheus)
opentelemetry-distro
opentelemetry-exporter-otlp
prometheus-client>=0.17

# Monitoramento de Drift
evidently>=0.4
streamlit>=1.30

# Testes
pytest>=8.0
pytest-cov>=5.0
httpx>=0.27  # para TestClient do FastAPI

# Dados
openpyxl>=3.1  # para ler XLSX

# Logging
loguru>=0.7
```

---

## 13. Riscos e Mitigacoes

| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Classes muito desbalanceadas | Media | Alto | SMOTE, class_weight='balanced', threshold tuning |
| Data leakage via Fase Ideal/IAN | Alta | Critico | Remover Fase Ideal, avaliar IAN cuidadosamente |
| Poucos dados (~3000 linhas) | Alta | Medio | Cross-validation, regularizacao, evitar overfitting |
| Colunas inconsistentes entre anos | Alta | Medio | Mapeamento explicito de colunas |
| IPP ausente em 2022 | Media | Baixo | Imputar com mediana ou usar apenas 2023+2024 |
| Valores missing nas notas | Media | Baixo | Imputacao com mediana |
