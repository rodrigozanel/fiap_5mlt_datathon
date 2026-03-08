# Passos Magicos - Predicao de Defasagem Escolar

Modelo preditivo de risco de defasagem escolar para a Associacao Passos Magicos, desenvolvido para o Datathon Pos Tech FIAP.

## Visao Geral

A Associacao Passos Magicos atua ha 32 anos transformando a vida de criancas e jovens de baixa renda em Embu-Guacu/SP. Este projeto implementa um sistema de Machine Learning que identifica precocemente estudantes em risco de defasagem escolar, permitindo intervencao direcionada pela equipe pedagogica.

**Problema:** Identificar quais alunos estao em risco de defasagem escolar antes que ela ocorra.

**Solucao:** Modelo de classificacao binaria que recebe indicadores educacionais de um aluno e retorna a probabilidade de defasagem, exposto via API REST.

**Stack:**
- Python 3.13
- scikit-learn + LightGBM + XGBoost
- FastAPI + Uvicorn
- Docker + Docker Compose + NGINX
- MLflow (experiment tracking)
- Feast (Feature Store)
- OpenTelemetry + SigNoz (observabilidade)
- Streamlit + Evidently (monitoramento de drift)

---

## Arquitetura da Solucao

```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                     Docker Compose Network                       │
                                    │                                                                 │
  ┌──────────┐    ┌──────────┐      │  ┌─────────────────────────────────────────────────────────┐    │
  │          │    │          │      │  │                    NGINX :80                             │    │
  │ Browser  │───▶│  Porta   │─────▶│  │  /          ──▶  Web UI                                 │    │
  │          │    │  80/3000 │      │  │  /api/v1/*  ──▶  FastAPI                                │    │
  └──────────┘    └──────────┘      │  │  /docs      ──▶  Swagger UI                             │    │
                                    │  │  /mlflow/   ──▶  MLflow                                 │    │
                                    │  │  /drift/    ──▶  Streamlit                              │    │
                                    │  │  /signoz/   ──▶  SigNoz                                 │    │
                                    │  └──────┬──────────────┬───────────────┬──────────────┬─────┘    │
                                    │         │              │               │              │          │
                                    │         ▼              ▼               ▼              ▼          │
                                    │  ┌─────────────┐ ┌──────────┐  ┌───────────┐  ┌───────────┐     │
                                    │  │  Web UI      │ │ FastAPI  │  │  Drift    │  │  MLflow   │     │
                                    │  │  :3000       │ │ API :8000│  │ Dashboard │  │  :5000    │     │
                                    │  │             │ │          │  │  :8501    │  │           │     │
                                    │  │  - Auth form │ │ JWT Auth │  │ Streamlit │  │ Tracking  │     │
                                    │  │  - Predicao  │ │ /predict │  │ Evidently │  │ Artifacts │     │
                                    │  │  - Historico │ │ /health  │  │           │  │ Metricas  │     │
                                    │  └──────┬──────┘ └────┬─────┘  └─────┬─────┘  └─────┬─────┘     │
                                    │         │             │              │              │            │
                                    │         │        ┌────┴────┐    ┌────┴────┐    ┌────┴────┐       │
                                    │         │        │  Model  │    │  Logs   │    │ SQLite  │       │
                                    │         │        │ .joblib │    │  JSON   │    │  + Vol  │       │
                                    │         │        └────┬────┘    └─────────┘    └─────────┘       │
                                    │         │             │                                          │
                                    │    ┌────┴─────────────┴──────────────────────────────────┐       │
                                    │    │              OpenTelemetry (auto-instrumentation)    │       │
                                    │    │  Spans: model.load, predict, auth.login             │       │
                                    │    │  Metrics: predictions.total, latency, risk_level    │       │
                                    │    └────────────────────────┬────────────────────────────┘       │
                                    │                             │ OTLP/HTTP :4318                    │
                                    │                             ▼                                    │
                                    │                   ┌──────────────────┐                           │
                                    │                   │   OTel Collector │                           │
                                    │                   └────────┬─────────┘                           │
                                    │                            │                                     │
                                    │                            ▼                                     │
                                    │  ┌──────────────────────────────────────────────────────────┐    │
                                    │  │                    SigNoz Stack                          │    │
                                    │  │  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐ │    │
                                    │  │  │ SigNoz   │  │ ClickHouse │  │ Zookeeper            │ │    │
                                    │  │  │ UI :8080 │  │ (storage)  │  │ (coordination)       │ │    │
                                    │  │  │ Traces   │  │ Traces     │  └──────────────────────┘ │    │
                                    │  │  │ Metrics  │  │ Metrics    │  ┌──────────────────────┐ │    │
                                    │  │  │ Logs     │  │ Logs       │  │ Schema Migrators     │ │    │
                                    │  │  │ Dashb.   │  │            │  │ (sync + async)       │ │    │
                                    │  │  └──────────┘  └────────────┘  └──────────────────────┘ │    │
                                    │  └──────────────────────────────────────────────────────────┘    │
                                    │                                                                 │
                                    │  ┌──────────────────────────────────────────────────────────┐    │
                                    │  │                 Training Pipeline (one-shot)             │    │
                                    │  │                                                          │    │
                                    │  │  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌───────┐ │    │
                                    │  │  │  XLSX   │──▶│ Preproc.  │──▶│ Feature  │──▶│ Train │ │    │
                                    │  │  │  Data   │   │ & Clean   │   │ Engineer │   │ 4 mod │ │    │
                                    │  │  └─────────┘   └───────────┘   └────┬─────┘   └───┬───┘ │    │
                                    │  │                                     │             │     │    │
                                    │  │                              ┌──────┴──────┐      │     │    │
                                    │  │                              │ Feast Store │      │     │    │
                                    │  │                              │ (opcional)  │      │     │    │
                                    │  │                              └─────────────┘      │     │    │
                                    │  │                                           ┌──────┴───┐ │    │
                                    │  │                                           │ Evaluate │ │    │
                                    │  │         Logs to MLflow ◀──────────────────│ Compare  │ │    │
                                    │  │         (params, metrics, artifacts)      │ Save best│ │    │
                                    │  │                                           └──────────┘ │    │
                                    │  └──────────────────────────────────────────────────────────┘    │
                                    └─────────────────────────────────────────────────────────────────┘
```

**Fluxo principal:**
1. **Treino** — O pipeline carrega dados do XLSX (ou Feast), treina 4 modelos (LightGBM, XGBoost, RF, LR), registra metricas no MLflow e salva o melhor como `model.joblib`
2. **Servico** — A API FastAPI carrega o modelo, recebe requests autenticados via JWT, aplica feature engineering e retorna a predicao com nivel de risco
3. **Observabilidade** — Cada request gera spans e metricas OTel enviados ao SigNoz via OTel Collector, com dashboards predefinidos para ML e saude da API
4. **Monitoramento** — O Drift Dashboard (Streamlit) analisa distribuicao de features e detecta desvios nos dados de producao

---

## Estrutura do Projeto

```
passos-magicos-ml/
|-- app/                          # API FastAPI
|   |-- main.py                   # Entry point, carrega modelo no startup
|   |-- routes.py                 # POST /api/v1/predict, GET /api/v1/health, POST /api/v1/auth/login
|   |-- schemas.py                # Pydantic: StudentInput, PredictionOutput, LoginRequest, TokenResponse
|   |-- auth.py                   # JWT: criacao e verificacao de tokens
|   |-- telemetry.py              # OpenTelemetry: tracers, meters, metricas customizadas
|   |-- model/
|       |-- model.joblib           # Modelo treinado serializado
|
|-- src/                          # Pipeline de ML
|   |-- preprocessing.py          # Carga, limpeza, padronizacao, split
|   |-- feature_engineering.py    # Criacao e encoding de features
|   |-- train.py                  # Treinamento, tuning, salvamento
|   |-- evaluate.py               # Metricas, reports, comparacao de modelos
|   |-- utils.py                  # Logger, constantes, paths
|
|-- scripts/                      # Scripts utilitarios
|   |-- train_pipeline.py         # Pipeline completo de treinamento com MLflow
|   |-- materialize_features.py   # Materializa features no Feast Feature Store
|   |-- provision_dashboards.sh   # Provisiona dashboards no SigNoz via API
|
|-- tests/                        # Testes unitarios (86% cobertura)
|   |-- conftest.py               # Fixtures compartilhadas
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   |-- test_train.py
|   |-- test_evaluate.py
|   |-- test_api.py
|
|-- feature_store/                # Feast Feature Store (opcional)
|   |-- feature_store.yaml        # Configuracao do Feast (provider: local)
|   |-- definitions.py            # Entidades e FeatureViews
|   |-- data/                     # Parquet offline + SQLite online store
|
|-- monitoring/                   # Monitoramento
|   |-- drift_dashboard.py        # Dashboard Streamlit de drift
|   |-- logger.py                 # Logging estruturado JSON
|
|-- webapp/                       # Interface web
|   |-- server.py                 # Servidor FastAPI para a UI
|   |-- static/index.html         # Single-page app (auth, predicao, historico)
|   |-- Dockerfile
|
|-- nginx/                        # Reverse proxy (producao)
|   |-- nginx.conf                # Rotas: /, /api, /docs, /drift, /mlflow, /signoz
|
|-- signoz/                       # Configuracoes SigNoz/OTel
|   |-- dashboards/               # Dashboards predefinidos (JSON)
|   |-- otel-collector-config.yaml
|   |-- clickhouse/               # Configuracao do ClickHouse
|
|-- data/                         # Dados brutos e processados
|-- docs/                         # PRD e Tech Spec
|-- postman/                      # Colecao Postman
|-- Dockerfile
|-- docker-compose.yml            # Desenvolvimento
|-- docker-compose.prod.yml       # Producao (todos os servicos + NGINX)
|-- pyproject.toml
|-- requirements.txt
```

---

## Instrucoes de Deploy

### Pre-requisitos
- **Docker e Docker Compose** - unico requisito obrigatorio
- Python 3.13+ com Poetry (opcional, apenas para desenvolvimento local)

> **Nota:** Nao e necessario instalar Python, LightGBM ou qualquer dependencia localmente. Tudo roda dentro do Docker (Linux), onde o LightGBM funciona nativamente.

### Passo a Passo Completo via Docker (do zero ate a API rodando)

Sao apenas **4 comandos**:

#### 1. Copiar o dataset para `data/raw/`

```bash
cp "drive-download-20260301T222743Z-1-001/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" data/raw/
```

#### 2. Configurar variaveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e defina pelo menos o `JWT_SECRET_KEY` com um valor seguro:

```bash
# .env
JWT_SECRET_KEY=sua-chave-secreta-forte-aqui
API_USERNAME=admin
API_PASSWORD=sua-senha-aqui
```

> **Importante:** Nunca use o valor padrao `changeme-secret-key-for-development` em producao.

#### 3. Build da imagem Docker

```bash
docker compose build
```

Isso cria a imagem com Python 3.13, LightGBM, scikit-learn, FastAPI, MLflow e todas as dependencias. Leva ~2-3 minutos na primeira vez.

#### 4. Subir o MLflow e treinar o modelo

```bash
# Subir o MLflow tracking server
docker compose up mlflow -d

# Treinar (dentro do Docker)
docker compose --profile train run --rm train
```

O comando de treino sobe um container temporario que:
1. **Carrega** as 3 abas do XLSX (PEDE2022, PEDE2023, PEDE2024) - ~3000 registros
2. **Padroniza** os nomes de colunas (diferem entre anos: "Defas" vs "Defasagem", "Matem" vs "Mat", etc.)
3. **Normaliza** genero ("Menina"/"Menino" -> "Feminino"/"Masculino") e booleanos ("Sim"/"Nao" -> 1/0)
4. **Cria a variavel-alvo:** `target = 1` se defasagem > 0 (aluno em risco), `0` caso contrario
5. **Aplica feature engineering:** media_notas, nota_min, anos_na_pm, pedra_encoded, indicadores_baixos, etc.
6. **Trata dados faltantes:** mediana para numericos, "desconhecido" para categoricos
7. **Divide** em treino/teste (estratificado 80/20)
8. **Treina 4 modelos:** LightGBM, XGBoost, Random Forest, Logistic Regression
9. **Compara** os modelos por F1-Score (weighted) e seleciona o melhor
10. **Registra cada modelo no MLflow** com metricas, hiperparametros e artefatos
11. **Salva** o modelo vencedor em `app/model/model.joblib`

Apos o treino, acesse **http://localhost:5001** para visualizar os experimentos no MLflow UI.

O container e removido automaticamente (`--rm`), mas o `model.joblib` persiste na sua maquina em `app/model/` via volume mount.

#### 5. Subir a API

```bash
docker compose up app -d
```

Pronto! A API esta rodando em **http://localhost:8000**.

Acesse:
- **http://localhost:8000/api/v1/health** - verificar se modelo carregou
- **http://localhost:8000/docs** - Swagger UI interativo (testar endpoints pelo navegador)
- **http://localhost:8000/redoc** - Documentacao ReDoc

#### 6. Autenticar e testar uma predicao

**Passo 1 - Obter token JWT:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "passos-magicos"}'
```

Resposta:
```json
{"access_token": "<token>", "token_type": "bearer"}
```

**Passo 2 - Usar o token na predicao:**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "fase": 3, "idade": 14, "genero": "Feminino",
    "ano_ingresso": 2020, "inde": 6.5, "pedra": "Agata",
    "iaa": 7.2, "ieg": 6.8, "ips": 7.0, "ida": 5.9,
    "ipp": 6.5, "ipv": 6.0, "ian": 5.5,
    "nota_mat": 6.0, "nota_por": 7.0, "nota_ing": 5.5,
    "atingiu_pv": false, "indicado_bolsa": false
  }'
```

### Servicos Opcionais (Desenvolvimento)

```bash
# Dashboard de monitoramento de drift (Streamlit)
docker compose --profile monitoring up dashboard -d
# -> http://localhost:8501

# Stack completa de observabilidade (SigNoz + OTel Collector)
docker compose --profile signoz up -d
# -> http://localhost:8080 (SigNoz UI)
```

**Todos os servicos (desenvolvimento):**
| Servico | URL | Comando |
|---------|-----|---------|
| API | http://localhost:8000 | `docker compose up app -d` |
| Swagger UI | http://localhost:8000/docs | (incluso na API) |
| Web UI | http://localhost:3000 | `docker compose up webapp -d` |
| MLflow | http://localhost:5001 | `docker compose up mlflow -d` |
| Drift Dashboard | http://localhost:8501 | `docker compose --profile monitoring up -d` |
| SigNoz | http://localhost:8080 | `docker compose up signoz-otel-collector -d` |

### Deploy de Producao (todos os servicos)

O `docker-compose.prod.yml` sobe todos os servicos atras de um NGINX reverse proxy na porta 80:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

**Rotas disponíveis (porta 80):**
| Rota | Servico |
|------|---------|
| `/` | Interface Web (auth, predicao, historico) |
| `/api/v1/*` | API FastAPI |
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/mlflow/` | MLflow (experimentos e model registry) |
| `/drift/` | Dashboard de drift (Streamlit) |
| `:8443` | SigNoz (traces, metricas, logs — porta dedicada) |

### Resumo - Todos os Comandos

```bash
# 1. Copiar dados
cp "<caminho>/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" data/raw/

# 2. Configurar variaveis de ambiente
cp .env.example .env
# editar .env e definir JWT_SECRET_KEY, API_USERNAME, API_PASSWORD

# 3. Build
docker compose build

# 4. Subir MLflow e treinar modelo
docker compose up mlflow -d
docker compose --profile train run --rm train

# 5. Subir API
docker compose up app -d

# 6. Verificar
curl http://localhost:8000/api/v1/health

# Ou: subir TUDO em producao (NGINX + API + MLflow + Drift + SigNoz + Web UI)
docker compose -f docker-compose.prod.yml up --build -d

# Parar tudo
docker compose down
```

### Desenvolvimento Local (alternativa sem Docker)

Para quem prefere rodar localmente sem Docker:

```bash
# Instalar dependencias
poetry install

# (macOS) Instalar libomp para LightGBM
brew install libomp

# Rodar testes
poetry run pytest

# Treinar modelo
poetry run python scripts/train_pipeline.py

# Iniciar API
poetry run uvicorn app.main:app --reload --port 8000
```

---

## Documentacao da API

### Endpoints

| Metodo | Endpoint | Auth | Descricao |
|--------|----------|------|-----------|
| `GET` | `/api/v1/health` | Nao | Verifica status da API e se o modelo esta carregado |
| `POST` | `/api/v1/auth/login` | Nao | Autentica e retorna um token JWT |
| `POST` | `/api/v1/predict` | **Sim** | Prediz risco de defasagem de um aluno |

### Autenticacao

A API utiliza autenticacao JWT via Bearer token. As credenciais sao configuradas por variaveis de ambiente (veja `.env.example`).

**Variaveis de ambiente:**

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `API_USERNAME` | `admin` | Usuario para login |
| `API_PASSWORD` | `passos-magicos` | Senha para login |
| `JWT_SECRET_KEY` | `changeme-secret-key-for-development` | Chave de assinatura JWT (trocar em producao) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Tempo de expiracao do token |

> **Atencao:** Em producao, defina `JWT_SECRET_KEY` como um segredo forte e unico.

**Fluxo de autenticacao:**

1. `POST /api/v1/auth/login` com `{"username": "...", "password": "..."}`
2. Receba o `access_token` na resposta
3. Inclua `Authorization: Bearer <token>` no header de todas as chamadas ao `/predict`

No **Swagger UI** (`http://localhost:8000/docs`), clique em **Authorize** e insira o token para testar os endpoints autenticados interativamente.

### Swagger UI (Documentacao Interativa)

Com a API rodando, acesse **http://localhost:8000/docs** no navegador para a documentacao Swagger completa e interativa. La voce pode:
- Visualizar todos os endpoints com schemas de request/response
- Autenticar via botao **Authorize** (insira o token obtido no login)
- Testar cada endpoint diretamente pelo navegador (botao "Try it out")
- Baixar o schema OpenAPI em JSON (`/openapi.json`)

Alternativamente, acesse **http://localhost:8000/redoc** para a documentacao em formato ReDoc.

### Postman Collection

Uma colecao Postman com todos os endpoints esta disponivel em [`postman/`](postman/) (inclui collection e environment).

**Para importar:**
1. Abra o Postman
2. Clique em **Import** (canto superior esquerdo)
3. Selecione o arquivo `postman/FIAP - Datathon.postman_collection.json`
4. Importe tambem o environment `postman/FIAP-Local.postman_environment.json`
5. A colecao "Passos Magicos API" aparecera com os seguintes requests:
   - **Login** - `POST /api/v1/auth/login`
   - **Health Check** - `GET /api/v1/health`
   - **Predict - Aluno Risco Alto** - aluno com indicadores baixos
   - **Predict - Aluno Risco Baixo** - aluno com indicadores altos
   - **Predict - Aluno Risco Medio** - aluno com indicadores medianos
   - **Predict - Input Invalido** - teste de validacao (422)
   - **Swagger UI** - abre a documentacao
   - **OpenAPI Schema** - schema JSON da API

A variavel `{{base_url}}` esta configurada como `http://localhost:8000` por padrao.

---

## Exemplos de Chamadas a API

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Resposta:
```json
{"status": "healthy", "model_loaded": true}
```

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "passos-magicos"}'
```

Resposta:
```json
{"access_token": "<jwt-token>", "token_type": "bearer"}
```

### Predicao Individual

```bash
# Obter token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "passos-magicos"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Fazer predicao
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "fase": 3,
    "idade": 14,
    "genero": "Feminino",
    "ano_ingresso": 2020,
    "inde": 6.5,
    "pedra": "Agata",
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
```

Resposta:
```json
{
  "prediction": 1,
  "probability": 0.73,
  "risk_level": "alto"
}
```

**Niveis de risco:**
| Nivel | Probabilidade | Acao Sugerida |
|-------|--------------|---------------|
| baixo | < 0.3 | Acompanhamento padrao |
| medio | 0.3 - 0.6 | Atencao redobrada |
| alto | >= 0.6 | Intervencao imediata |

### Sem autenticacao (retorna 401)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"fase": 3, ...}'
# HTTP 401 - {"detail": "Not authenticated"}
```

### Input invalido (retorna 422)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"fase": "invalido"}'
```

---

## Pipeline de Machine Learning

### 1. Pre-processamento (`src/preprocessing.py`)

- **Carga:** Le 3 abas do XLSX (PEDE2022, PEDE2023, PEDE2024) com ~3000 registros
- **Padronizacao:** Mapeia colunas entre anos (nomes diferem entre 2022/2023/2024)
- **Normalizacao:** Genero ("Menina"/"Menino" -> "Feminino"/"Masculino"), booleanos ("Sim"/"Nao" -> 1/0)
- **Variavel-alvo:** `target = 1 se defasagem > 0, else 0` (classificacao binaria)
- **Dados faltantes:** Mediana para numericos, "desconhecido" para categoricos
- **Split:** Estratificado 80/20 ou temporal (treino 2022-2023, teste 2024)

### 2. Feature Engineering (`src/feature_engineering.py`)

| Feature | Calculo |
|---------|---------|
| media_notas | Media de Mat, Por, Ing |
| nota_min | Menor nota entre as 3 disciplinas |
| anos_na_pm | ano - ano_ingresso |
| fase_num | Numero extraido da Fase (ALFA=0) |
| pedra_encoded | Quartzo=1, Agata=2, Ametista=3, Topazio=4 |
| genero_encoded | Feminino=0, Masculino=1 |
| indicadores_baixos | Contagem de indicadores abaixo de 5.0 |

**Features de entrada do modelo (22 total):**
INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN, notas (Mat/Por/Ing), idade, ponto_virada, indicado_bolsa, ano, media_notas, nota_min, anos_na_pm, fase_num, pedra_encoded, genero_encoded, indicadores_baixos

### 3. Treinamento (`src/train.py`)

- **Modelos candidatos:** LightGBM, XGBoost, Random Forest, Logistic Regression
- **Pipeline:** StandardScaler + SimpleImputer + Classificador
- **Validacao:** StratifiedKFold (5 folds)
- **Tuning:** GridSearchCV no melhor modelo
- **Metrica principal:** F1-Score (weighted)
- **Serializacao:** joblib
- **Experiment tracking:** MLflow (cada run registra metricas, hiperparametros e artefatos)

### 4. Avaliacao (`src/evaluate.py`)

Metricas reportadas:
- F1-Score (weighted e macro)
- Accuracy, Precision, Recall
- AUC-ROC
- Matriz de confusao
- Classification report

### 5. Experiment Tracking (`MLflow`)

O pipeline de treino integra com MLflow para rastreamento completo de experimentos:

- **Cada modelo** treinado e registrado como um run separado no MLflow
- **Parametros logados:** hiperparametros do classificador, estrategia de split, numero de features
- **Metricas logadas:** F1 (weighted/macro), accuracy, precision, recall, AUC-ROC, matriz de confusao
- **Artefatos:** pipeline sklearn completo versionado como artefato do run

Acesso: `http://localhost:5001` (dev) ou `http://localhost/mlflow/` (producao via NGINX)

### 6. Monitoramento e Observabilidade

- **Logs de predicao:** Cada request ao `/api/v1/predict` gera log JSON em `logs/predictions.log`
- **Dashboard de drift:** Streamlit mostra distribuicao de features, probabilidades e latencia
- **OpenTelemetry + SigNoz:** traces distribuidos, metricas customizadas e logs centralizados

**Spans customizados (OTel):**

| Span | O que mede |
|------|-----------|
| `model.load` | Tempo de carga do modelo no startup |
| `auth.login` | Tentativas de login (sucesso/falha) |
| `predict` | Span pai da predicao com atributos de negocio (risco, probabilidade, dados do aluno) |
| `predict.feature_engineering` | Duracao do feature engineering |
| `predict.model_inference` | Duracao da inferencia do modelo |

**Metricas customizadas (OTel):**

| Metrica | Tipo | Valor |
|---------|------|-------|
| `predictions.total` | Counter | Volume de uso da API |
| `predictions.risk_level` | Counter por nivel | Distribuicao de risco (baixo/medio/alto) |
| `predictions.probability` | Histogram | Distribuicao de probabilidades preditas |
| `predictions.latency_ms` | Histogram | Latencia end-to-end (P50/P95/P99) |
| `auth.login.total` | Counter | Tentativas de login |
| `auth.login.failures` | Counter | Falhas de autenticacao |
| `model.load_time_ms` | Histogram | Tempo de carga do modelo |

**Dashboards SigNoz predefinidos:**

O projeto inclui dois dashboards prontos para importar no SigNoz:

- **ML Predictions** — throughput de predicoes, latencia (p50/p95/p99), distribuicao de risco, probabilidade, latencia de feature engineering e inferencia
- **API Health & Security** — tentativas de login (sucesso/falha), tempo de carga do modelo, request rate por rota, erros HTTP, latencia por endpoint

Para provisionar automaticamente via API:

```bash
./scripts/provision_dashboards.sh http://localhost:8080
```

Ou importe manualmente: SigNoz UI -> Dashboards -> New Dashboard -> Import JSON (arquivos em `signoz/dashboards/`).

### 7. Feature Store (Feast) - Opcional

O projeto integra Feast Feature Store para centralizar e reutilizar features entre treino e servico, evitando training-serving skew.

**Componentes:**
- `feature_store/feature_store.yaml` — configuracao (provider local, offline parquet, online SQLite)
- `feature_store/definitions.py` — entidade `student` e FeatureView com 22 features
- `scripts/materialize_features.py` — carrega XLSX, aplica feature engineering, salva parquet e materializa no online store

**Como usar:**

```bash
# 1. Materializar features (gera parquet a partir do XLSX)
docker compose --profile feast run --rm materialize

# 2. Treinar usando Feature Store
USE_FEATURE_STORE=true docker compose --profile train run --rm train

# 3. Subir API com Feature Store
USE_FEATURE_STORE=true docker compose up app -d
```

> **Nota:** O Feature Store e opcional. Por padrao (`USE_FEATURE_STORE=false`), o pipeline carrega dados diretamente do XLSX e aplica feature engineering inline.

### 8. Variaveis de Ambiente

Todas as variaveis configuráveis do projeto:

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `JWT_SECRET_KEY` | `changeme-...` | Chave de assinatura JWT (**obrigatorio em producao**) |
| `API_USERNAME` | `admin` | Usuario para login |
| `API_PASSWORD` | `passos-magicos` | Senha para login |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Tempo de expiracao do token |
| `ENVIRONMENT` | `development` | Ambiente (`development` ou `production`) |
| `LOG_LEVEL` | `INFO` | Nivel de log |
| `USE_FEATURE_STORE` | `false` | Usar Feast Feature Store em vez de feature engineering inline |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI do servidor MLflow |
| `MODEL_PATH` | `app/model/model.joblib` | Caminho do modelo treinado |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://signoz-otel-collector:4318` | Endpoint do OTel Collector |
| `SIGNOZ_VERSION` | `v0.101.0` | Versao do SigNoz |
| `SIGNOZ_OTELCOL_TAG` | `v0.129.8` | Versao do OTel Collector SigNoz |

---

## Avaliacao de Resultados

### Comparacao entre Modelos

Foram treinados e avaliados 4 modelos de classificacao para identificar alunos em risco de defasagem escolar. A tabela abaixo apresenta o desempenho de cada modelo no conjunto de teste:

| Modelo | F1 (weighted) | F1 (macro) | Accuracy | Precision (weighted) | Recall (weighted) | AUC-ROC |
|--------|--------------|------------|----------|---------------------|-------------------|---------|
| **LightGBM** | **0.9702** | **0.8657** | **0.9719** | **0.9702** | **0.9719** | **0.9590** |
| XGBoost | 0.9642 | 0.8429 | 0.9653 | 0.9635 | 0.9653 | 0.9567 |
| Random Forest | 0.9569 | 0.7951 | 0.9620 | 0.9590 | 0.9620 | 0.9383 |
| Logistic Regression | 0.9278 | 0.6476 | 0.9389 | 0.9243 | 0.9389 | 0.9300 |

O **LightGBM** foi selecionado como modelo final por apresentar o melhor desempenho em todas as metricas avaliadas.

### Analise Detalhada do Modelo Selecionado (LightGBM)

O classification report do LightGBM no conjunto de teste revela um aspecto importante do problema:

| Classe | Precision | Recall | F1-Score | Amostras |
|--------|-----------|--------|----------|----------|
| Sem Defasagem | 0.98 | 0.99 | 0.99 | 568 (93.7%) |
| Com Defasagem | 0.86 | 0.66 | 0.75 | 38 (6.3%) |

### Desbalanceamento de Classes

O dataset apresenta um desbalanceamento significativo: apenas **6.3% dos alunos** no conjunto de teste possuem defasagem escolar. Isso tem implicacoes diretas na interpretacao dos resultados:

- **Metricas weighted e accuracy (0.97)** sao infladas pela classe majoritaria ("Sem Defasagem"), que representa 93.7% dos dados. O modelo acerta quase todos esses casos, o que eleva as metricas gerais.
- **F1 macro (0.87)** oferece uma visao mais equilibrada, pois calcula a media simples entre as duas classes sem ponderar pelo numero de amostras.
- **AUC-ROC (0.96)** indica que o modelo possui boa capacidade de discriminacao entre as classes em diferentes thresholds de decisao.

### Performance na Classe de Interesse

Para o contexto educacional deste projeto, a classe mais importante e a de **alunos com defasagem** — sao esses os casos que exigem intervencao pedagogica. Nessa classe:

- **Precision (0.86):** quando o modelo indica que um aluno esta em risco, ele acerta em 86% dos casos. A taxa de falsos alarmes e baixa.
- **Recall (0.66):** o modelo identifica corretamente 66% dos alunos com defasagem. Isso significa que **aproximadamente 1 em cada 3 alunos com defasagem nao e detectado** pelo modelo.
- **F1-Score (0.75):** a media harmonica entre precision e recall reflete esse trade-off.

### Consideracoes e Melhorias Futuras

O modelo atual apresenta um bom desempenho geral e ja oferece valor pratico como ferramenta de triagem para a equipe pedagogica. No entanto, o recall de 66% na classe minoritaria indica espaco para evolucao. Estrategias que podem ser exploradas em iteracoes futuras:

- **Tecnicas de balanceamento:** aplicar SMOTE (oversampling sintetico) ou undersampling da classe majoritaria durante o treinamento
- **Ajuste de threshold:** reduzir o limiar de classificacao (atualmente 0.5) para aumentar o recall, aceitando uma reducao controlada na precision
- **Class weights:** utilizar pesos diferenciados para as classes no treinamento, penalizando mais os erros na classe minoritaria
- **Otimizacao por metrica alternativa:** treinar otimizando F1 macro ou recall da classe minoritaria em vez de F1 weighted

---

## Infraestrutura como Codigo (IaC)

O projeto inclui configuracao Terraform para deploy automatico em um Droplet DigitalOcean.

### Pre-requisitos

- [Terraform](https://www.terraform.io/downloads) >= 1.5
- Conta DigitalOcean com [API token](https://cloud.digitalocean.com/account/api/tokens)
- Chave SSH registrada no DigitalOcean

### Deploy Manual via Terraform

```bash
cd infra/terraform

# Copiar e preencher variaveis
cp terraform.tfvars.example terraform.tfvars
# Editar terraform.tfvars com seus valores

# Inicializar, planejar e aplicar
terraform init
terraform plan
terraform apply
```

Apos o apply, o Terraform exibe:
- `droplet_ip` — IP publico do servidor
- `app_url` — URL de acesso (dominio ou IP)
- `ssh_command` — comando SSH para conectar

### Deploy Automatizado via GitHub Actions

O workflow `CD: Deploy to DigitalOcean` (`.github/workflows/cd-deploy.yml`) executa o Terraform automaticamente.

**Secrets necessarios no repositorio GitHub:**

| Secret | Descricao |
|--------|-----------|
| `DO_TOKEN` | API token DigitalOcean |
| `DO_SSH_KEY_FINGERPRINT` | Fingerprint da chave SSH |
| `JWT_SECRET_KEY` | Chave de assinatura JWT |
| `API_USERNAME` | Usuario da API |
| `API_PASSWORD` | Senha da API |

**Para executar:** Actions -> CD: Deploy to DigitalOcean -> Run workflow

O workflow tambem suporta destruir a infraestrutura selecionando `destroy: true`.

### O que o deploy cria

1. **Droplet** Ubuntu 20.04 com Docker (4 vCPU, 8 GB RAM)
2. **Firewall** — portas 22 (SSH), 80 (HTTP), 443 (HTTPS)
3. **DNS** (opcional) — registro A apontando para o IP do droplet
4. **Cloud-init** automatiza: clone do repo, criacao do `.env`, build e start do docker-compose.prod.yml
5. **Systemd service** para auto-start apos reboot
