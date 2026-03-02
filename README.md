# Passos Magicos - Predicao de Defasagem Escolar

Modelo preditivo de risco de defasagem escolar para a Associacao Passos Magicos, desenvolvido para o Datathon Pos Tech FIAP.

## Visao Geral

A Associacao Passos Magicos atua ha 32 anos transformando a vida de criancas e jovens de baixa renda em Embu-Guacu/SP. Este projeto implementa um sistema de Machine Learning que identifica precocemente estudantes em risco de defasagem escolar, permitindo intervencao direcionada pela equipe pedagogica.

**Problema:** Identificar quais alunos estao em risco de defasagem escolar antes que ela ocorra.

**Solucao:** Modelo de classificacao binaria que recebe indicadores educacionais de um aluno e retorna a probabilidade de defasagem, exposto via API REST.

**Stack:**
- Python 3.13
- scikit-learn + LightGBM
- FastAPI + Uvicorn
- Docker + Docker Compose
- OpenTelemetry + SigNoz (observabilidade)
- Streamlit + Evidently (monitoramento de drift)

---

## Estrutura do Projeto

```
passos-magicos-ml/
|-- app/                          # API FastAPI
|   |-- main.py                   # Entry point, carrega modelo no startup
|   |-- routes.py                 # POST /predict, GET /health
|   |-- schemas.py                # Pydantic: StudentInput, PredictionOutput
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
|-- tests/                        # Testes unitarios (86% cobertura)
|   |-- conftest.py               # Fixtures compartilhadas
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   |-- test_train.py
|   |-- test_evaluate.py
|   |-- test_api.py
|
|-- monitoring/                   # Monitoramento
|   |-- drift_dashboard.py        # Dashboard Streamlit de drift
|   |-- logger.py                 # Logging estruturado JSON
|
|-- signoz/                       # Configuracoes SigNoz/OTel
|-- data/                         # Dados brutos e processados
|-- docs/                         # PRD e Tech Spec
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
```

---

## Instrucoes de Deploy

### Pre-requisitos
- Docker e Docker Compose instalados
- Python 3.13+ (para desenvolvimento local)

### Deploy com Docker (recomendado)

```bash
# 1. Build da imagem
docker compose build

# 2. Subir a API
docker compose up api -d

# 3. Verificar se esta rodando
curl http://localhost:8000/health

# 4. (Opcional) Subir o dashboard de drift
docker compose --profile monitoring up dashboard -d

# 5. (Opcional) Subir stack de observabilidade SigNoz
docker compose --profile signoz up -d
```

**Servicos disponiveis:**
| Servico | URL | Descricao |
|---------|-----|-----------|
| API | http://localhost:8000 | Endpoints /predict e /health |
| Swagger UI | http://localhost:8000/docs | Documentacao interativa da API |
| Drift Dashboard | http://localhost:8501 | Monitoramento de drift (Streamlit) |
| SigNoz | http://localhost:3301 | Observabilidade (traces, metricas, logs) |

### Desenvolvimento local

```bash
# Instalar dependencias
poetry install

# Rodar testes
poetry run pytest

# Treinar modelo
poetry run python -m src.train

# Iniciar API localmente
poetry run uvicorn app.main:app --reload --port 8000
```

---

## Documentacao da API

### Swagger UI (Documentacao Interativa)

Com a API rodando, acesse **http://localhost:8000/docs** no navegador para a documentacao Swagger completa e interativa. La voce pode:
- Visualizar todos os endpoints com schemas de request/response
- Testar cada endpoint diretamente pelo navegador (botao "Try it out")
- Baixar o schema OpenAPI em JSON (`/openapi.json`)

Alternativamente, acesse **http://localhost:8000/redoc** para a documentacao em formato ReDoc.

### Postman Collection

Uma colecao Postman com todos os endpoints esta disponivel em [`docs/postman_collection.json`](docs/postman_collection.json).

**Para importar:**
1. Abra o Postman
2. Clique em **Import** (canto superior esquerdo)
3. Selecione o arquivo `docs/postman_collection.json`
4. A colecao "Passos Magicos API" aparecera com os seguintes requests:
   - **Health Check** - `GET /health`
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
curl http://localhost:8000/health
```

Resposta:
```json
{"status": "healthy", "model_loaded": true}
```

### Predicao Individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
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

### Input invalido (retorna 422)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
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

- **Modelos candidatos:** LightGBM, Random Forest, Logistic Regression
- **Pipeline:** StandardScaler + SimpleImputer + Classificador
- **Validacao:** StratifiedKFold (5 folds)
- **Tuning:** GridSearchCV no melhor modelo
- **Metrica principal:** F1-Score (weighted)
- **Serializacao:** joblib

### 4. Avaliacao (`src/evaluate.py`)

Metricas reportadas:
- F1-Score (weighted e macro)
- Accuracy, Precision, Recall
- AUC-ROC
- Matriz de confusao
- Classification report

### 5. Monitoramento

- **Logs de predicao:** Cada request ao `/predict` gera log JSON em `logs/predictions.log`
- **Dashboard de drift:** Streamlit em `http://localhost:8501` mostra distribuicao de features, probabilidades e latencia
- **Observabilidade:** OpenTelemetry + SigNoz para traces distribuidos, metricas de sistema e logs centralizados
