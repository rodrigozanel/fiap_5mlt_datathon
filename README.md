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
- Python 3.13+ com [Poetry](https://python-poetry.org/)
- Docker e Docker Compose (para deploy containerizado)
- (macOS) `brew install libomp` (necessario para LightGBM)

### Passo a Passo Completo (do zero ate a API rodando)

#### 1. Instalar dependencias

```bash
# Instalar todas as dependencias do projeto (incluindo dev)
poetry install

# (macOS apenas) Instalar libomp para o LightGBM funcionar
brew install libomp
```

Isso cria um virtualenv automaticamente e instala: scikit-learn, lightgbm, pandas, fastapi, uvicorn, pytest, etc.

#### 2. Rodar os testes (validar que tudo funciona)

```bash
poetry run pytest
```

Resultado esperado: **71 tests passed, 86% coverage**. Se o LightGBM nao estiver instalado, 1 teste sera pulado (ok).

#### 3. Treinar o modelo com os dados reais

Este e o passo que gera o arquivo `app/model/model.joblib` - sem ele a API nao consegue fazer predicoes.

```bash
# Coloque o XLSX na pasta data/raw/
cp "drive-download-20260301T222743Z-1-001/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" data/raw/

# Rodar o script de treinamento
poetry run python scripts/train_pipeline.py
```

**O que o script faz internamente:**
1. **Carrega** as 3 abas do XLSX (PEDE2022, PEDE2023, PEDE2024) - ~3000 registros
2. **Padroniza** os nomes de colunas (diferem entre anos: "Defas" vs "Defasagem", "Matem" vs "Mat", etc.)
3. **Normaliza** genero ("Menina"/"Menino" -> "Feminino"/"Masculino") e booleanos ("Sim"/"Nao" -> 1/0)
4. **Cria a variavel-alvo:** `target = 1` se defasagem > 0 (aluno em risco), `0` caso contrario
5. **Aplica feature engineering:** media_notas, nota_min, anos_na_pm, pedra_encoded, indicadores_baixos, etc.
6. **Trata dados faltantes:** mediana para numericos, "desconhecido" para categoricos
7. **Divide** em treino/teste (estratificado 80/20)
8. **Treina 3 modelos:** LightGBM, Random Forest, Logistic Regression
9. **Compara** os modelos por F1-Score (weighted) e seleciona o melhor
10. **Salva** o modelo vencedor em `app/model/model.joblib`
11. **Imprime** metricas (F1, Accuracy, Precision, Recall, AUC-ROC) e classification report

Saida esperada:
```
=== Passos Magicos - Training Pipeline ===
Loading data from data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx...
Combined dataset: XXXX rows, XX features
Training lgbm... done
Training rf... done
Training lr... done

=== Model Comparison ===
          f1_weighted  accuracy  auc_roc
lgbm         0.XXXX    0.XXXX   0.XXXX
rf           0.XXXX    0.XXXX   0.XXXX
lr           0.XXXX    0.XXXX   0.XXXX

Best model: lgbm (F1=0.XXXX)
Model saved to app/model/model.joblib
```

#### 4. Iniciar a API localmente

```bash
poetry run uvicorn app.main:app --reload --port 8000
```

Acesse:
- **http://localhost:8000/health** - verificar se modelo carregou
- **http://localhost:8000/docs** - Swagger UI interativo

#### 5. Testar uma predicao

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fase": 3, "idade": 14, "genero": "Feminino",
    "ano_ingresso": 2020, "inde": 6.5, "pedra": "Agata",
    "iaa": 7.2, "ieg": 6.8, "ips": 7.0, "ida": 5.9,
    "ipp": 6.5, "ipv": 6.0, "ian": 5.5,
    "nota_mat": 6.0, "nota_por": 7.0, "nota_ing": 5.5,
    "atingiu_pv": false, "indicado_bolsa": false
  }'
```

### Deploy com Docker

Depois de treinar o modelo (passo 3), o arquivo `model.joblib` ja esta em `app/model/` e sera copiado para dentro da imagem Docker.

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
| ReDoc | http://localhost:8000/redoc | Documentacao alternativa da API |
| Drift Dashboard | http://localhost:8501 | Monitoramento de drift (Streamlit) |
| SigNoz | http://localhost:3301 | Observabilidade (traces, metricas, logs) |

### Resumo dos Comandos

```bash
# Setup completo (uma vez)
poetry install
brew install libomp              # macOS apenas
cp <XLSX> data/raw/

# Pipeline de treino + deploy
poetry run pytest                # validar testes
poetry run python scripts/train_pipeline.py  # treinar modelo
docker compose build             # build da imagem
docker compose up api -d         # subir API
curl http://localhost:8000/health  # verificar
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
