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
|   |-- routes.py                 # POST /api/v1/predict, GET /api/v1/health, POST /api/v1/auth/login
|   |-- schemas.py                # Pydantic: StudentInput, PredictionOutput, LoginRequest, TokenResponse
|   |-- auth.py                   # JWT: criacao e verificacao de tokens
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

Isso cria a imagem com Python 3.13, LightGBM, scikit-learn, FastAPI e todas as dependencias. Leva ~2-3 minutos na primeira vez.

#### 4. Treinar o modelo (dentro do Docker)

```bash
docker compose --profile train run --rm train
```

Este comando sobe um container temporario que:
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

### Servicos Opcionais

```bash
# Dashboard de monitoramento de drift (Streamlit)
docker compose --profile monitoring up dashboard -d
# -> http://localhost:8501

# Stack completa de observabilidade (SigNoz + OTel Collector)
docker compose --profile signoz up -d
# -> http://localhost:8080 (SigNoz UI)
```

**Todos os servicos:**
| Servico | URL | Comando |
|---------|-----|---------|
| API | http://localhost:8000 | `docker compose up app -d` |
| Swagger UI | http://localhost:8000/docs | (incluso na API) |
| Drift Dashboard | http://localhost:8501 | `docker compose --profile monitoring up -d` |
| SigNoz | http://localhost:8080 | `docker compose --profile signoz up -d` |

### Resumo - Todos os Comandos

```bash
# 1. Copiar dados
cp "<caminho>/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" data/raw/

# 2. Configurar variaveis de ambiente
cp .env.example .env
# editar .env e definir JWT_SECRET_KEY, API_USERNAME, API_PASSWORD

# 3. Build
docker compose build

# 4. Treinar modelo (dentro do Docker)
docker compose --profile train run --rm train

# 5. Subir API
docker compose up app -d

# 6. Verificar
curl http://localhost:8000/api/v1/health

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

Uma colecao Postman com todos os endpoints esta disponivel em [`docs/postman_collection.json`](docs/postman_collection.json).

**Para importar:**
1. Abra o Postman
2. Clique em **Import** (canto superior esquerdo)
3. Selecione o arquivo `docs/postman_collection.json`
4. A colecao "Passos Magicos API" aparecera com os seguintes requests:
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

- **Logs de predicao:** Cada request ao `/api/v1/predict` gera log JSON em `logs/predictions.log`
- **Dashboard de drift:** Streamlit em `http://localhost:8501` mostra distribuicao de features, probabilidades e latencia
- **Observabilidade:** OpenTelemetry + SigNoz para traces distribuidos, metricas de sistema e logs centralizados
