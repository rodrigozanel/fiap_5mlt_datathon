# Contribuindo com Passos Magicos - Predicao de Defasagem Escolar

Obrigado por contribuir com este projeto. Este documento cobre configuracao, fluxo de trabalho e padroes de desenvolvimento.

## Sumario

- [Primeiros Passos](#primeiros-passos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Fluxo de Desenvolvimento](#fluxo-de-desenvolvimento)
- [Estilo de Codigo](#estilo-de-codigo)
- [Testes](#testes)
- [Desenvolvimento com Docker](#desenvolvimento-com-docker)
- [Enviando Alteracoes](#enviando-alteracoes)
- [Reportando Problemas](#reportando-problemas)

## Primeiros Passos

### Pre-requisitos

- Python 3.13+
- Docker Desktop
- Git

### Configuracao Local

```bash
git clone https://github.com/YOUR_USERNAME/fiap_5mlt_datathon.git
cd fiap_5mlt_datathon

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Treinar o Modelo

A API requer um modelo treinado em `app/model/model.joblib`. Treine antes de rodar a API:

```bash
# Via Docker (recomendado)
docker compose --profile train up train

# Ou localmente
PYTHONPATH=. python scripts/train_pipeline.py
```

### Rodar a API Localmente

```bash
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

## Estrutura do Projeto

```
fiap_5mlt_datathon/
├── app/                        # Aplicacao FastAPI
│   ├── main.py                 # Entry point, carrega o modelo no startup
│   ├── routes.py               # POST /predict, GET /health
│   ├── schemas.py              # Modelos Pydantic: StudentInput, PredictionOutput
│   └── model/
│       └── model.joblib        # Modelo treinado (nao versionado)
├── src/                        # Pipeline de ML
│   ├── preprocessing.py        # Carga, limpeza e split dos dados
│   ├── feature_engineering.py  # Criacao e encoding de features
│   ├── train.py                # Treinamento, tuning e salvamento do modelo
│   ├── evaluate.py             # Metricas e avaliacao do modelo
│   └── utils.py                # Logger, constantes e paths
├── monitoring/                 # Monitoramento
│   ├── drift_dashboard.py      # Dashboard Streamlit de drift (porta 8501)
│   └── logger.py               # Logging estruturado JSON das predicoes
├── scripts/
│   └── train_pipeline.py       # Script de treinamento end-to-end
├── tests/                      # Suite de testes
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_api.py
├── signoz/                     # Configuracoes OpenTelemetry / SigNoz
├── data/                       # Dados brutos e processados
├── docs/                       # PRD e Tech Spec
├── Dockerfile
└── docker-compose.yml
```

## Fluxo de Desenvolvimento

### 1. Crie uma Branch

```bash
git checkout -b feat/nome-da-funcionalidade
# ou
git checkout -b fix/descricao-do-problema
```

### 2. Faca as Alteracoes

- Escreva ou atualize testes para qualquer logica alterada
- Mantenha funcoes pequenas e com responsabilidade unica
- Nao commite `app/model/model.joblib` nem arquivos de dados

### 3. Rode os Testes

```bash
pytest
```

### 4. Commit

Siga o formato [Conventional Commits](https://www.conventionalcommits.org/):

```
tipo(escopo): descricao
```

**Tipos:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Exemplos:**
```
feat(api): adicionar intervalo de confianca na resposta de predicao
fix(routes): chamar log_prediction para registrar logs do dashboard
test(preprocessing): adicionar casos para valores ausentes em pedra
docs(readme): atualizar instrucoes de configuracao do Docker
```

## Estilo de Codigo

O projeto segue PEP 8. Mantenha consistencia com o codigo existente:

- Use type hints nos parametros e retornos de funcoes
- Organize os imports: stdlib, terceiros, locais
- Nao adicione docstrings ou comentarios em codigo que voce nao alterou

## Testes

Os testes ficam em `tests/` e usam `pytest`. A cobertura e configurada em `pyproject.toml` para reportar `src/` e `app/`.

```bash
# Rodar todos os testes com cobertura
pytest

# Rodar um arquivo especifico
pytest tests/test_api.py

# Rodar sem cobertura (mais rapido)
pytest --no-cov
```

Mantenha a cobertura acima de 80%. Teste cenarios de sucesso e de falha. Use as fixtures de `conftest.py` em vez de duplicar logica de setup.

## Desenvolvimento com Docker

### Servicos

| Servico     | Profile      | Porta | Descricao                          |
|-------------|--------------|-------|------------------------------------|
| `train`     | `train`      | —     | Treinamento do modelo (one-shot)   |
| `app`       | _(padrao)_   | 8000  | API FastAPI de predicao            |
| `dashboard` | `monitoring` | 8501  | Dashboard Streamlit de drift       |
| `signoz`    | _(padrao)_   | 8080  | Interface de observabilidade       |

### Comandos Comuns

```bash
# Treinar o modelo
docker compose --profile train up train

# Iniciar a API
docker compose up -d app

# Iniciar o dashboard de drift
docker compose --profile monitoring up -d dashboard

# Reconstruir apos alteracoes no codigo (sem volume mount no source)
docker compose build app && docker compose up -d app

# Ver logs
docker compose logs -f app

# Rodar testes dentro do container
docker compose run --rm app pytest
```

> Atencao: os servicos `app` e `dashboard` compartilham a mesma imagem. Alteracoes no codigo exigem rebuild — apenas reiniciar o container nao e suficiente.

## Enviando Alteracoes

1. Certifique-se de que sua branch esta atualizada com `main`
2. Rode a suite completa de testes (`pytest`)
3. Abra um Pull Request com titulo e descricao claros
4. Referencie issues relacionadas

### Template de Descricao do PR

```markdown
## O que mudou
Descricao resumida das alteracoes.

## Tipo de mudanca
- [ ] Correcao de bug
- [ ] Nova funcionalidade
- [ ] Refatoracao
- [ ] Documentacao

## Testes
- [ ] Testes adicionados / atualizados
- [ ] Todos os testes passando
- [ ] Teste manual realizado
```

## Reportando Problemas

Ao reportar um bug, inclua:

- Passos para reproduzir
- Comportamento esperado vs. comportamento observado
- Versao do Python, sistema operacional
- Logs ou mensagens de erro relevantes
