# PRD - Product Requirements Document

## Passos Magicos - Sistema de Predicao de Risco de Defasagem Escolar

---

## 1. Visao Geral

**Produto:** Sistema de predicao de risco de defasagem escolar para a Associacao Passos Magicos.

**Problema:** A Passos Magicos atende ~1000+ criancas/jovens por ano em Embu-Guacu/SP. Identificar precocemente quais estudantes estao em risco de defasagem escolar permite intervencao direcionada (reforco, acompanhamento psicologico, etc.), otimizando recursos limitados.

**Solucao:** Modelo de Machine Learning que recebe indicadores educacionais de um aluno e retorna a probabilidade de defasagem, exposto via API REST, com ciclo completo de MLOps.

---

## 2. Usuarios

| Usuario | Necessidade |
|---------|-------------|
| Equipe pedagogica Passos Magicos | Identificar alunos em risco para intervencao precoce |
| Gestores da associacao | Dashboard de visao geral dos riscos por turma/fase |
| Avaliadores (datathon) | Validar pipeline MLOps completa |

---

## 3. Requisitos Funcionais

| ID | Requisito | Prioridade |
|----|-----------|-----------|
| RF01 | Modelo preditivo binario: prever se aluno tem risco de defasagem (sim/nao) | Alta |
| RF02 | Pipeline completa: preprocessing -> feature engineering -> training -> evaluation | Alta |
| RF03 | API REST com endpoint `POST /predict` que recebe dados e retorna predicao | Alta |
| RF04 | API REST com endpoint `GET /health` para verificar disponibilidade | Alta |
| RF05 | Modelo serializado com joblib para carregamento rapido | Alta |
| RF06 | Resposta da API inclui: predicao binaria, probabilidade, nivel de risco (baixo/medio/alto) | Alta |
| RF07 | Validacao de input na API (campos obrigatorios, tipos, ranges) | Media |
| RF08 | Logs de cada predicao (input, output, timestamp, latencia) | Media |
| RF09 | Dashboard de monitoramento de drift | Media |
| RF10 | Testes unitarios com >= 80% de cobertura | Alta |
| RF11 | Observabilidade com OpenTelemetry: traces distribuidos, metricas e logs | Media |
| RF12 | Dashboard SigNoz para visualizacao de traces, metricas e logs | Media |
| RF13 | Instrumentacao automatica da API (FastAPI, HTTP clients) | Media |
| RF14 | Spans customizados no pipeline de predicao (load model, validate, predict) | Baixa |

---

## 4. Requisitos Nao-Funcionais

| ID | Requisito | Criterio |
|----|-----------|---------|
| RNF01 | Tempo de resposta da API | < 500ms por predicao |
| RNF02 | Disponibilidade | Container Docker levanta em < 30s |
| RNF03 | Portabilidade | Funciona via Docker em qualquer SO |
| RNF04 | Codigo organizado | Modulos separados por responsabilidade |
| RNF05 | Documentacao | README com 5 secoes obrigatorias |
| RNF06 | Cobertura de testes | >= 80% |
| RNF07 | Observabilidade | Traces, metricas e logs via OTel + SigNoz |
| RNF08 | Dashboard de observabilidade | SigNoz UI acessivel em localhost:8080 |

---

## 5. Dados de Entrada

**Fonte:** Dataset PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) 2022-2024

**Arquivos disponiveis:**

| Arquivo | Descricao | Linhas | Colunas |
|---------|-----------|--------|---------|
| `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (aba PEDE2022) | Dados PEDE 2022 | 861 | 42 |
| `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (aba PEDE2023) | Dados PEDE 2023 | 1014 | 48 |
| `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (aba PEDE2024) | Dados PEDE 2024 | 1156 | 50 |
| `PEDE_PASSOS_DATASET_FIAP.csv` (base antiga) | Dados 2020-2022 wide format | 1349 | 69 |

**Volume total:** ~3000 registros (861 + 1014 + 1156)

**Variavel-alvo:** Defasagem binaria (0 = sem risco, 1 = com risco)
- Definicao: `Defasagem = Fase_Ideal - Fase_Atual`
- Valor 0 = sem defasagem, valores positivos = aluno atrasado
- Binarizacao: `target = 1 se Defasagem > 0, else 0`

**Features do modelo (input da API):**

| Categoria | Features |
|-----------|----------|
| Indicadores educacionais | INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN |
| Classificacao | PEDRA (ordinal: Quartzo < Agata < Ametista < Topazio) |
| Notas por disciplina | Matematica, Portugues, Ingles |
| Contexto do aluno | Fase, Idade, Genero, Ano de ingresso |
| Booleanos | Atingiu Ponto de Virada, Indicado para bolsa |

**Indicadores-chave (dicionario resumido):**

| Indicador | Significado | Tipo |
|-----------|------------|------|
| INDE | Indice de Desenvolvimento Educacional (composto de todos indicadores) | Numerico 0-10 |
| PEDRA | Classificacao por faixa de INDE | Categorico ordinal |
| IAA | Indicador de Auto Avaliacao | Numerico 0-10 |
| IEG | Indicador de Engajamento | Numerico 0-10 |
| IPS | Indicador Psicossocial | Numerico 0-10 |
| IDA | Indicador de Aprendizagem | Numerico 0-10 |
| IPP | Indicador Psicopedagogico | Numerico 0-10 |
| IPV | Indicador de Ponto de Virada | Numerico 0-10 |
| IAN | Indicador de Adequacao ao Nivel | Numerico 0-10 |

---

## 6. Dados de Saida (resposta da API)

```json
{
  "prediction": 1,
  "probability": 0.73,
  "risk_level": "alto"
}
```

**Niveis de risco:**
- **baixo**: probabilidade < 0.3
- **medio**: 0.3 <= probabilidade < 0.6
- **alto**: probabilidade >= 0.6

---

## 7. Entregaveis

| # | Entregavel | Formato |
|---|-----------|---------|
| 1 | Codigo-fonte organizado e documentado | Repositorio GitHub |
| 2 | Documentacao do projeto | README.md com 5 secoes obrigatorias |
| 3 | API funcionando | Docker local (`localhost:8000`) |
| 4 | Video de apresentacao | Ate 5 minutos, formato gerencial |

---

## 8. Criterios de Aceitacao

- [ ] Modelo treinado com metrica documentada (F1-Score weighted)
- [ ] API retorna predicoes corretas para inputs validos (status 200)
- [ ] API retorna erro 422 para inputs invalidos
- [ ] Docker build e run funcionam sem erros
- [ ] Testes passam com >= 80% cobertura
- [ ] Logs registram cada predicao (timestamp, input, output, latencia)
- [ ] Dashboard de drift disponivel (Streamlit)
- [ ] README.md completo com as 5 secoes obrigatorias
- [ ] Repositorio GitHub organizado com estrutura modular
- [ ] Video de ate 5 minutos gravado

---

## 9. Observabilidade

O sistema inclui stack completa de observabilidade baseada em OpenTelemetry + SigNoz:

| Pilar | Ferramenta | O que captura |
|-------|-----------|---------------|
| **Traces** | OTel + SigNoz | Fluxo completo de cada request, spans por etapa da predicao |
| **Metricas** | OTel + Prometheus | Request count, latencia, predicoes por tipo, model loaded |
| **Logs** | OTel + SigNoz | Logs estruturados JSON com correlacao de trace |

**Acesso:**
- SigNoz UI: `http://localhost:8080`
- OTel Collector gRPC: `localhost:4317`
- OTel Collector HTTP: `localhost:4318`

---

## 10. Fora de Escopo

- Deploy em nuvem (sera local com Docker)
- Interface grafica para usuario final
- Retreinamento automatico do modelo
- Integracao com sistemas da Passos Magicos
- Processamento de linguagem natural nas recomendacoes textuais (MVP)
