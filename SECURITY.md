# Politica de Seguranca

## Versoes Suportadas

| Versao | Suportada |
| ------ | --------- |
| 0.1.x  | Sim       |

## Reportando uma Vulnerabilidade

Reporte vulnerabilidades de seguranca abrindo uma issue privada no GitHub:

1. Acesse a pagina de [Issues](https://github.com/YOUR_USERNAME/fiap_5mlt_datathon/issues)
2. Clique em "New Issue" e selecione "Security Vulnerability" se o template estiver disponivel
3. Forneca uma descricao detalhada (veja abaixo)

**Tempo de resposta:** Buscamos confirmar o recebimento em ate 48 horas e apresentar um plano de resolucao em ate 7 dias.

### O que Incluir

- Descricao da vulnerabilidade e seu impacto potencial
- Passos para reproduzir
- Prova de conceito (se disponivel)
- Detalhes do ambiente (SO, versao do Python, versao do Docker)

## Consideracoes de Seguranca

### Privacidade dos Dados

Este projeto processa dados educacionais de estudantes da Associacao Passos Magicos — uma organizacao social que atende criancas e jovens. Esses dados sao sensiveis:

- Os campos de entrada incluem idade, serie escolar, notas academicas e indicadores de bolsa
- Nenhum arquivo de dados e versionado no repositorio (`data/` esta no `.gitignore`)
- Os logs de predicao gravados em `logs/predictions.log` contem as features de entrada — restrinja o acesso a esse arquivo em producao
- Nao registre dados brutos de estudantes alem do necessario para o monitoramento de drift

### Seguranca da API

- Todos os inputs sao validados via modelos Pydantic (`app/schemas.py`) antes de chegar ao modelo
- Os endpoints estao atualmente abertos sem autenticacao — destinados a uso interno e demonstracao
- Em producao, adicione autenticacao (API key ou OAuth 2.0) antes de expor a API externamente
- Implemente rate limiting para prevenir abuso

### Seguranca de Containers

A imagem e construida a partir de `python:3.13-slim`. Mantenha-a atualizada:

```bash
# Escanear vulnerabilidades
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasecurity/trivy image fiap_5mlt_datathon-app:latest

# Rodar com privilegios reduzidos
docker run --security-opt=no-new-privileges \
           --cap-drop=ALL \
           fiap_5mlt_datathon-app:latest
```

### Seguranca de Dependencias

```bash
# Verificar vulnerabilidades conhecidas
pip install pip-audit
pip-audit

# Linting de seguranca Python
pip install bandit
bandit -r src/ app/ monitoring/
```

### Seguranca do Modelo de ML

- **Validacao de entrada:** Os schemas Pydantic impõem tipos e intervalos de valores na fronteira da API
- **Arquivo do modelo:** `app/model/model.joblib` e um pipeline scikit-learn serializado — carregue apenas modelos de fontes confiáveis; nunca aceite arquivos de modelo vindos de entrada externa
- **Monitoramento de drift:** O dashboard Streamlit (`logs/predictions.log`) e para uso interno apenas; nao exponha a porta 8501 publicamente

## Checklist para Deploy em Producao

- [ ] HTTPS/TLS habilitado (reverse proxy: nginx ou traefik)
- [ ] Autenticacao da API implementada
- [ ] Rate limiting configurado
- [ ] Headers de seguranca configurados (HSTS, CSP, X-Frame-Options)
- [ ] Porta 8501 (dashboard) nao exposta publicamente
- [ ] Acesso ao diretorio `logs/` restrito
- [ ] Imagem do container escaneada por vulnerabilidades
- [ ] Auditoria de dependencias passando (`pip-audit`)
- [ ] Linting de seguranca passando (`bandit -r src/ app/`)

## Contribuidores: Checklist de Seguranca

Antes de abrir um PR, verifique:

- [ ] Nenhum dado de estudante ou informacao pessoal (PII) commitada no repositorio
- [ ] Nenhuma credencial ou segredo hardcoded
- [ ] Novos inputs validados na fronteira da API
- [ ] Campos sensiveis nao incluidos nos logs
- [ ] Dependencias adicionadas sao de fontes confiaveis e mantidas ativamente
