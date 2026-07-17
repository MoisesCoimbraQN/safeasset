# 🛡️ SafeAsset

**Aquisição de ativos de forma segura** — Dashboard analítico para avaliação de risco de crédito em carteiras de recebíveis (boletos), voltado à aquisição de ativos por FIDCs.

Desenvolvido em parceria acadêmica **FIAP + Núclea**, o SafeAsset combina engenharia de features, machine learning, detecção de fraude e dados macroeconômicos públicos (BCB/IBGE) em um único painel interativo, ajudando o analista de crédito a decidir **quais CNPJs vale a pena adquirir**.

---

## 📋 Índice

- [Visão geral](#-visão-geral)
- [Principais funcionalidades](#-principais-funcionalidades)
- [Arquitetura do projeto](#-arquitetura-do-projeto)
- [Pipeline de dados](#-pipeline-de-dados)
- [Dados de entrada](#-dados-de-entrada)
- [Instalação](#-instalação)
- [Como executar](#-como-executar)
- [Deploy (Render.com)](#-deploy-rendercom)
- [Stack tecnológica](#-stack-tecnológica)
- [Estrutura das abas do dashboard](#-estrutura-das-abas-do-dashboard)
- [Licença](#-licença)

---

## 🔎 Visão geral

O SafeAsset recebe três bases (base auxiliar de sacados, histórico de boletos e, opcionalmente, a carteira nova a ser adquirida) e produz:

- Um **Score FIDC (0–1000)** e um **rating de carteira** (A a E) por CNPJ pagador;
- Um **score de Machine Learning** como segunda opinião, com detecção de divergências entre os dois modelos;
- Um **indicador de detecção de fraude** (boletos duplicados, concentração excessiva de emitentes);
- Um **contexto macroeconômico setorial**, obtido em tempo real das APIs do Banco Central (SGS) e do IBGE (SIDRA);
- Uma **recomendação final automática** (Recomendado / Recomendado com Atenção / Não Recomendado) para a carteira nova.

---

## ✨ Principais funcionalidades

| Módulo | Descrição |
|---|---|
| 📁 **Upload de dados** | Base auxiliar de sacados, base de boletos históricos e carteira nova (opcional), via CSV |
| 🚨 **Detecção de Fraude** | Identifica boletos duplicados por ID ou por conteúdo (valor + vencimento + pagador + beneficiário) e sinaliza CNPJs com padrões suspeitos |
| 📊 **EDA** | Nulos, distribuições de scores/liquidez, atraso por UF, volume mensal de boletos, tipos de baixa |
| 🎯 **Target real** | Adimplência calculada a partir do comportamento real de pagamento dos boletos (sem *data leakage*), excluindo cancelamentos comerciais |
| 🤖 **Modelagem ML** | Treina e compara Regressão Logística, Random Forest e XGBoost (ou Gradient Boosting como fallback), com cross-validation (k=5), curvas ROC e matrizes de confusão |
| 🥇 **Score Final** | Fórmula ponderada de negócio: Liquidez do Sacado (40%) + Materialidade (25%) + Quantidade (15%) + Atraso (12%) + Inadimplência (8%) |
| 🏭 **Análise Setorial (CNAE)** | Perfil de risco por setor de atividade, com denominações oficiais CNAE 2.0 |
| 🌐 **Contexto Macro** | SELIC, IPCA, câmbio, PIB e inadimplência PJ por setor (BCB SGS + IBGE SIDRA), com fallback offline |
| 📈 **Indicador de Risco Setorial** | Z-score da inadimplência do setor predominante da carteira frente à média histórica de 24 meses |
| ⚠️ **Cobertura da Carteira Nova** | Segrega CNPJs com e sem histórico PCR, com análise dedicada para os "primeiro contato" |
| 📥 **Exportações CSV** | CNPJs suspeitos de fraude, CNPJs com/sem histórico, CNPJs com divergência Score × ML |

---

## 🏗️ Arquitetura do projeto

```
safeasset/
├── app.py           # Ponto de entrada — inicializa o Dash e registra os callbacks
├── layout.py         # Componentes visuais (sidebar, header, cards, KPIs, tabela de ranking)
├── callbacks.py       # Lógica reativa — uploads, filtros, execução do pipeline, montagem do dashboard
├── pipeline.py       # Feature engineering, target, modelagem ML e score final (sem dependência de Dash)
├── charts.py         # Geração de todas as figuras Plotly
├── macro.py          # Integração com APIs BCB SGS / IBGE SIDRA e cálculo do Score Macro Setorial
├── cnae_denominacoes.csv  # Tabela de de-para CNAE → denominação (necessário para análise setorial)
└── requirements.txt
```

Cada módulo tem responsabilidade única:
- **`pipeline.py`** é puro Python de ciência de dados (sem Dash/frontend), podendo ser testado isoladamente;
- **`charts.py`** só depende de `plotly` e `pipeline` (paleta de cores/ratings);
- **`macro.py`** encapsula toda a comunicação com APIs externas e tem fallback automático caso não haja internet;
- **`callbacks.py`** conecta tudo e monta o HTML final do dashboard;
- **`app.py`** apenas inicializa a aplicação.

---

## ⚙️ Pipeline de dados

O `run_pipeline()` (em `pipeline.py`) executa, nesta ordem:

1. **Detecção de duplicatas** nos boletos (fraude)
2. **Processamento dos boletos** (datas, flags de atraso/protesto/cancelamento)
3. **Feature engineering** — agregação por CNPJ pagador
4. **Definição do target** — adimplência real (sem leakage)
5. **Cálculo de correlação** das features com o target
6. **Treinamento de modelos ML** (Logistic Regression, Random Forest, XGBoost/GB) com validação cruzada
7. **Feature importance** do melhor modelo
8. **Score Final (Score FIDC)** — fórmula de negócio ponderada
9. **Probabilidade ML** — score auxiliar contínuo e detecção de divergências
10. **Perfil setorial por CNAE**

> ⚠️ **Sobre o target:** ele é calculado exclusivamente a partir do comportamento *real* de pagamento dos boletos (`vlr_baixa`), e as features usadas no modelo (`FEATURES` em `pipeline.py`) são exclusivamente indicadores do **sacado (pagador)**, medidos de forma independente dos boletos da carteira atual — evitando vazamento de dados (*data leakage*).

---

## 📂 Dados de entrada

| Arquivo | Obrigatório | Descrição |
|---|---|---|
| `base_auxiliar_fiap.csv` | ✅ | Cadastro e indicadores históricos por CNPJ (scores Núclea, liquidez, UF, CNAE, atraso médio etc.) |
| `base_boletos_fiap.csv` | ✅ | Histórico de boletos (emissão, vencimento, pagamento, valores, tipo de baixa) |
| `carteira_atual.csv` | Opcional | Carteira nova a ser adquirida — usada para análise de cobertura PCR e detecção de fraude específica |
| `cnae_denominacoes.csv` | Interno | Tabela de-para de códigos CNAE para nomes de setores (deve estar na mesma pasta de `pipeline.py`) |

---

## 💻 Instalação

```bash
git clone https://github.com/MoisesCoimbraQN/safeasset
cd safeasset

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### `requirements.txt` 

```
dash
dash-bootstrap-components
pandas
numpy
plotly
scikit-learn
xgboost
(...)

> O XGBoost é opcional — se não estiver instalado, o pipeline usa automaticamente `GradientBoostingClassifier` do scikit-learn como fallback.

---

## ▶️ Como executar

```bash
python app.py
```

Acesse **https://safeasset.onrender.com/**

**Passo a passo no dashboard:**
1. Faça upload da **Base Auxiliar** e da **Base de Boletos** na barra lateral;
2. Clique em **▶ Executar Análise Histórica**;
3. (Opcional) Faça upload da **Carteira Nova** e clique em **🔍 Analisar Carteira Nova**;
4. Ajuste, se desejar, os parâmetros de ML e os thresholds de detecção de fraude nos painéis recolhíveis;
5. Navegue pelas abas: Resumo, EDA, Fraude, Target & Correlação, Modelagem, Score Final, CNPJs Novos, CNPJs com Histórico e Macro.

---

## ☁️ Deploy (Render.com)

O `app.py` já expõe a variável `server` (WSGI) para uso com Gunicorn e lê a porta da variável de ambiente `PORT`, injetada automaticamente pelo Render.

**Start command sugerido:**
```bash
gunicorn app:server
```

Variáveis de ambiente úteis:
- `PORT` — definida automaticamente pelo Render
- `DASH_DEBUG` — `true`/`false` (padrão: `false`)

---

## 🧰 Stack tecnológica

- **[Dash](https://dash.plotly.com/)** + **Dash Bootstrap Components** — frontend web reativo
- **Plotly** — visualização de dados
- **Pandas / NumPy** — manipulação de dados
- **Scikit-learn** — modelagem preditiva, pipelines, calibração
- **XGBoost** *(opcional)* — modelo de boosting
- **APIs públicas:** [BCB SGS](https://www.bcb.gov.br/estabilidadefinanceira/sgs) e [IBGE SIDRA](https://sidra.ibge.gov.br/) para dados macroeconômicos, com fallback local em caso de indisponibilidade

---

## 🗂️ Estrutura das abas do dashboard

| Aba | Conteúdo |
|---|---|
| 📋 **Resumo** | Veredicto automático de aquisição, cobertura da carteira, indicadores consolidados |
| 📊 **EDA** | Análise exploratória geral e por setor CNAE |
| 🚨 **Fraude** | KPIs e gráficos de boletos duplicados, download de CNPJs suspeitos |
| 🎯 **Target & Correlação** | Distribuição do target e correlação das features |
| 🤖 **Modelagem** | Comparação de modelos, CV, ROC, matriz de confusão, importância de features |
| 🥇 **Score Final** | Score FIDC, ranking filtrável de CNPJs, segunda opinião via ML |
| ⚠️ **Aquisição – CNPJs Novos** | Análise de CNPJs sem histórico PCR (primeiro contato) |
| ✅ **Aquisição – CNPJs c/ Histórico** | Visão dedicada aos CNPJs da carteira nova já conhecidos na base |
| 🌐 **Macro** | Indicadores macroeconômicos, score macro setorial e indicador de risco setorial (z-score 24m) |

---

## 📄 Licença

Projeto acadêmico desenvolvido para fins educacionais no contexto **FIAP + Núclea**. 

---

*Feito com 💙 para tornar a aquisição de recebíveis mais segura e transparente.*
