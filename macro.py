# =============================================================================
# macro.py — SafeAsset
# Responsável por: busca de dados macroeconômicos externos (Camada 1)
#
# Fontes utilizadas (todas gratuitas e sem autenticação):
#   BCB SGS  — api.bcb.gov.br      (inadimplência PJ, SELIC, câmbio, crédito)
#   IBGE     — servicodados.ibge.gov.br  (PIB por atividade, PMC, PMS)
#
# Arquitetura:
#   1. Tenta buscar dados em tempo real via HTTP
#   2. Em caso de falha (sem internet, timeout) usa fallback com dados recentes
#   3. Calcula Score Setorial Macro (0–100) por divisão CNAE
#   4. Retorna dicionário completo para o dashboard
# =============================================================================

import json
import warnings
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BCB_BASE  = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/{n}?formato=json"
IBGE_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados/{ag}/periodos/last%20{n}/variaveis/{var}?localidades=N1[all]"

TIMEOUT = 8  # segundos por requisição

# Mapeamento CNAE divisão (2 dígitos) → setor macro BCB/IBGE
# Baseado na classificação de atividades do BCB para crédito PJ
CNAE_SETOR = {
    # Agropecuária
    **{str(d): "agro" for d in range(1, 4)},
    # Indústria
    **{str(d): "industria" for d in list(range(5, 10)) + list(range(10, 34))},
    # Construção civil
    **{str(d): "construcao" for d in range(41, 44)},
    # Comércio
    **{str(d): "comercio" for d in range(45, 48)},
    # Transporte e logística
    **{str(d): "transporte" for d in range(49, 54)},
    # Serviços de alojamento e alimentação
    **{str(d): "alojamento" for d in range(55, 57)},
    # Serviços de informação
    **{str(d): "info_ti" for d in range(58, 64)},
    # Finanças e seguros
    **{str(d): "financeiro" for d in range(64, 67)},
    # Saúde
    **{str(d): "saude" for d in range(86, 89)},
    # Educação
    "85": "educacao",
    # Demais serviços
    **{str(d): "servicos" for d in list(range(68, 86)) + list(range(89, 100))},
}

# Séries BCB SGS por tema
SERIES_BCB = {
    # Inadimplência PJ por setor (% carteira)
    "inadimplencia_pj_total":      21082,  # total PJ
    "inadimplencia_pj_industria":  21087,
    "inadimplencia_pj_comercio":   21088,
    "inadimplencia_pj_servicos":   21089,
    "inadimplencia_pj_agro":       21093,
    # Indicadores gerais
    "selic_meta":                    432,   # taxa SELIC meta (% a.a.)
    "cambio_brl_usd":                  1,   # câmbio R$/USD
    "ipca_acumulado_12m":           13522,  # IPCA 12 meses (%)
    "concessao_credito_pj":         20629,  # concessões PJ (R$ milhões)
    "spread_pj":                    20786,  # spread médio PJ (p.p.)
}

# Dados de fallback — valores de referência recentes (atualizar periodicamente)
FALLBACK = {
    "inadimplencia_pj_total":     3.6,
    "inadimplencia_pj_industria": 2.8,
    "inadimplencia_pj_comercio":  3.9,
    "inadimplencia_pj_servicos":  4.1,
    "inadimplencia_pj_agro":      1.9,
    "selic_meta":                10.75,
    "cambio_brl_usd":             5.10,
    "ipca_acumulado_12m":         4.62,
    "concessao_credito_pj":   280000.0,
    "spread_pj":                  15.8,
    "pib_variacao_anual":          2.9,
    "pib_variacao_trimestral":     0.8,
    "pmc_variacao":                1.2,   # Pesquisa Mensal do Comércio
    "pms_variacao":                2.1,   # Pesquisa Mensal de Serviços
    "caged_saldo_mensal":      100000,
    "data_referencia": "2025-T3 (fallback — sem conexão com a internet)",
    "fonte": "fallback",
}


# ─────────────────────────────────────────────────────────────────────────────
# CAMADA DE ACESSO — HTTP com timeout e fallback
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(url: str) -> dict | list | None:
    """Faz GET JSON com User-Agent de browser. Retorna None em caso de falha."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, */*',
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        return None


def _bcb_serie(codigo: int, n: int = 4) -> float | None:
    """Busca o último valor disponível de uma série BCB SGS."""
    url  = BCB_BASE.format(codigo=codigo, n=n)
    data = _get_json(url)
    if data and isinstance(data, list) and len(data) > 0:
        try:
            return float(data[-1]['valor'].replace(',', '.'))
        except Exception:
            return None
    return None


def _bcb_serie_historico(codigo: int, n: int = 8) -> pd.DataFrame:
    """Retorna série histórica de n períodos de uma série BCB SGS."""
    url  = BCB_BASE.format(codigo=codigo, n=n)
    data = _get_json(url)
    if data and isinstance(data, list):
        try:
            df = pd.DataFrame(data)
            df['valor'] = df['valor'].str.replace(',', '.').astype(float)
            df['data']  = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
            return df[['data', 'valor']].dropna()
        except Exception:
            pass
    return pd.DataFrame(columns=['data', 'valor'])


def _ibge_pib_variacao(n: int = 4) -> dict:
    """Busca variação do PIB trimestral e anual (IBGE SIDRA agregado 1846)."""
    # var 585 = variação acumulada 4 trimestres, var 584 = variação trimestral
    result = {}
    for var, key in [(585, 'pib_variacao_anual'), (584, 'pib_variacao_trimestral')]:
        url  = IBGE_BASE.format(ag=1846, n=n, var=var)
        data = _get_json(url)
        if data:
            try:
                series = data[0]['resultados'][0]['series'][0]['serie']
                ultimo = list(series.values())[-1]
                result[key] = float(ultimo)
            except Exception:
                result[key] = FALLBACK[key]
        else:
            result[key] = FALLBACK[key]
    return result


def _ibge_pmc_variacao() -> float:
    """Pesquisa Mensal do Comércio — variação % mensal (agregado 8186, var 11709)."""
    url  = IBGE_BASE.format(ag=8186, n=3, var=11709)
    data = _get_json(url)
    if data:
        try:
            series = data[0]['resultados'][0]['series'][0]['serie']
            return float(list(series.values())[-1])
        except Exception:
            pass
    return FALLBACK['pmc_variacao']


def _ibge_pms_variacao() -> float:
    """Pesquisa Mensal de Serviços — variação % mensal (agregado 8163, var 11622)."""
    url  = IBGE_BASE.format(ag=8163, n=3, var=11622)
    data = _get_json(url)
    if data:
        try:
            series = data[0]['resultados'][0]['series'][0]['serie']
            return float(list(series.values())[-1])
        except Exception:
            pass
    return FALLBACK['pms_variacao']


# ─────────────────────────────────────────────────────────────────────────────
# BUSCA PRINCIPAL — coleta todos os indicadores
# ─────────────────────────────────────────────────────────────────────────────

def buscar_indicadores() -> dict:
    """
    Coleta todos os indicadores macroeconômicos externos.
    Tenta APIs em tempo real; usa fallback se não houver conexão.

    Retorna
    -------
    dict com chaves:
        indicadores   : valores mais recentes de cada série
        historico     : DataFrames históricos para gráficos
        pib           : variações do PIB
        data_coleta   : timestamp da coleta
        fonte         : 'api' ou 'fallback'
    """
    resultado = {
        'indicadores': {},
        'historico':   {},
        'pib':         {},
        'data_coleta': datetime.now().strftime('%d/%m/%Y %H:%M'),
        'fonte':       'api',
    }

    # ── Indicadores pontuais BCB ──────────────────────────────────
    for nome, codigo in SERIES_BCB.items():
        val = _bcb_serie(codigo, n=2)
        resultado['indicadores'][nome] = val if val is not None else FALLBACK.get(nome)
        if val is None:
            resultado['fonte'] = 'fallback'

    # ── Séries históricas para gráficos ───────────────────────────
    series_graf = {
        'inadimplencia_pj_total':     21082,
        'inadimplencia_pj_comercio':  21088,
        'inadimplencia_pj_servicos':  21089,
        'inadimplencia_pj_industria': 21087,
        'selic_meta':                   432,
        'ipca_acumulado_12m':         13522,
    }
    for nome, codigo in series_graf.items():
        df_hist = _bcb_serie_historico(codigo, n=12)
        resultado['historico'][nome] = df_hist

    # ── PIB IBGE ──────────────────────────────────────────────────
    pib = _ibge_pib_variacao()
    resultado['pib'] = pib
    resultado['indicadores'].update(pib)

    # ── Comércio e Serviços ───────────────────────────────────────
    resultado['indicadores']['pmc_variacao'] = _ibge_pmc_variacao()
    resultado['indicadores']['pms_variacao'] = _ibge_pms_variacao()

    # Se todos vieram do fallback, marcar data de referência
    if resultado['fonte'] == 'fallback':
        resultado['data_coleta'] = FALLBACK['data_referencia']

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# SCORE SETORIAL MACRO (0–100)
# ─────────────────────────────────────────────────────────────────────────────

# Inadimplência média por setor usada como referência de risco
_INAD_REF = {
    'agro':       1.9,
    'industria':  2.8,
    'comercio':   3.9,
    'servicos':   4.1,
    'construcao': 5.2,
    'transporte': 4.8,
    'alojamento': 5.5,
    'info_ti':    2.1,
    'financeiro': 1.5,
    'saude':      2.4,
    'educacao':   3.0,
}
_INAD_REF_DEFAULT = 4.0


def calcular_score_macro_setor(indicadores: dict, setor: str) -> dict:
    """
    Calcula o Score Macro Setorial (0–100) para um setor específico.

    Componentes e pesos:
      40% — Inadimplência setorial  (invertida: menor inad → maior score)
      25% — Crescimento PIB anual   (positivo → melhor)
      15% — SELIC                   (invertida: SELIC alta → risco maior)
      10% — IPCA 12m                (invertida: inflação alta → risco maior)
      10% — Indicador setorial      (PMC para comércio, PMS para serviços, PIB demais)

    Retorna dict com score final e componentes individuais.
    """
    ind = indicadores

    # ── Componente 1: Inadimplência setorial ─────────────────────
    inad_chave = f"inadimplencia_pj_{setor}" if setor in [
        'industria', 'comercio', 'servicos', 'agro'
    ] else 'inadimplencia_pj_total'
    inad_val = ind.get(inad_chave, ind.get('inadimplencia_pj_total', 4.0))
    inad_ref = _INAD_REF.get(setor, _INAD_REF_DEFAULT)
    # Normaliza: inad=0% → 100pts; inad=10% → 0pts
    comp_inad = max(0, min(100, 100 - (inad_val / 10.0) * 100))

    # ── Componente 2: PIB anual ───────────────────────────────────
    pib_anual = ind.get('pib_variacao_anual', 2.0)
    # PIB -4% → 0pts; PIB +6% → 100pts
    comp_pib = max(0, min(100, (pib_anual + 4.0) / 10.0 * 100))

    # ── Componente 3: SELIC ───────────────────────────────────────
    selic = ind.get('selic_meta', 10.75)
    # SELIC 2% → 100pts; SELIC 15%+ → 0pts
    comp_selic = max(0, min(100, 100 - ((selic - 2.0) / 13.0) * 100))

    # ── Componente 4: IPCA ────────────────────────────────────────
    ipca = ind.get('ipca_acumulado_12m', 4.5)
    # IPCA 0% → 100pts; IPCA 12%+ → 0pts
    comp_ipca = max(0, min(100, 100 - (ipca / 12.0) * 100))

    # ── Componente 5: Indicador setorial específico ───────────────
    if setor == 'comercio':
        var_set = ind.get('pmc_variacao', 1.0)
    elif setor in ('servicos', 'alojamento', 'info_ti', 'financeiro', 'saude', 'educacao'):
        var_set = ind.get('pms_variacao', 1.5)
    else:
        var_set = pib_anual
    # var -5% → 0pts; var +5% → 100pts
    comp_setor = max(0, min(100, (var_set + 5.0) / 10.0 * 100))

    # ── Score final ponderado ─────────────────────────────────────
    score = (
        comp_inad  * 0.40 +
        comp_pib   * 0.25 +
        comp_selic * 0.15 +
        comp_ipca  * 0.10 +
        comp_setor * 0.10
    )

    # Classificação qualitativa
    if score >= 75:
        nivel, cor = "Favorável",   "#00cc70"
    elif score >= 55:
        nivel, cor = "Neutro",      "#F59E0B"
    elif score >= 35:
        nivel, cor = "Atenção",     "#ff6b35"
    else:
        nivel, cor = "Desfavorável","#EF4444"

    return {
        'score':           round(score, 1),
        'nivel':           nivel,
        'cor':             cor,
        'comp_inadimp':    round(comp_inad, 1),
        'comp_pib':        round(comp_pib, 1),
        'comp_selic':      round(comp_selic, 1),
        'comp_ipca':       round(comp_ipca, 1),
        'comp_setor':      round(comp_setor, 1),
        'inad_setor_pct':  round(inad_val, 2),
        'pib_anual_pct':   round(pib_anual, 2),
        'selic_pct':       round(selic, 2),
        'ipca_pct':        round(ipca, 2),
        'var_setor_pct':   round(var_set, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENRIQUECIMENTO DO PERFIL CNAE COM SCORE MACRO
# ─────────────────────────────────────────────────────────────────────────────

def enriquecer_perfil_com_macro(perfil_cnae: pd.DataFrame,
                                 indicadores: dict) -> pd.DataFrame:
    """
    Adiciona o Score Macro Setorial ao DataFrame de perfil por CNAE.
    Usa os 2 primeiros dígitos do cd_cnae_fmt para mapear ao setor macro.

    Retorna perfil_cnae com colunas adicionais:
        setor_macro, score_macro, nivel_macro, cor_macro
    """
    df = perfil_cnae.copy()

    def get_setor(cnae_fmt):
        try:
            div = str(cnae_fmt).split('.')[0]  # ex: "47.81-4" → "47"
            return CNAE_SETOR.get(div, 'servicos')
        except Exception:
            return 'servicos'

    def get_score(setor):
        r = calcular_score_macro_setor(indicadores, setor)
        return r['score'], r['nivel'], r['cor']

    df['setor_macro'] = df['cd_cnae_fmt'].apply(get_setor)
    scores = df['setor_macro'].apply(lambda s: get_score(s))
    df['score_macro'] = scores.apply(lambda x: x[0])
    df['nivel_macro']  = scores.apply(lambda x: x[1])
    df['cor_macro']    = scores.apply(lambda x: x[2])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL — chamada pelo run_pipeline()
# ─────────────────────────────────────────────────────────────────────────────

def run_macro(perfil_cnae: pd.DataFrame) -> dict:
    """
    Ponto de entrada principal. Chamada por run_pipeline() após calcular_perfil_cnae().

    Retorna dict com:
        indicadores      : dict de valores mais recentes
        historico        : dict de DataFrames históricos
        pib              : variações do PIB
        perfil_enriquecido: perfil_cnae + colunas macro
        scores_setor     : dict setor → score detalhado
        data_coleta      : string com timestamp
        fonte            : 'api' ou 'fallback'
    """
    # 1. Buscar indicadores externos
    macro = buscar_indicadores()

    # 2. Calcular score por setor único presente na carteira
    setores_unicos = {}
    if not perfil_cnae.empty and 'cd_cnae_fmt' in perfil_cnae.columns:
        for cnae_fmt in perfil_cnae['cd_cnae_fmt'].dropna().unique():
            try:
                div   = str(cnae_fmt).split('.')[0]
                setor = CNAE_SETOR.get(div, 'servicos')
                if setor not in setores_unicos:
                    setores_unicos[setor] = calcular_score_macro_setor(
                        macro['indicadores'], setor
                    )
            except Exception:
                pass

    # 3. Enriquecer perfil CNAE com scores
    perfil_enriquecido = enriquecer_perfil_com_macro(
        perfil_cnae, macro['indicadores']
    )

    return {
        'indicadores':        macro['indicadores'],
        'historico':          macro['historico'],
        'pib':                macro['pib'],
        'perfil_enriquecido': perfil_enriquecido,
        'scores_setor':       setores_unicos,
        'data_coleta':        macro['data_coleta'],
        'fonte':              macro['fonte'],
    }
