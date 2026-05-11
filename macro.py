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

TIMEOUT = 4  # segundos por requisição

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
    # Inadimplência PJ Total SFN (série 21083) — única série por setor disponível via API BCB
    # O BCB não publica inadimplência por setor CNAE no SGS
    "inadimplencia_pj_total":      21083,  # PJ Total SFN — série correta
    "inadimplencia_pj_industria":  21083,  # proxy PJ Total (sem série setorial BCB)
    "inadimplencia_pj_comercio":   21083,  # proxy PJ Total (sem série setorial BCB)
    "inadimplencia_pj_servicos":   21083,  # proxy PJ Total (sem série setorial BCB)
    "inadimplencia_pj_agro":       21083,  # proxy PJ Total (sem série setorial BCB)
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



def calcular_quartis_macro() -> tuple:
    """
    Calcula P25 e P75 do score macro histórico UMA ÚNICA VEZ.
    Usa séries BCB: inadimplência total (21082), SELIC (432), IPCA (13522).
    Retorna (p25, p75) para uso em calcular_score_macro_setor.
    """
    try:
        import numpy as _np
        _dfs = [
            _bcb_serie_historico(21082, n=20),
            _bcb_serie_historico(432,   n=20),
            _bcb_serie_historico(13522, n=20),
        ]
        _n = min(len(d) for d in _dfs)
        if _n >= 8:
            _sh = []
            for _i in range(_n):
                _ci = max(0, min(100, 100-(float(_dfs[0]['valor'].iloc[_i])/10)*100))
                _cs = max(0, min(100, 100-((float(_dfs[1]['valor'].iloc[_i])-2)/13)*100))
                _cp = max(0, min(100, 100-(float(_dfs[2]['valor'].iloc[_i])/12)*100))
                _sh.append(_ci*0.50 + _cs*0.30 + _cp*0.20)
            return float(_np.percentile(_sh, 25)), float(_np.percentile(_sh, 75))
    except Exception:
        pass
    return 45.0, 65.0


def calcular_score_macro_setor(indicadores: dict, setor: str,
                               p25: float = 45.0, p75: float = 65.0) -> dict:
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

    # Fórmula: 3 componentes BCB — inadimplência(50%) + SELIC(30%) + IPCA(20%)
    comp_inad  = max(0, min(100, 100 - (inad_val / 10.0) * 100))
    selic      = ind.get('selic_meta', 10.75)
    comp_selic = max(0, min(100, 100 - ((selic - 2.0) / 13.0) * 100))
    ipca       = ind.get('ipca_acumulado_12m', 4.5)
    comp_ipca  = max(0, min(100, 100 - (ipca / 12.0) * 100))
    # Manter para compatibilidade com o retorno
    pib_anual  = ind.get('pib_variacao_anual', 2.0)
    comp_pib   = max(0, min(100, (pib_anual + 4.0) / 10.0 * 100))
    comp_setor = comp_pib  # fallback
    var_set    = pib_anual  # mantido para compatibilidade do retorno

    score = comp_inad * 0.50 + comp_selic * 0.30 + comp_ipca * 0.20

    if score >= p75:
        nivel, cor = "Favorável",    "#00cc70"
    elif score >= p25:
        nivel, cor = "Neutro",       "#F59E0B"
    elif score >= p25 * 0.7:
        nivel, cor = "Atenção",      "#ff6b35"
    else:
        nivel, cor = "Desfavorável", "#EF4444"

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


# ─────────────────────────────────────────────────────────────────────────────
# INDICADOR DE INADIMPLÊNCIA PJ — z-score histórico 19 meses (BCB 21083)
# ─────────────────────────────────────────────────────────────────────────────

# Séries BCB por setor macro
# IMPORTANTE: O BCB não disponibiliza inadimplência por setor CNAE via SGS.
# As séries 21082–21102 são por modalidade de crédito (capital de giro, cheque, etc.)
# A série correta para inadimplência PJ geral é 21083 (Inadimplência PJ Total — SFN).
# Usamos 21083 como série única para todos os setores, sendo transparentes na interface.
SERIES_SETOR = {
    'agro':       21083,
    'industria':  21083,
    'comercio':   21083,
    'servicos':   21083,
    'construcao': 21083,
    'transporte': 21083,
    'alojamento': 21083,
    'info_ti':    21083,
    'financeiro': 21083,
    'saude':      21083,
    'educacao':   21083,
}

# Fallback histórico — 19 meses históricos + 1 atual (índice 19) — BCB 21083
# Valores baseados no comportamento histórico BCB por setor (referência 2023-2025)
# Cada setor tem padrão de sazonalidade e tendência distintos → z-scores diferenciados
# Dados reais BCB série 21083 — Inadimplência PJ Total SFN — ago/2024 a mar/2026
# Fonte: api.bcb.gov.br/dados/serie/bcdata.sgs.21083/dados/ultimos/20?formato=json
# Coletado em mai/2026. Para atualizar, consultar a URL acima.
# NOTA: O BCB não publica inadimplência por setor CNAE via API SGS.
# Todos os setores usam a mesma série PJ Total como referência de mercado.
_SERIE_21083 = [2.36,2.37,2.33,2.31,2.03,2.23,2.30,2.25,2.46,2.40,
                2.40,2.51,2.56,2.51,2.56,2.48,2.43,2.60,2.78,
                2.75]

FALLBACK_HISTORICO = {
    'agro':       _SERIE_21083,
    'industria':  _SERIE_21083,
    'comercio':   _SERIE_21083,
    'servicos':   _SERIE_21083,
    'construcao': _SERIE_21083,
    'transporte': _SERIE_21083,
    'alojamento': _SERIE_21083,
    'info_ti':    _SERIE_21083,
    'financeiro': _SERIE_21083,
    'saude':      _SERIE_21083,
    'educacao':   _SERIE_21083,
    'default':    _SERIE_21083,
}


def identificar_setor_predominante(df_aux: pd.DataFrame,
                                    df_bol: pd.DataFrame) -> tuple:
    """
    Identifica o setor BCB predominante da carteira pelo valor total dos boletos.

    Retorna (setor_bcb, pct_valor, vlr_total_setor, setor_label)
    """
    import pandas as _pd

    # Merge para associar CNAE ao valor dos boletos
    aux_cnae = df_aux[['id_cnpj', 'cd_cnae_prin']].copy()
    bol_val  = df_bol.groupby('id_pagador')['vlr_nominal'].sum().reset_index()
    bol_val.columns = ['id_cnpj', 'vlr_total']

    merged = bol_val.merge(aux_cnae, on='id_cnpj', how='left')
    merged['cd_cnae_prin'] = merged['cd_cnae_prin'].fillna('0')

    # Mapear CNAE → setor BCB
    merged['setor_bcb'] = merged['cd_cnae_prin'].apply(
        lambda x: CNAE_SETOR.get(str(x)[:2], 'servicos')
    )

    # Agrupar por setor e calcular % do valor total
    por_setor  = merged.groupby('setor_bcb')['vlr_total'].sum()
    vlr_total  = por_setor.sum()
    pct_setor  = (por_setor / vlr_total * 100).sort_values(ascending=False)

    setor_pred  = pct_setor.index[0]
    pct_pred    = pct_setor.iloc[0]
    vlr_pred    = por_setor.iloc[0]

    SETOR_LABEL = {
        'agro': 'Agronegócio', 'industria': 'Indústria',
        'comercio': 'Comércio', 'servicos': 'Serviços',
        'construcao': 'Construção Civil', 'transporte': 'Transporte',
        'alojamento': 'Alojamento e Alimentação', 'info_ti': 'TI e Informação',
        'financeiro': 'Financeiro', 'saude': 'Saúde', 'educacao': 'Educação',
    }
    label = SETOR_LABEL.get(setor_pred, setor_pred.title())

    return setor_pred, round(pct_pred, 1), round(vlr_pred, 2), label


def calcular_indicador_risco_setorial(df_aux: pd.DataFrame,
                                       df_bol: pd.DataFrame) -> dict:
    """
    Calcula o Indicador de Risco Setorial baseado no z-score histórico de
    19 meses da inadimplência PJ Total SFN (BCB 21083).

    Lógica:
      1. Identifica setor predominante por valor dos boletos
      2. Busca 25 meses da série BCB (24 histórico + 1 atual)
      3. Calcula média e desvio padrão dos 19 meses históricos
      4. Calcula z-score do valor atual
      5. Classifica tag:
           z ≤ -0.5 → Recomendado  (inadimplência abaixo da média)
           -0.5 < z < +0.5 → Regular (inadimplência na média)
           z ≥ +0.5 → Atenção       (inadimplência acima da média)

    Retorna dict com todos os dados para o dashboard.
    """
    # 1. Setor predominante
    setor, pct_valor, vlr_setor, setor_label = identificar_setor_predominante(
        df_aux, df_bol
    )

    # 2. Série histórica — 25 meses (24 + atual)
    codigo_serie = SERIES_SETOR.get(setor, 21082)
    df_hist = _bcb_serie_historico(codigo_serie, n=20)
    fonte = 'api'

    if df_hist.empty or len(df_hist) < 5:
        # Fallback com dados de referência
        vals_hist = FALLBACK_HISTORICO.get(setor, FALLBACK_HISTORICO['default'])
        df_hist = pd.DataFrame({
            'data':  pd.date_range(end=pd.Timestamp.today(), periods=20, freq='ME'),
            'valor': vals_hist
        })
        fonte = 'fallback'

    # Garantir ordenação cronológica
    df_hist = df_hist.sort_values('data').reset_index(drop=True)

    # 3. Separar histórico (19) e valor atual (último)
    historico = df_hist.iloc[:-1]['valor'].values  # 19 meses anteriores
    valor_atual = float(df_hist.iloc[-1]['valor'])
    data_atual  = df_hist.iloc[-1]['data']

    media_24m  = float(np.mean(historico))
    desvio_24m = float(np.std(historico, ddof=1))

    # 4. Z-score
    if desvio_24m > 0:
        z = (valor_atual - media_24m) / desvio_24m
    else:
        z = 0.0

    # 5. Tag
    if z <= -0.5:
        tag, cor, emoji = 'Recomendado', '#00cc70', '✅'
        interpretacao = (
            f"A inadimplência PJ geral está {abs(z):.2f} desvios abaixo da média "
            f"histórica dos últimos 19 meses — momento favorável para aquisição."
        )
    elif z < 0.5:
        tag, cor, emoji = 'Regular', '#F59E0B', '🔶'
        interpretacao = (
            f"A inadimplência PJ geral está dentro da faixa histórica normal "
            f"(z-score: {z:+.2f}) — momento neutro para aquisição."
        )
    else:
        tag, cor, emoji = 'Atenção', '#EF4444', '🚨'
        interpretacao = (
            f"A inadimplência PJ geral está {z:.2f} desvios acima da média "
            f"histórica dos últimos 19 meses — momento desfavorável para aquisição."
        )

    print(f"[SafeAsset] Risco Setorial — setor: {setor_label} | "
          f"inadimp: {valor_atual:.2f}% | z: {z:+.2f} → {tag}")

    return {
        'tag':           tag,
        'cor':           cor,
        'emoji':         emoji,
        'interpretacao': interpretacao,
        'setor':         setor,
        'setor_label':   setor_label,
        'pct_valor':     pct_valor,
        'vlr_setor':     vlr_setor,
        'codigo_serie':  codigo_serie,
        'valor_atual':   round(valor_atual, 2),
        'media_24m':     round(media_24m, 2),
        'desvio_24m':    round(desvio_24m, 2),
        'z_score':       round(z, 3),
        'banda_sup':     round(media_24m + 0.5 * desvio_24m, 2),
        'banda_inf':     round(media_24m - 0.5 * desvio_24m, 2),
        'df_historico':  df_hist,
        'fonte':         fonte,
        'data_atual':    data_atual.strftime('%b/%Y') if hasattr(data_atual, 'strftime') else str(data_atual),
    }

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
        'indicador_risco':    None,  # calculado pelo callback com df_bol
    }