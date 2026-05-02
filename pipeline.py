# =============================================================================
# pipeline.py — SafeAsset
# Responsável por: Feature Engineering, Target, Modelagem ML e Score Final
# Sem dependências de Dash ou frontend — Python puro de ciência de dados
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────


# Removidas: 'sacado_indice_liquidez_1m', 'score_materialidade_v2', 'media_atraso_dias' - estão no target
FEATURES = [
    'cedente_indice_liquidez_1m',
    'score_materialidade_evolucao',
    'indicador_liquidez_quantitativo_3m',
    'share_vl_inad_pag_bol_6_a_15d',
    'score_quantidade_v2',
    'bol_qtd_total', 'bol_pct_atrasado', 'bol_pct_sem_pgto',
    'bol_taxa_recuperacao', 'bol_atraso_medio', 'bol_pct_protestado',]

RATING_COLOR = {
    'A — Excelente':      '#00ff88',
    'B — Bom':            '#00d4ff',
    'C — Risco Moderado': '#ffd700',
    'D — Risco Elevado':  '#ff6b35',
    'E — Alto Risco':     '#ff2d55',
}

RATING_BINS   = [0, 300, 500, 700, 850, 1001]
RATING_LABELS = ['E — Alto Risco', 'D — Risco Elevado', 'C — Risco Moderado', 'B — Bom', 'A — Excelente']

# Thresholds de detecção de fraude (configuráveis)
FRAUDE_PCT_DUP_THRESH   = 0.05   # % mínimo de boletos duplicados para levantar alerta
FRAUDE_N_EMITENTES_THRESH = 10   # nº de beneficiários distintos por pagador para alertar

# Tabela de denominações CNAE (carregada uma vez ao importar o módulo)
import os as _os
_CNAE_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'cnae_denominacoes.csv')
try:
    CNAE_DENOMINACOES = pd.read_csv(_CNAE_CSV)
except FileNotFoundError:
    CNAE_DENOMINACOES = pd.DataFrame(columns=['cd_cnae_fmt', 'denominacao'])



# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Conversão de código CNAE
# ─────────────────────────────────────────────────────────────────────────────

def _formatar_cnae(val) -> str:
    """
    Converte o código CNAE de 7 dígitos inteiro (ex: 4645101) para o formato
    XX.XX-D usado na tabela oficial (ex: 46.45-1).
    Usa apenas os 5 primeiros dígitos conforme estrutura da CNAE 2.0.
    Retorna None para valores inválidos.
    """
    try:
        s = str(int(val))[:5]
        if len(s) < 5:
            return None
        return f"{s[:2]}.{s[2:4]}-{s[4]}"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE SETORIAL — Perfil por CNAE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_perfil_cnae(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os CNPJs por área de atuação (CNAE) e enriquece com a denominação
    oficial da CNAE 2.0.

    Para cada CNAE calcula:
      - qtd_cnpjs          : número de CNPJs na carteira com aquele CNAE
      - score_medio        : score FIDC médio dos CNPJs do setor
      - pct_suspeitos      : % de CNPJs com flag_risco_fraude = 1
      - rating_predominante: rating mais frequente entre os CNPJs do setor
      - denominacao        : descrição oficial da atividade econômica

    Parâmetros
    ----------
    df_full : pd.DataFrame
        Base completa pós-pipeline (com score_fidc, rating_carteira,
        cd_cnae_prin e flag_risco_fraude).

    Retorna
    -------
    pd.DataFrame ordenado por qtd_cnpjs decrescente.
    """
    df = df_full.copy()

    # Converter código para formato padrão de comparação
    df['cd_cnae_fmt'] = df['cd_cnae_prin'].apply(_formatar_cnae)

    # Garantir que flag_risco_fraude existe (pode não existir se fraude não rodou)
    if 'flag_risco_fraude' not in df.columns:
        df['flag_risco_fraude'] = 0

    # Agregação por CNAE
    perfil = df.groupby('cd_cnae_fmt').agg(
        qtd_cnpjs           = ('id_cnpj',          'count'),
        score_medio         = ('score_fidc',         'mean'),
        pct_suspeitos       = ('flag_risco_fraude',  'mean'),
        rating_predominante = ('rating_carteira',
                               lambda x: x.value_counts().index[0]
                               if len(x) > 0 else 'N/D'),
    ).reset_index()

    perfil['score_medio']   = perfil['score_medio'].round(0).astype(int)
    perfil['pct_suspeitos'] = (perfil['pct_suspeitos'] * 100).round(1)

    # Enriquecer com denominação oficial
    perfil = perfil.merge(CNAE_DENOMINACOES, on='cd_cnae_fmt', how='left')
    perfil['denominacao'] = perfil['denominacao'].fillna('Não identificado')

    # Abreviar denominações longas para exibição em gráfico
    perfil['denominacao_curta'] = perfil['denominacao'].apply(
        lambda x: x[:45] + '…' if len(str(x)) > 45 else x
    )

    return perfil.sort_values('qtd_cnpjs', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 2B — DETECÇÃO DE BOLETOS DUPLICADOS E RISCO DE FRAUDE
# ─────────────────────────────────────────────────────────────────────────────

def detectar_duplicatas(df_bol: pd.DataFrame,
                        pct_dup_thresh: float = FRAUDE_PCT_DUP_THRESH,
                        n_emitentes_thresh: int = FRAUDE_N_EMITENTES_THRESH) -> dict:
    """
    Analisa a base de boletos BRUTA e identifica três tipos de duplicação:

      Tipo 1 — id_boleto repetido: mesmo identificador emitido mais de uma vez.
               Indica possível reaproveitamento fraudulento de boleto já liquidado.

      Tipo 2 — Mesma combinação (vlr_nominal + dt_vencimento + id_pagador) com
               id_boleto diferente: duplicata "disfarçada" que tenta parecer
               um boleto novo mas repete os mesmos atributos comerciais.

      Tipo 3 — Concentração de emitentes: um pagador com muitos beneficiários
               distintos emitindo contra ele pode indicar esquema de boletos
               fictícios coordenado entre múltiplos emitentes.

    Parâmetros
    ----------
    df_bol              : Base de boletos (pode ser ainda não processada)
    pct_dup_thresh      : % mínimo de boletos duplicados para flag de alerta
    n_emitentes_thresh  : Nº de beneficiários distintos para flag de alerta

    Retorna
    -------
    dict com chaves:
      df_bol_marcado     : df_bol com colunas 'flag_dup_id', 'flag_dup_conteudo'
      resumo_duplicatas  : DataFrame — 1 linha por grupo duplicado, com contagem
      fraude_por_cnpj    : DataFrame — métricas de fraude agregadas por pagador
      stats              : dict com totais gerais (para KPIs do dashboard)
    """
    df = df_bol.copy()

    # ── Tipo 1: id_boleto repetido ────────────────────────────────────────
    contagem_id = df.groupby('id_boleto')['id_boleto'].transform('count')
    df['flag_dup_id'] = (contagem_id > 1).astype(int)

    # ── Tipo 2: mesmo conteúdo, id diferente ─────────────────────────────
    # Chave composta: valor + vencimento + pagador + beneficiário
    chave_cols = ['vlr_nominal', 'dt_vencimento', 'id_pagador', 'id_beneficiario']
    # Garantir que dt_vencimento está no formato correto para agrupamento
    df['_dt_venc_str'] = pd.to_datetime(df['dt_vencimento'], errors='coerce').dt.strftime('%Y-%m-%d')
    chave_cols_str = ['vlr_nominal', '_dt_venc_str', 'id_pagador', 'id_beneficiario']
    contagem_conteudo = df.groupby(chave_cols_str)['id_boleto'].transform('count')
    df['flag_dup_conteudo'] = (contagem_conteudo > 1).astype(int)

    # Flag unificado: duplicata de qualquer tipo
    df['flag_duplicata'] = ((df['flag_dup_id'] == 1) | (df['flag_dup_conteudo'] == 1)).astype(int)

    # ── Resumo dos grupos duplicados ─────────────────────────────────────
    dup_id = (df[df['flag_dup_id'] == 1]
              .groupby('id_boleto')
              .agg(qtd_ocorrencias=('id_boleto', 'count'),
                   id_pagador=('id_pagador', 'first'),
                   vlr_nominal=('vlr_nominal', 'first'))
              .reset_index()
              .assign(tipo_duplicata='ID repetido'))

    dup_cont = (df[df['flag_dup_conteudo'] == 1]
                .groupby(chave_cols_str)
                .agg(qtd_ocorrencias=('id_boleto', 'count')) # REMOVED: id_pagador=('id_pagador', 'first')
                .reset_index()
                .rename(columns={'vlr_nominal': 'vlr_nominal',
                                 '_dt_venc_str': 'dt_vencimento'})
                .assign(tipo_duplicata='Conteúdo idêntico'))

    resumo_cols = ['id_pagador', 'vlr_nominal', 'qtd_ocorrencias', 'tipo_duplicata']
    resumo = pd.concat([
        dup_id[['id_pagador', 'vlr_nominal', 'qtd_ocorrencias', 'tipo_duplicata']],
        dup_cont[['id_pagador', 'vlr_nominal', 'qtd_ocorrencias', 'tipo_duplicata']],
    ], ignore_index=True).sort_values('qtd_ocorrencias', ascending=False)

    # ── Tipo 3 + métricas por pagador ────────────────────────────────────
    fraude_cnpj = df.groupby('id_pagador').agg(
        bol_qtd_total          = ('id_boleto',        'count'),
        bol_qtd_dup_id         = ('flag_dup_id',      'sum'),
        bol_qtd_dup_conteudo   = ('flag_dup_conteudo','sum'),
        bol_qtd_dup_total      = ('flag_duplicata',   'sum'),
        bol_n_emitentes        = ('id_beneficiario',  'nunique'),
    ).reset_index()

    fraude_cnpj['bol_pct_duplicado'] = (
        fraude_cnpj['bol_qtd_dup_total'] / fraude_cnpj['bol_qtd_total']
    ).fillna(0)

    # Flag de risco de fraude: excede threshold em qualquer critério
    fraude_cnpj['flag_risco_fraude'] = (
        (fraude_cnpj['bol_pct_duplicado']  >= pct_dup_thresh) |
        (fraude_cnpj['bol_n_emitentes']    >= n_emitentes_thresh)
    ).astype(int)

    fraude_cnpj['motivo_alerta'] = ''
    mask_dup = fraude_cnpj['bol_pct_duplicado'] >= pct_dup_thresh
    mask_emit = fraude_cnpj['bol_n_emitentes']  >= n_emitentes_thresh
    fraude_cnpj.loc[mask_dup & ~mask_emit,  'motivo_alerta'] = 'Duplicatas excessivas'
    fraude_cnpj.loc[~mask_dup & mask_emit,  'motivo_alerta'] = 'Muitos emitentes'
    fraude_cnpj.loc[mask_dup & mask_emit,   'motivo_alerta'] = 'Duplicatas + Muitos emitentes'

    # Estatísticas gerais para KPIs
    stats = {
        'total_boletos':          len(df),
        'total_dup_id':           int(df['flag_dup_id'].sum()),
        'total_dup_conteudo':     int(df['flag_dup_conteudo'].sum()),
        'total_duplicatas':       int(df['flag_duplicata'].sum()),
        'pct_duplicatas':         round(df['flag_duplicata'].mean() * 100, 2),
        'cnpjs_suspeitos':        int(fraude_cnpj['flag_risco_fraude'].sum()),
        'total_cnpjs_com_boleto': len(fraude_cnpj),
        'beneficiarios_envolvidos': int(
            df[df['flag_duplicata'] == 1]['id_beneficiario'].nunique()
        ),
        'pct_dup_thresh':   pct_dup_thresh,
        'n_emit_thresh':    n_emitentes_thresh,
    }

    # Limpar coluna auxiliar
    df = df.drop(columns=['_dt_venc_str'])

    return {
        'df_bol_marcado':   df,
        'resumo_duplicatas': resumo,
        'fraude_por_cnpj':  fraude_cnpj,
        'stats':            stats,
    }



# ─────────────────────────────────────────────────────────────────────────────
# PASSO 3 — PROCESSAR BOLETOS
# ─────────────────────────────────────────────────────────────────────────────

def processar_boletos(df_bol: pd.DataFrame) -> pd.DataFrame:
    """Calcula atraso real e cria flags por boleto."""
    df = df_bol.copy()
    for col in ['dt_emissao', 'dt_vencimento', 'dt_pagamento']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['atraso_dias_real']        = (df['dt_pagamento'] - df['dt_vencimento']).dt.days
    df['flag_atrasado']           = (df['atraso_dias_real'] > 0).astype(int)
    df['flag_protestado']         = df['tipo_baixa'].str.contains('protesto', case=False, na=False).astype(int)
    df['flag_sem_pgto']           = df['dt_pagamento'].isna().astype(int)
    df['flag_cedente_solicitou']  = df['tipo_baixa'].str.contains('cedente', case=False, na=False).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def agregar_boletos(df_bol: pd.DataFrame) -> pd.DataFrame:
    """Agrega boletos por CNPJ gerando features de comportamento de pagamento."""
    feat = df_bol.groupby('id_pagador').agg(
        bol_qtd_total         = ('id_boleto',        'count'),
        bol_vlr_nominal_total = ('vlr_nominal',      'sum'),
        bol_vlr_baixa_total   = ('vlr_baixa',        'sum'),
        bol_atraso_medio      = ('atraso_dias_real',  'mean'),
        bol_atraso_max        = ('atraso_dias_real',  'max'),
        bol_pct_atrasado      = ('flag_atrasado',     'mean'),
        bol_pct_sem_pgto      = ('flag_sem_pgto',     'mean'),
        bol_pct_protestado    = ('flag_protestado',   'mean'),
        bol_pct_ced_solicitou = ('flag_cedente_solicitou', 'mean'),
        bol_n_especies        = ('tipo_especie',      'nunique'),
    ).reset_index()
    feat['bol_taxa_recuperacao'] = (
        feat['bol_vlr_baixa_total'] / feat['bol_vlr_nominal_total']
    ).clip(0, 1)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 5 — TARGET
# ─────────────────────────────────────────────────────────────────────────────

def definir_target(df_full: pd.DataFrame, liq_thresh: float, mat_thresh: float):
    """
    Target=1 (carteira BOA) quando satisfaz as 3 condições:
      score_materialidade_v2 >= mat_thresh
      sacado_indice_liquidez_1m >= liq_thresh
      media_atraso_dias <= P75 da base
    """
    p75 = df_full['media_atraso_dias'].quantile(0.75)
    df  = df_full.copy()
    df['target'] = (
        (df['score_materialidade_v2']    >= mat_thresh) &
        (df['sacado_indice_liquidez_1m'] >= liq_thresh) &
        (df['media_atraso_dias']         <= p75)
    ).astype(int)
    return df, p75


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 6 — CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_correlacao(df_full: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação de Pearson entre features e target."""
    cols = [f for f in FEATURES + ['target'] if f in df_full.columns]
    return df_full[cols].corr()


# ─────────────────────────────────────────────────────────────────────────────
# PASSOS 7–8 — MODELAGEM ML
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline(model) -> Pipeline:
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('mdl', model),
    ])


def treinar_modelos(df_full: pd.DataFrame, avail: list,
                    test_size: float, n_estimators: int):
    """
    Treina Logistic Regression, Random Forest e GradientBoosting/XGBoost
    com cross-validation estratificada (k=5).
    Retorna (resultados, X_test, y_test).
    """
    df_model = df_full[avail + ['target']].dropna(subset=['target'])
    X = df_model[avail]
    y = df_model['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    boost = (
        XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      use_label_encoder=False, eval_metric='logloss',
                      random_state=42, n_jobs=-1)
        if XGBOOST_AVAILABLE else
        GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42)
    )
    boost_name = 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'

    modelos = {
        'Logistic Regression': _make_pipeline(
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        ),
        'Random Forest': _make_pipeline(
            RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
        ),
        boost_name: _make_pipeline(boost),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    for nome, pipe in modelos.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring='roc_auc', n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        resultados[nome] = dict(
            pipe      = pipe,
            cv_mean   = cv_scores.mean(),
            cv_std    = cv_scores.std(),
            auc       = roc_auc_score(y_test, y_proba),
            ap        = average_precision_score(y_test, y_proba),
            y_pred    = y_pred,
            y_proba   = y_proba,
            fpr       = fpr.tolist(),
            tpr       = tpr.tolist(),
            cm        = confusion_matrix(y_test, y_pred).tolist(),
            cv_scores = cv_scores.tolist(),
        )

    return resultados, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 10 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_feature_importance(ml_results: dict, avail: list,
                                 best_name: str) -> pd.Series:
    """Extrai importância de features do melhor modelo."""
    sm = ml_results[best_name]['pipe'].named_steps['mdl']
    if hasattr(sm, 'feature_importances_'):
        imp = sm.feature_importances_
    elif hasattr(sm, 'coef_'):
        imp = np.abs(sm.coef_[0])
    else:
        imp = np.ones(len(avail)) / len(avail)
    return pd.Series(imp, index=avail).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 11 — SCORE FINAL (fórmula composta ponderada)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_score_final(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Score FIDC (0–1000) por fórmula composta ponderada:
      35% Liquidez Sacado 1m       → capacidade de honrar boletos
      25% Score Materialidade v2   → histórico consolidado Núclea
      15% Score Quantidade v2      → consistência de volume
      10% Liquidez Quantitativa 3m → estabilidade temporal
       8% Atraso invertido         → penaliza atrasos
       7% Inadimplência invertida  → detecta deterioração recente
    """
    df = df_full.copy()

    atraso_max   = df['media_atraso_dias'].quantile(0.99).clip(1)
    atraso_norm  = (df['media_atraso_dias'].fillna(atraso_max) / atraso_max).clip(0, 1)
    inad_norm    = df['share_vl_inad_pag_bol_6_a_15d'].fillna(0).clip(0, 1)

    c_liquidez   = df['sacado_indice_liquidez_1m'].fillna(0).clip(0, 1)
    c_mat        = (df['score_materialidade_v2'].fillna(0) / 1000).clip(0, 1)
    c_qtd        = (df['score_quantidade_v2'].fillna(0) / 1000).clip(0, 1)
    c_liq3m      = df['indicador_liquidez_quantitativo_3m'].fillna(0).clip(0, 1)
    c_atraso_inv = (1 - atraso_norm)
    c_inad_inv   = (1 - inad_norm)

    score_raw = (
        c_liquidez   * 0.35 +
        c_mat        * 0.25 +
        c_qtd        * 0.15 +
        c_liq3m      * 0.10 +
        c_atraso_inv * 0.08 +
        c_inad_inv   * 0.07
    )

    df['score_fidc'] = (score_raw * 1000).round(0).astype(int).clip(0, 1000)
    df['rating_carteira'] = pd.cut(
        df['score_fidc'],
        bins=RATING_BINS,
        labels=RATING_LABELS,
        right=False,
    ).astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df_aux: pd.DataFrame, df_bol: pd.DataFrame,
                 test_size: float = 0.2, n_estimators: int = 300,
                 liq_thresh: float = 0.70, mat_thresh: float = 900,
                 pct_dup_thresh: float = FRAUDE_PCT_DUP_THRESH,
                 n_emitentes_thresh: int = FRAUDE_N_EMITENTES_THRESH) -> dict:
    """
    Executa o pipeline completo (passos 2B–11) e retorna dict com
    todos os resultados para o dashboard.
    """
    result = {}

    # ── Passo 2B: detecção de fraude (antes de qualquer transformação) ────
    fraude = detectar_duplicatas(df_bol, pct_dup_thresh, n_emitentes_thresh)
    result['fraude']          = fraude
    df_bol_marcado            = fraude['df_bol_marcado']

    # ── Passo 3: processar boletos (usa df já marcado com flags de fraude) ─
    df_bol          = processar_boletos(df_bol_marcado)
    result['df_bol'] = df_bol

    # ── Passo 4: feature engineering ─────────────────────────────────────
    feat_bol = agregar_boletos(df_bol)

    # Mesclar features de fraude por CNPJ no feat_bol
    fraude_cnpj = fraude['fraude_por_cnpj'][
        ['id_pagador', 'bol_pct_duplicado', 'bol_n_emitentes',
         'bol_qtd_dup_total', 'flag_risco_fraude', 'motivo_alerta']
    ].rename(columns={'id_pagador': 'id_cnpj'})

    feat_bol = feat_bol.rename(columns={'id_pagador': 'id_cnpj'})
    feat_bol = feat_bol.merge(fraude_cnpj, on='id_cnpj', how='left')
    feat_bol['flag_risco_fraude'] = feat_bol['flag_risco_fraude'].fillna(0).astype(int)
    feat_bol['bol_pct_duplicado'] = feat_bol['bol_pct_duplicado'].fillna(0)
    feat_bol['bol_n_emitentes']   = feat_bol['bol_n_emitentes'].fillna(0)

    df_full  = df_aux.merge(feat_bol, on='id_cnpj', how='left')

    df_full, p75       = definir_target(df_full, liq_thresh, mat_thresh)
    result['p75_atraso'] = p75
    result['corr_matrix'] = calcular_correlacao(df_full)

    avail = [f for f in FEATURES if f in df_full.columns]
    ml_results, X_test, y_test = treinar_modelos(df_full, avail, test_size, n_estimators)

    result['ml']          = ml_results
    result['X_test']      = X_test
    result['y_test']      = y_test
    result['avail_feats'] = avail

    best_name           = max(ml_results, key=lambda n: ml_results[n]['auc'])
    result['best_name'] = best_name
    result['feat_imp']  = calcular_feature_importance(ml_results, avail, best_name)

    df_full           = calcular_score_final(df_full)
    result['df_full'] = df_full

    # Perfil setorial por CNAE (roda após score para incluir rating no agregado)
    perfil_cnae = calcular_perfil_cnae(df_full)
    result['perfil_cnae'] = perfil_cnae

    return result
