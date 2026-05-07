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
from sklearn.calibration import CalibratedClassifierCV
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

# Features do modelo — apenas perspectiva do PAGADOR, sem leakage com o target.
# Target = adimplência real dos boletos (vlr_baixa > 0, excluindo cancelamentos).
# Features seguras: indicadores históricos gerais do sacado na PCR,
# calculados de forma independente dos boletos da carteira atual.
#
# Removidas por leakage ou perspectiva errada:
#   bol_* — calculadas sobre os mesmos boletos que definem o target
#   cedente_indice_liquidez_1m — mede comportamento como CEDENTE, não pagador
FEATURES = [
    'sacado_indice_liquidez_1m',           # % boletos pagos como pagador (1m)
    'score_materialidade_evolucao',         # tendência do score de risco de pagamento
    'score_quantidade_v2',                  # score risco por quantidade de boletos
    'score_materialidade_v2',               # score risco por valor de boletos
    'indicador_liquidez_quantitativo_3m',   # liquidez 3m — janela mais estável
    'share_vl_inad_pag_bol_6_a_15d',       # % atraso leve (6-15 dias) como pagador
    'media_atraso_dias',                    # média de dias de atraso como pagador
]

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
FRAUDE_PCT_DUP_THRESH    = 0.05   # % mínimo de boletos duplicados para levantar alerta
FRAUDE_N_EMITENTES_THRESH = 10   # nº de beneficiários distintos por pagador para alertar

# Tipos de baixa que representam cancelamento comercial —
# NÃO são inadimplência real do sacado
BAIXAS_CANCELAMENTO = {
    '5 - Baixa integral por solicitacao do cedente',
    '8 - Baixa integral por solicitacao da instituicao destinataria',
}

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
    df = df_full.copy()
    df['cd_cnae_fmt'] = df['cd_cnae_prin'].apply(_formatar_cnae)
    if 'flag_risco_fraude' not in df.columns:
        df['flag_risco_fraude'] = 0

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

    perfil = perfil.merge(CNAE_DENOMINACOES, on='cd_cnae_fmt', how='left')
    perfil['denominacao'] = perfil['denominacao'].fillna('Não identificado')
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
    df = df_bol.copy()

    contagem_id = df.groupby('id_boleto')['id_boleto'].transform('count')
    df['flag_dup_id'] = (contagem_id > 1).astype(int)

    df['_dt_venc_str'] = pd.to_datetime(df['dt_vencimento'], errors='coerce').dt.strftime('%Y-%m-%d')
    chave_cols_str = ['vlr_nominal', '_dt_venc_str', 'id_pagador', 'id_beneficiario']
    contagem_conteudo = df.groupby(chave_cols_str)['id_boleto'].transform('count')
    df['flag_dup_conteudo'] = (contagem_conteudo > 1).astype(int)
    df['flag_duplicata'] = ((df['flag_dup_id'] == 1) | (df['flag_dup_conteudo'] == 1)).astype(int)

    _dup_id_grp = (df[df['flag_dup_id'] == 1]
                   .groupby('id_boleto', as_index=False)
                   .agg(qtd_ocorrencias=('id_boleto', 'count'),
                        id_pagador=('id_pagador', 'first'),
                        vlr_nominal=('vlr_nominal', 'first'))
                   .assign(tipo_duplicata='ID repetido'))
    dup_id = _dup_id_grp[['id_pagador', 'vlr_nominal',
                           'qtd_ocorrencias', 'tipo_duplicata']].copy()

    _dup_cont_grp = (df[df['flag_dup_conteudo'] == 1]
                     .groupby(chave_cols_str, as_index=False)
                     .agg(qtd_ocorrencias=('id_boleto', 'count'))
                     .rename(columns={'_dt_venc_str': 'dt_vencimento'}))
    dup_cont = _dup_cont_grp[['id_pagador', 'vlr_nominal', 'qtd_ocorrencias']].copy()
    dup_cont['tipo_duplicata'] = 'Conteúdo idêntico'

    resumo = pd.concat([
        dup_id[['id_pagador', 'vlr_nominal', 'qtd_ocorrencias', 'tipo_duplicata']],
        dup_cont[['id_pagador', 'vlr_nominal', 'qtd_ocorrencias', 'tipo_duplicata']],
    ], ignore_index=True).sort_values('qtd_ocorrencias', ascending=False)

    fraude_cnpj = df.groupby('id_pagador').agg(
        bol_qtd_total        = ('id_boleto',         'count'),
        bol_qtd_dup_id       = ('flag_dup_id',       'sum'),
        bol_qtd_dup_conteudo = ('flag_dup_conteudo', 'sum'),
        bol_qtd_dup_total    = ('flag_duplicata',    'sum'),
        bol_n_emitentes      = ('id_beneficiario',   'nunique'),
    ).reset_index()

    fraude_cnpj['bol_pct_duplicado'] = (
        fraude_cnpj['bol_qtd_dup_total'] / fraude_cnpj['bol_qtd_total']
    ).fillna(0)

    fraude_cnpj['flag_risco_fraude'] = (
        (fraude_cnpj['bol_pct_duplicado'] >= pct_dup_thresh) |
        (fraude_cnpj['bol_n_emitentes']   >= n_emitentes_thresh)
    ).astype(int)

    fraude_cnpj['motivo_alerta'] = ''
    mask_dup  = fraude_cnpj['bol_pct_duplicado'] >= pct_dup_thresh
    mask_emit = fraude_cnpj['bol_n_emitentes']   >= n_emitentes_thresh
    fraude_cnpj.loc[mask_dup & ~mask_emit,  'motivo_alerta'] = 'Duplicatas excessivas'
    fraude_cnpj.loc[~mask_dup & mask_emit,  'motivo_alerta'] = 'Muitos emitentes'
    fraude_cnpj.loc[mask_dup & mask_emit,   'motivo_alerta'] = 'Duplicatas + Muitos emitentes'

    stats = {
        'total_boletos':           len(df),
        'total_dup_id':            int(df['flag_dup_id'].sum()),
        'total_dup_conteudo':      int(df['flag_dup_conteudo'].sum()),
        'total_duplicatas':        int(df['flag_duplicata'].sum()),
        'pct_duplicatas':          round(df['flag_duplicata'].mean() * 100, 2),
        'cnpjs_suspeitos':         int(fraude_cnpj['flag_risco_fraude'].sum()),
        'total_cnpjs_com_boleto':  len(fraude_cnpj),
        'beneficiarios_envolvidos': int(
            df[df['flag_duplicata'] == 1]['id_beneficiario'].nunique()
        ),
        'pct_dup_thresh':  pct_dup_thresh,
        'n_emit_thresh':   n_emitentes_thresh,
    }

    df = df.drop(columns=['_dt_venc_str'])
    return {
        'df_bol_marcado':    df,
        'resumo_duplicatas': resumo,
        'fraude_por_cnpj':   fraude_cnpj,
        'stats':             stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 3 — PROCESSAR BOLETOS
# ─────────────────────────────────────────────────────────────────────────────

def processar_boletos(df_bol: pd.DataFrame) -> pd.DataFrame:
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
    feat = df_bol.groupby('id_pagador').agg(
        bol_qtd_total         = ('id_boleto',         'count'),
        bol_vlr_nominal_total = ('vlr_nominal',       'sum'),
        bol_vlr_baixa_total   = ('vlr_baixa',         'sum'),
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

def definir_target(df_full: pd.DataFrame,
                   df_bol_marcado: pd.DataFrame = None,
                   liq_thresh: float = 0.65,
                   mat_thresh: float = 800):
    """
    Target baseado em histórico REAL de pagamento dos boletos (sem leakage).

    Target = 1 (ADIMPLENTE) se o sacado não tem nenhum boleto inadimplente real.
    Inadimplente real = vlr_baixa nulo ou zero, excluindo cancelamentos comerciais:
      - '5 - Baixa integral por solicitacao do cedente'
      - '8 - Baixa integral por solicitacao da instituicao destinataria'

    Fallback: se df_bol_marcado não disponível, usa regra anterior (2 de 3).
    """
    df = df_full.copy()

    if df_bol_marcado is not None and 'vlr_baixa' in df_bol_marcado.columns:
        bol = df_bol_marcado.copy()

        bol['inadimplente_real'] = (
            (bol['vlr_baixa'].isna() | (bol['vlr_baixa'] == 0)) &
            (~bol['tipo_baixa'].isin(BAIXAS_CANCELAMENTO))
        ).astype(int)

        inad_por_pag = (bol.groupby('id_pagador')['inadimplente_real']
                           .sum()
                           .reset_index()
                           .rename(columns={'id_pagador': 'id_cnpj',
                                            'inadimplente_real': '_n_inad'}))

        df = df.merge(inad_por_pag, on='id_cnpj', how='left')
        df['_n_inad'] = df['_n_inad'].fillna(0)
        df['target']  = (df['_n_inad'] == 0).astype(int)
        df.drop(columns=['_n_inad'], inplace=True)

        pct = df['target'].mean() * 100
        print(f"[SafeAsset] Target (adimplência real) — "
              f"Adimplentes: {df['target'].sum():,} ({pct:.1f}%)  "
              f"Inadimplentes: {(df['target']==0).sum():,} ({100-pct:.1f}%)")

    else:
        raise ValueError(
            "Base de boletos não disponível para calcular o target. "
            "Faça o upload do arquivo base_boletos_fiap.csv."
        )

    p75 = df['media_atraso_dias'].quantile(0.75) if 'media_atraso_dias' in df.columns else 0
    return df, p75


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 6 — CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_correlacao(df_full: pd.DataFrame) -> pd.DataFrame:
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
    df_model = df_full[avail + ['target']].dropna(subset=['target'])
    X = df_model[avail]
    y = df_model['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"[SafeAsset] Treino — target=1: {n_pos} ({n_pos/(n_pos+n_neg)*100:.1f}%)  "
          f"target=0: {n_neg}  scale_pos_weight: {scale_pos:.1f}")

    boost = (
        XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      use_label_encoder=False, eval_metric='logloss',
                      scale_pos_weight=scale_pos,
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

    best = max(resultados, key=lambda n: resultados[n]['auc'])
    try:
        calibrated = CalibratedClassifierCV(
            resultados[best]['pipe'], cv='prefit', method='sigmoid'
        )
        calibrated.fit(X_test, y_test)
        resultados[best]['pipe_calibrated'] = calibrated
        print(f"[SafeAsset] Calibração OK — {best}")
    except Exception as e:
        print(f"[SafeAsset] Calibração falhou ({e}) — usando pipe original")
        resultados[best]['pipe_calibrated'] = resultados[best]['pipe']

    return resultados, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 10 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_feature_importance(ml_results: dict, avail: list,
                                 best_name: str) -> pd.Series:
    sm = ml_results[best_name]['pipe'].named_steps['mdl']
    if hasattr(sm, 'feature_importances_'):
        imp = sm.feature_importances_
    elif hasattr(sm, 'coef_'):
        imp = np.abs(sm.coef_[0])
    else:
        imp = np.ones(len(avail)) / len(avail)
    return pd.Series(imp, index=avail).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# PRODUTO 2 — Probabilidade ML (score auxiliar contínuo)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_prob_ml(df_full: pd.DataFrame,
                     ml_results: dict,
                     avail: list,
                     best_name: str) -> pd.DataFrame:
    df   = df_full.copy()
    pipe = ml_results[best_name].get('pipe_calibrated', ml_results[best_name]['pipe'])

    X_todos    = df[avail].copy()
    idx_validos = X_todos.dropna().index

    prob = np.zeros(len(df))
    if len(idx_validos) > 0:
        probs = pipe.predict_proba(X_todos.loc[idx_validos])
        prob[df.index.get_indexer(idx_validos)] = probs[:, 1]

    df['prob_ml_bom'] = (prob * 100).round(1)

    df['alerta_divergencia'] = (
        ((df['score_fidc'] >= 800) & (df['prob_ml_bom'] <  30)) |
        ((df['score_fidc'] <  300) & (df['prob_ml_bom'] >= 70))
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASSO 11 — SCORE FINAL (fórmula composta ponderada)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_score_final(df_full: pd.DataFrame) -> pd.DataFrame:
    df = df_full.copy()

    atraso_max  = df['media_atraso_dias'].quantile(0.99).clip(1)
    atraso_norm = (df['media_atraso_dias'].fillna(atraso_max) / atraso_max).clip(0, 1)
    inad_norm   = df['share_vl_inad_pag_bol_6_a_15d'].fillna(0).clip(0, 1)

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
# COBERTURA DA CARTEIRA NOVA
# ─────────────────────────────────────────────────────────────────────────────

def calcular_cobertura_carteira(df_full: pd.DataFrame,
                                 df_cart: pd.DataFrame) -> dict:
    """
    Analisa a cobertura da carteira nova — quantos CNPJs têm histórico PCR
    e quantos são primeiro contato.

    Retorna dict com:
      total_cnpjs       : total de CNPJs na carteira
      com_historico     : CNPJs com histórico PCR
      sem_historico     : CNPJs sem histórico (primeiro contato)
      pct_com_historico : % com histórico
      pct_sem_historico : % sem histórico
      recomendados      : CNPJs com histórico e rating A ou B
      atencao           : CNPJs com histórico e rating C
      nao_recomendados  : CNPJs com histórico e rating D ou E
      vlr_total         : valor total da carteira
      vlr_sem_historico : valor em risco nos CNPJs sem histórico
    """
    col_pag = 'id_pagador' if 'id_pagador' in df_cart.columns else 'id_cnpj'
    cnpjs_cart = set(df_cart[col_pag].astype(str).unique())
    total = len(cnpjs_cart)

    # Cruzar com df_full que tem histórico
    tem_hist = set(df_full[df_full['sem_historico'] == 0]['id_cnpj'].astype(str))
    sem_hist  = set(df_full[df_full['sem_historico'] == 1]['id_cnpj'].astype(str))

    n_com  = len(cnpjs_cart & tem_hist)
    n_sem  = len(cnpjs_cart & sem_hist)
    # CNPJs na carteira mas não na base auxiliar
    n_novo = total - n_com - n_sem
    n_sem  = n_sem + n_novo  # tratar não encontrados como sem histórico

    # Ratings dos que têm histórico
    df_hist = df_full[
        (df_full['id_cnpj'].astype(str).isin(cnpjs_cart)) &
        (df_full['sem_historico'] == 0)
    ]
    n_rec  = int(df_hist['rating_carteira'].isin(['A — Excelente','B — Bom']).sum())
    n_atc  = int((df_hist['rating_carteira'] == 'C — Risco Moderado').sum())
    n_nrec = int(df_hist['rating_carteira'].isin(['D — Risco Elevado','E — Alto Risco']).sum())

    # Valores
    vlr_total = float(df_cart['vlr_nominal'].sum()) if 'vlr_nominal' in df_cart.columns else 0
    sem_hist_ids = cnpjs_cart - tem_hist
    vlr_sem = float(df_cart[df_cart[col_pag].astype(str).isin(sem_hist_ids)]['vlr_nominal'].sum())               if 'vlr_nominal' in df_cart.columns else 0

    print(f"[SafeAsset] Cobertura — {n_com:,} com histórico ({n_com/total*100:.1f}%) · "
          f"{n_sem:,} sem histórico ({n_sem/total*100:.1f}%)")

    return {
        'total_cnpjs':        total,
        'com_historico':      n_com,
        'sem_historico':      n_sem,
        'pct_com_historico':  round(n_com/total*100, 1) if total else 0,
        'pct_sem_historico':  round(n_sem/total*100, 1) if total else 0,
        'recomendados':       n_rec,
        'atencao':            n_atc,
        'nao_recomendados':   n_nrec,
        'vlr_total':          round(vlr_total, 2),
        'vlr_sem_historico':  round(vlr_sem, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df_aux: pd.DataFrame, df_bol: pd.DataFrame,
                 df_carteira: pd.DataFrame = None,
                 test_size: float = 0.2, n_estimators: int = 300,
                 liq_thresh: float = 0.65, mat_thresh: float = 800,
                 pct_dup_thresh: float = FRAUDE_PCT_DUP_THRESH,
                 n_emitentes_thresh: int = FRAUDE_N_EMITENTES_THRESH) -> dict:
    # df_carteira: boletos da nova carteira a ser adquirida (sem vlr_baixa)
    # Se None, usa df_bol como proxy para demonstração
    result = {}

    # Passo 2B — detecção de fraude
    fraude         = detectar_duplicatas(df_bol, pct_dup_thresh, n_emitentes_thresh)
    result['fraude'] = fraude
    df_bol_marcado   = fraude['df_bol_marcado']

    # Passo 3 — processar boletos
    df_bol          = processar_boletos(df_bol_marcado)
    result['df_bol'] = df_bol

    # Passo 4 — feature engineering
    feat_bol = agregar_boletos(df_bol)

    fraude_cnpj = fraude['fraude_por_cnpj'][
        ['id_pagador', 'bol_pct_duplicado', 'bol_n_emitentes',
         'bol_qtd_dup_total', 'flag_risco_fraude', 'motivo_alerta']
    ].rename(columns={'id_pagador': 'id_cnpj'})

    feat_bol = feat_bol.rename(columns={'id_pagador': 'id_cnpj'})
    feat_bol = feat_bol.merge(fraude_cnpj, on='id_cnpj', how='left')
    feat_bol['flag_risco_fraude'] = feat_bol['flag_risco_fraude'].fillna(0).astype(int)
    feat_bol['bol_pct_duplicado'] = feat_bol['bol_pct_duplicado'].fillna(0)
    feat_bol['bol_n_emitentes']   = feat_bol['bol_n_emitentes'].fillna(0)

    df_full = df_aux.merge(feat_bol, on='id_cnpj', how='left')

    # Flag de sacados sem histórico de boletos na PCR
    bol_features = ['bol_qtd_total', 'bol_pct_atrasado', 'bol_taxa_recuperacao']
    bol_disp = [f for f in bol_features if f in df_full.columns]
    if bol_disp:
        df_full['sem_historico'] = df_full[bol_disp[0]].isna().astype(int)
    else:
        df_full['sem_historico'] = 0
    n_sem = int(df_full['sem_historico'].sum())
    pct_sem = df_full['sem_historico'].mean() * 100
    print(f"[SafeAsset] Histórico — com histórico: {len(df_full)-n_sem:,} "
          f"({100-pct_sem:.1f}%)  sem histórico: {n_sem:,} ({pct_sem:.1f}%)")

    # Passo 5 — target baseado em adimplência real dos boletos
    df_full, p75       = definir_target(df_full, df_bol_marcado, liq_thresh, mat_thresh)
    result['p75_atraso'] = p75
    result['corr_matrix'] = calcular_correlacao(df_full)

    # Passos 7-8 — modelagem ML
    avail = [f for f in FEATURES if f in df_full.columns]
    ml_results, X_test, y_test = treinar_modelos(df_full, avail, test_size, n_estimators)

    result['ml']          = ml_results
    result['X_test']      = X_test
    result['y_test']      = y_test
    result['avail_feats'] = avail

    best_name           = max(ml_results, key=lambda n: ml_results[n]['auc'])
    result['best_name'] = best_name
    result['feat_imp']  = calcular_feature_importance(ml_results, avail, best_name)

    # Passo 11 — score final
    df_full = calcular_score_final(df_full)

    # Produto 2 — score ML auxiliar
    df_full = calcular_prob_ml(df_full, ml_results, avail, best_name)
    result['df_full'] = df_full

    # Perfil setorial por CNAE
    perfil_cnae = calcular_perfil_cnae(df_full)
    result['perfil_cnae'] = perfil_cnae

    # ── Cobertura da carteira nova ────────────────────────────────────────
    df_cart = df_carteira if df_carteira is not None else df_bol
    cobertura = calcular_cobertura_carteira(df_full, df_cart)
    result['cobertura'] = cobertura

    return result