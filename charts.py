# =============================================================================
# charts.py — SafeAsset
# Responsável por: criar todas as figuras Plotly para o dashboard
# Sem dependências de Dash — apenas plotly, pandas, numpy
# =============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import RATING_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# TEMA GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

NAVY   = '#0a1628'
BLUE   = '#1e3a5f'
ACCENT = '#00d4ff'
ACCENT2= '#00ff88'
WARN   = '#ff6b35'
MUTED  = '#8892a4'
BORDER = '#1e3a5f'
WHITE  = '#e8f0fe'
CARD_BG = '#111d2e'
AMBER  = '#F59E0B'

from pipeline import RATING_COLOR

# Template base para todos os gráficos (fundo transparente, fonte clara)
BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=WHITE, family='Space Grotesk'),
)

def _fig(fig, height=360, margin=None, **kwargs):
    """Aplica tema e dimensões padrão a qualquer figura."""
    m = margin or dict(l=70, r=30, t=30, b=50)
    fig.update_layout(**BASE_LAYOUT, height=height, margin=m, **kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ABA EDA — BASE AUXILIAR
# ─────────────────────────────────────────────────────────────────────────────

def fig_nulos(df_aux: pd.DataFrame) -> go.Figure:
    """Barras horizontais com % de valores nulos por coluna."""
    pct = (df_aux.isnull().sum() / len(df_aux) * 100).round(1)
    fig = px.bar(x=pct.values, y=pct.index, orientation='h',
                 labels={'x': '% Nulos', 'y': ''},
                 color_discrete_sequence=[WARN])
    return _fig(fig, height=340, margin=dict(l=230, r=30, t=20, b=40))


def fig_scores(df_aux: pd.DataFrame) -> go.Figure:
    """Histogramas sobrepostos dos 3 scores de crédito."""
    cols   = ['score_materialidade_evolucao', 'score_quantidade_v2', 'score_materialidade_v2']
    labels = ['Mat. Evolução', 'Quantidade v2', 'Materialidade v2']
    colors = [ACCENT, ACCENT2, WARN]
    fig = go.Figure([
        go.Histogram(x=df_aux[c].dropna(), name=lbl, opacity=.75,
                     nbinsx=40, marker_color=col)
        for c, lbl, col in zip(cols, labels, colors)
    ])
    return _fig(fig, height=360, barmode='overlay',
                xaxis_title='Score (0–1000)', yaxis_title='Frequência',
                legend=dict(orientation='h', y=1.05, font=dict(size=12)))


def fig_liquidez(df_aux: pd.DataFrame) -> go.Figure:
    """Histogramas sobrepostos dos 3 índices de liquidez."""
    cols   = ['sacado_indice_liquidez_1m', 'cedente_indice_liquidez_1m',
              'indicador_liquidez_quantitativo_3m']
    labels = ['Sacado 1m', 'Cedente 1m', 'Quantitativo 3m']
    colors = [ACCENT, ACCENT2, WARN]
    fig = go.Figure([
        go.Histogram(x=df_aux[c].dropna(), name=lbl, opacity=.75,
                     nbinsx=35, marker_color=col)
        for c, lbl, col in zip(cols, labels, colors)
    ])
    return _fig(fig, height=360, barmode='overlay',
                xaxis_title='Índice (0–1)', yaxis_title='Frequência',
                legend=dict(orientation='h', y=1.05, font=dict(size=12)))


def fig_atraso_uf(df_aux: pd.DataFrame) -> go.Figure:
    """Box plot de atraso médio pelas Top 12 UFs."""
    top_ufs = df_aux['uf'].value_counts().head(12).index
    df_top  = df_aux[df_aux['uf'].isin(top_ufs)]
    fig = px.box(df_top, x='uf', y='media_atraso_dias',
                 color_discrete_sequence=[ACCENT],
                 labels={'media_atraso_dias': 'Dias de Atraso', 'uf': 'UF'})
    return _fig(fig, height=380, margin=dict(l=70, r=30, t=20, b=50))


def fig_volume_mensal(df_bol: pd.DataFrame) -> go.Figure:
    """Barras de volume de boletos por mês de emissão."""
    vol = (df_bol.assign(mes=df_bol['dt_emissao'].dt.to_period('M').astype(str))
                 .groupby('mes').size().reset_index(name='qtd').sort_values('mes'))
    fig = px.bar(vol, x='mes', y='qtd',
                 color_discrete_sequence=[ACCENT2],
                 labels={'mes': 'Mês de Emissão', 'qtd': 'Quantidade de Boletos'})
    return _fig(fig, height=380, margin=dict(l=70, r=30, t=20, b=70),
                xaxis_tickangle=-45)


def fig_tipos_baixa(df_bol: pd.DataFrame) -> go.Figure:
    """Donut chart com distribuição de tipos de baixa."""
    tb = df_bol['tipo_baixa'].value_counts().reset_index()
    tb.columns = ['tipo', 'qtd']
    fig = px.pie(tb, values='qtd', names='tipo', hole=0.42,
                 color_discrete_sequence=[ACCENT, ACCENT2, WARN, '#a78bfa', '#f472b6'])
    return _fig(fig, height=400, margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(size=11), orientation='v', x=0.75))


def fig_atraso_real(df_bol: pd.DataFrame) -> go.Figure:
    """Histograma de atraso real dos boletos pagos com atraso."""
    atrasados = df_bol[df_bol['atraso_dias_real'] > 0]
    fig = px.histogram(atrasados, x='atraso_dias_real', nbins=40,
                       color_discrete_sequence=[WARN],
                       labels={'atraso_dias_real': 'Dias de Atraso'})
    return _fig(fig, height=360, bargap=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# ABA TARGET & CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def fig_target_pizza(df_full: pd.DataFrame) -> go.Figure:
    """Donut com proporção Boa vs Ruim."""
    dist = df_full['target'].map({0: 'Ruim (0)', 1: 'Boa (1)'}).value_counts().reset_index()
    dist.columns = ['target', 'count']
    fig = px.pie(dist, values='count', names='target', hole=0.48,
                 color_discrete_map={'Boa (1)': ACCENT2, 'Ruim (0)': WARN})
    return _fig(fig, height=380, margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(font=dict(size=14)))


def fig_boxplot_target(df_full: pd.DataFrame, col: str, label: str) -> go.Figure:
    """Box plot de uma variável por target."""
    x_vals = df_full['target'].map({0: 'Ruim (0)', 1: 'Boa (1)'})
    fig = px.box(df_full, x=x_vals, y=col,
                 color=x_vals,
                 color_discrete_map={'Boa (1)': ACCENT2, 'Ruim (0)': WARN},
                 labels={'x': 'Classificação', 'y': label})
    return _fig(fig, height=380, showlegend=False,
                margin=dict(l=80, r=30, t=20, b=60))


def fig_correlacao_target(corr_mat: pd.DataFrame) -> go.Figure:
    """Barras horizontais de correlação de cada feature com o target."""
    corr = corr_mat['target'].drop('target').sort_values(key=abs, ascending=False)
    colors = [ACCENT2 if v >= 0 else WARN for v in corr.values]
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation='h',
        marker_color=colors, opacity=.85,
    ))
    return _fig(fig, height=400, margin=dict(l=240, r=40, t=20, b=50),
                xaxis_title='Coeficiente de Pearson',
                yaxis=dict(tickfont=dict(size=12)))


def fig_heatmap_correlacao(corr_mat: pd.DataFrame) -> go.Figure:
    """Heatmap completo da matriz de correlação."""
    fig = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale=[[0, '#ff2d55'], [0.5, '#1e3a5f'], [1, '#00ff88']],
        zmid=0, zmin=-1, zmax=1,
        text=corr_mat.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=9),
        showscale=True,
    ))
    return _fig(fig, height=560, margin=dict(l=170, r=80, t=30, b=170),
                xaxis=dict(tickfont=dict(size=9), tickangle=-45),
                yaxis=dict(tickfont=dict(size=9)))


# ─────────────────────────────────────────────────────────────────────────────
# ABA MODELAGEM
# ─────────────────────────────────────────────────────────────────────────────

def fig_cv_folds(ml_results: dict) -> go.Figure:
    """Barras agrupadas de AUC por fold para cada modelo."""
    fig = go.Figure([
        go.Bar(name=nome, x=[f'Fold {i+1}' for i in range(5)],
               y=res['cv_scores'], opacity=.85)
        for nome, res in ml_results.items()
    ])
    return _fig(fig, height=400, barmode='group',
                margin=dict(l=70, r=30, t=20, b=60),
                yaxis=dict(range=[0.5, 1.05], title='AUC-ROC'),
                legend=dict(orientation='h', y=1.05, font=dict(size=12)))


def fig_cv_vs_test(ml_results: dict) -> go.Figure:
    """Barras lado a lado CV AUC vs Test AUC por modelo."""
    nomes = list(ml_results.keys())
    fig = go.Figure([
        go.Bar(name='CV AUC (treino)', x=nomes,
               y=[r['cv_mean'] for r in ml_results.values()],
               marker_color=ACCENT, opacity=.85,
               error_y=dict(type='data',
                            array=[r['cv_std'] for r in ml_results.values()],
                            visible=True)),
        go.Bar(name='Test AUC', x=nomes,
               y=[r['auc'] for r in ml_results.values()],
               marker_color=ACCENT2, opacity=.85),
    ])
    return _fig(fig, height=400, barmode='group',
                margin=dict(l=70, r=30, t=20, b=70),
                yaxis=dict(range=[0.5, 1.05], title='AUC-ROC'),
                xaxis=dict(tickfont=dict(size=13)),
                legend=dict(orientation='h', y=1.05, font=dict(size=12)))


def fig_feature_importance(feat_imp: pd.Series, best_name: str) -> go.Figure:
    """Barras horizontais de importância das features."""
    fig = go.Figure(go.Bar(
        x=feat_imp.values[::-1],
        y=feat_imp.index[::-1],
        orientation='h',
        marker=dict(
            color=feat_imp.values[::-1],
            colorscale=[[0, BLUE], [0.5, ACCENT], [1, ACCENT2]],
            showscale=False,
        ),
        text=[f'{v:.3f}' for v in feat_imp.values[::-1]],
        textposition='outside',
        textfont=dict(size=11, color=WHITE),
    ))
    return _fig(fig, height=480, margin=dict(l=260, r=90, t=20, b=50),
                xaxis_title='Importância Relativa',
                yaxis=dict(tickfont=dict(size=12)))


# ─────────────────────────────────────────────────────────────────────────────
# ABA ROC & CONFUSÃO
# ─────────────────────────────────────────────────────────────────────────────

def fig_roc(ml_results: dict) -> go.Figure:
    """Curvas ROC sobrepostas para todos os modelos."""
    cores = [ACCENT, ACCENT2, WARN]
    traces = [
        go.Scatter(
            x=res['fpr'], y=res['tpr'],
            name=f'{nome}  (AUC = {res["auc"]:.3f})',
            mode='lines', line=dict(width=2.5, color=cor)
        )
        for (nome, res), cor in zip(ml_results.items(), cores)
    ]
    traces.append(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color=MUTED, width=1), showlegend=False,
    ))
    fig = go.Figure(traces)
    return _fig(fig, height=500, margin=dict(l=70, r=40, t=30, b=70),
                xaxis_title='Taxa de Falso Positivo (FPR)',
                yaxis_title='Taxa de Verdadeiro Positivo (TPR)',
                legend=dict(x=0.38, y=0.08, font=dict(size=13),
                            bgcolor='rgba(0,0,0,0.4)'))


def fig_confusion_matrix(ml_results: dict) -> go.Figure:
    """Matrizes de confusão lado a lado para cada modelo."""
    n = len(ml_results)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=list(ml_results.keys()),
                        horizontal_spacing=0.10)
    for i, (nome, res) in enumerate(ml_results.items(), 1):
        fig.add_trace(go.Heatmap(
            z=res['cm'],
            x=['Pred. Ruim', 'Pred. Boa'],
            y=['Real Ruim',  'Real Boa'],
            colorscale=[[0, NAVY], [1, ACCENT]],
            showscale=False,
            text=res['cm'],
            texttemplate='%{text}',
            textfont=dict(size=16, color=WHITE),
            xgap=3, ygap=3,
        ), row=1, col=i)
    return _fig(fig, height=460, margin=dict(l=90, r=40, t=60, b=60))


# ─────────────────────────────────────────────────────────────────────────────
# ABA SCORE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_histograma(df_full: pd.DataFrame) -> go.Figure:
    """Histograma de distribuição do Score FIDC."""
    fig = px.histogram(df_full, x='score_fidc', nbins=40,
                       color_discrete_sequence=[ACCENT],
                       labels={'score_fidc': 'Score FIDC (0–1000)'})
    return _fig(fig, height=380, margin=dict(l=70, r=30, t=20, b=60), bargap=0.04)


def fig_rating_barras(df_full: pd.DataFrame) -> go.Figure:
    """Barras horizontais de CNPJs por rating."""
    rc = (df_full['rating_carteira'].value_counts()
                                     .reset_index()
                                     .rename(columns={'rating_carteira': 'Rating', 'count': 'CNPJs'}))
    fig = px.bar(rc, x='CNPJs', y='Rating', orientation='h',
                 color='Rating', color_discrete_map=RATING_COLOR)
    return _fig(fig, height=380, margin=dict(l=200, r=60, t=20, b=60),
                showlegend=False, yaxis=dict(tickfont=dict(size=13)))


def fig_score_por_uf(df_full: pd.DataFrame) -> go.Figure:
    """Barras de score médio por UF."""
    uf_score = (df_full.groupby('uf')['score_fidc'].mean()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={'score_fidc': 'Score Médio'}))
    fig = px.bar(uf_score, x='uf', y='Score Médio',
                 color='Score Médio',
                 color_continuous_scale=[[0, WARN], [0.5, ACCENT], [1, ACCENT2]],
                 labels={'uf': 'UF', 'Score Médio': 'Score FIDC Médio'})
    return _fig(fig, height=380, margin=dict(l=70, r=30, t=20, b=60),
                coloraxis_showscale=True)


def fig_scatter(df_full: pd.DataFrame, x_col: str,
                x_label: str, y_label: str = 'Score FIDC') -> go.Figure:
    """Scatter plot genérico de variável vs Score FIDC, colorido por rating."""
    fig = px.scatter(df_full, x=x_col, y='score_fidc',
                     color='rating_carteira',
                     color_discrete_map=RATING_COLOR,
                     opacity=.6,
                     labels={x_col: x_label, 'score_fidc': y_label})
    return _fig(fig, height=420, margin=dict(l=70, r=30, t=20, b=70),
                legend=dict(orientation='h', y=1.06, font=dict(size=10)))



# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE SETORIAL — Gráficos por CNAE
# ─────────────────────────────────────────────────────────────────────────────

def fig_cnpjs_por_cnae(perfil_cnae: pd.DataFrame, top_n: int = 7) -> go.Figure:
    """
    Barras horizontais com os top N setores por quantidade de CNPJs na carteira.
    Colorido pelo score médio do setor — do vermelho (baixo) ao verde (alto).
    """
    df = perfil_cnae.head(top_n).copy()
    df = df.sort_values('qtd_cnpjs', ascending=True)  # crescente para barras horizontais

    fig = go.Figure(go.Bar(
        x=df['qtd_cnpjs'],
        y=df['denominacao_curta'],
        orientation='h',
        marker=dict(
            color=df['score_medio'],
            colorscale=[[0, WARN], [0.5, ACCENT], [1, ACCENT2]],
            showscale=True,
            colorbar=dict(title=dict(text='Score Médio', font=dict(color=WHITE, size=11)),
                          tickfont=dict(color=WHITE, size=10)),
        ),
        text=[f"{v:,} CNPJs" for v in df['qtd_cnpjs']],
        textposition='outside',
        textfont=dict(size=11, color=WHITE),
        customdata=df[['score_medio', 'pct_suspeitos', 'rating_predominante']].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "CNPJs: %{x:,}<br>"
            "Score médio: %{customdata[0]}<br>"
            "Suspeitos: %{customdata[1]:.1f}%<br>"
            "Rating predominante: %{customdata[2]}<extra></extra>"
        ),
    ))
    return _fig(fig, height=max(340, top_n * 52),
                margin=dict(l=340, r=100, t=20, b=50),
                xaxis_title='Quantidade de CNPJs',
                yaxis=dict(tickfont=dict(size=11)))


def fig_score_por_cnae(perfil_cnae: pd.DataFrame, top_n: int = 7) -> go.Figure:
    """
    Scatter com os top N setores: eixo X = score médio, eixo Y = qtd CNPJs,
    tamanho da bolha = % de suspeitos, cor = rating predominante.
    """
    df = perfil_cnae.head(top_n).copy()

    fig = px.scatter(
        df,
        x='score_medio',
        y='qtd_cnpjs',
        size=df['pct_suspeitos'].clip(lower=1),
        color='rating_predominante',
        color_discrete_map=RATING_COLOR,
        text='denominacao_curta',
        hover_data={'denominacao': True, 'qtd_cnpjs': True,
                    'score_medio': True, 'pct_suspeitos': True},
        labels={'score_medio': 'Score FIDC Médio', 'qtd_cnpjs': 'Qtd de CNPJs',
                'rating_predominante': 'Rating predominante'},
    )
    fig.update_traces(textposition='top center', textfont=dict(size=9, color=WHITE))
    return _fig(fig, height=420, margin=dict(l=70, r=30, t=30, b=70),
                legend=dict(orientation='h', y=1.08, font=dict(size=10)))


# ─────────────────────────────────────────────────────────────────────────────
# ABA FRAUDE — Detecção de Boletos Duplicados e Risco de Fraude
# ─────────────────────────────────────────────────────────────────────────────

def fig_duplicatas_tipo(stats: dict) -> go.Figure:
    """Barras comparando total de duplicatas por tipo (ID repetido vs Conteúdo idêntico)."""
    fig = go.Figure([
        go.Bar(name='ID repetido',       x=['Duplicatas'], y=[stats['total_dup_id']],
               marker_color=WARN,   opacity=.85),
        go.Bar(name='Conteúdo idêntico', x=['Duplicatas'], y=[stats['total_dup_conteudo']],
               marker_color=ACCENT, opacity=.85),
    ])
    return _fig(fig, height=320, barmode='group',
                margin=dict(l=70, r=30, t=20, b=50),
                yaxis_title='Qtd de boletos',
                legend=dict(orientation='h', y=1.05, font=dict(size=11)))


def fig_top_cnpjs_suspeitos(fraude_cnpj: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Barras horizontais dos CNPJs com maior % de boletos duplicados."""
    df = (fraude_cnpj[fraude_cnpj['bol_pct_duplicado'] > 0]
          .nlargest(top_n, 'bol_pct_duplicado')
          .copy())
    df['pct_label'] = (df['bol_pct_duplicado'] * 100).round(1)
    colors = [WARN if f == 1 else ACCENT for f in df['flag_risco_fraude']]
    fig = go.Figure(go.Bar(
        x=df['bol_pct_duplicado'] * 100,
        y=df['id_pagador'].astype(str),
        orientation='h',
        marker_color=colors,
        opacity=.85,
        text=[f"{v:.1f}%" for v in df['bol_pct_duplicado'] * 100],
        textposition='outside',
        textfont=dict(size=10, color=WHITE),
    ))
    return _fig(fig, height=max(320, top_n * 24), margin=dict(l=200, r=60, t=20, b=50),
                xaxis_title='% de boletos duplicados',
                yaxis=dict(tickfont=dict(size=10)))


def fig_emitentes_por_pagador(fraude_cnpj: pd.DataFrame) -> go.Figure:
    """Histograma da distribuição do número de emitentes distintos por pagador."""
    fig = px.histogram(fraude_cnpj, x='bol_n_emitentes', nbins=30,
                       color_discrete_sequence=[ACCENT],
                       labels={'bol_n_emitentes': 'Nº de beneficiários distintos'})
    return _fig(fig, height=340, bargap=0.05,
                margin=dict(l=70, r=30, t=20, b=60),
                yaxis_title='Qtd de CNPJs')


def fig_scatter_fraude(df_full: pd.DataFrame) -> go.Figure:
    """Scatter: % duplicados vs Score FIDC, destacando CNPJs com flag de fraude."""
    if 'bol_pct_duplicado' not in df_full.columns:
        return go.Figure()
    df = df_full.copy()
    df['suspeito'] = df.get('flag_risco_fraude', 0).map({0: 'Normal', 1: 'Suspeito'})
    fig = px.scatter(df, x='bol_pct_duplicado', y='score_fidc',
                     color='suspeito',
                     color_discrete_map={'Normal': ACCENT, 'Suspeito': WARN},
                     opacity=.65,
                     labels={'bol_pct_duplicado': '% Boletos Duplicados',
                             'score_fidc': 'Score FIDC'})
    return _fig(fig, height=400, margin=dict(l=70, r=30, t=20, b=70),
                legend=dict(orientation='h', y=1.06, font=dict(size=11)))


def fig_resumo_duplicatas(resumo: pd.DataFrame) -> go.Figure:
    """Barras dos grupos duplicados com maior quantidade de ocorrências."""
    df = resumo.head(20).copy()
    if df.empty:
        return go.Figure()
    colors = [WARN if t == 'ID repetido' else ACCENT for t in df['tipo_duplicata']]
    fig = go.Figure(go.Bar(
        x=df['qtd_ocorrencias'],
        y=[f"{row.id_pagador[:12]}… R${row.vlr_nominal:,.0f}" for row in df.itertuples()],
        orientation='h',
        marker_color=colors,
        opacity=.85,
    ))
    return _fig(fig, height=max(320, len(df) * 22), margin=dict(l=230, r=50, t=20, b=50),
                xaxis_title='Ocorrências do grupo',
                yaxis=dict(tickfont=dict(size=9)))


# ─────────────────────────────────────────────────────────────────────────────
# ABA MACRO — Contexto Macroeconômico Externo (Camada 1)
# Fontes: BCB SGS e IBGE SIDRA
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_macro_setores(scores_setor: dict) -> go.Figure:
    """Barras horizontais com o Score Macro por setor presente na carteira."""
    if not scores_setor:
        return go.Figure()

    nomes  = [s.replace('_', ' ').title() for s in scores_setor]
    vals   = [v['score'] for v in scores_setor.values()]
    cores  = [v['cor']   for v in scores_setor.values()]
    niveis = [v['nivel'] for v in scores_setor.values()]

    ordem = sorted(range(len(vals)), key=lambda i: vals[i])
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in ordem],
        y=[nomes[i] for i in ordem],
        orientation='h',
        marker_color=[cores[i] for i in ordem],
        text=[f"{vals[i]:.0f} — {niveis[i]}" for i in ordem],
        textposition='outside',
        textfont=dict(size=10, color=WHITE),
    ))
    fig.add_vline(x=75, line_dash="dash", line_color="#00cc70",
                  annotation_text="Favorável", annotation_font_color="#00cc70")
    fig.add_vline(x=55, line_dash="dash", line_color=WARN,
                  annotation_text="Atenção", annotation_font_color=WARN)
    return _fig(fig, height=max(300, len(scores_setor)*60),
                margin=dict(l=130, r=100, t=30, b=50),
                xaxis=dict(range=[0, 115], title='Score Macro Setorial (0–100)'),
                yaxis=dict(tickfont=dict(size=12)))


def fig_inadimplencia_historico(historico: dict) -> go.Figure:
    """Linhas históricas de inadimplência PJ por setor (últimos 12 meses BCB)."""
    series_plot = {
        'Total PJ':    'inadimplencia_pj_total',
        'Comércio':    'inadimplencia_pj_comercio',
        'Serviços':    'inadimplencia_pj_servicos',
        'Indústria':   'inadimplencia_pj_industria',
    }
    cores = [WARN, ACCENT, ACCENT2, '#a78bfa']
    fig = go.Figure()
    for (nome, chave), cor in zip(series_plot.items(), cores):
        df = historico.get(chave, pd.DataFrame())
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df['data'], y=df['valor'],
                name=nome, mode='lines+markers',
                line=dict(width=2, color=cor),
                marker=dict(size=5),
            ))
        else:
            # Dados de fallback simulados para visualização
            import numpy as np
            datas = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='ME')
            base  = {'Total PJ': 3.6, 'Comércio': 3.9, 'Serviços': 4.1, 'Indústria': 2.8}
            vals  = base[nome] + np.random.normal(0, 0.15, 12).cumsum() * 0.1
            fig.add_trace(go.Scatter(
                x=datas, y=np.clip(vals, 1.0, 8.0),
                name=f"{nome} (ref.)", mode='lines+markers',
                line=dict(width=2, color=cor, dash='dot'),
                marker=dict(size=5),
            ))
    return _fig(fig, height=380, margin=dict(l=70, r=30, t=30, b=60),
                xaxis_title='Período', yaxis_title='Inadimplência PJ (%)',
                legend=dict(orientation='h', y=1.08, font=dict(size=11)))


def fig_selic_ipca(historico: dict, indicadores: dict) -> go.Figure:
    """Gráfico duplo: SELIC e IPCA histórico com valores atuais destacados."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['SELIC Meta (% a.a.)', 'IPCA Acumulado 12m (%)'],
                        horizontal_spacing=0.12)

    for (chave, col, cor) in [
        ('selic_meta', 1, ACCENT),
        ('ipca_acumulado_12m', 2, WARN),
    ]:
        df = historico.get(chave, pd.DataFrame())
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df['data'], y=df['valor'],
                mode='lines+markers', line=dict(width=2.5, color=cor),
                marker=dict(size=5), showlegend=False,
            ), row=1, col=col)
            # Último valor destacado
            ultimo = df.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[ultimo['data']], y=[ultimo['valor']],
                mode='markers+text',
                marker=dict(size=12, color=cor, symbol='circle'),
                text=[f"  {ultimo['valor']:.2f}%"],
                textfont=dict(color=WHITE, size=11), showlegend=False,
            ), row=1, col=col)

    return _fig(fig, height=360, margin=dict(l=70, r=30, t=50, b=60))


def fig_pib_variacao(indicadores: dict) -> go.Figure:
    """Barras mostrando variação do PIB anual e trimestral."""
    categorias = ['PIB Anual (%)', 'PIB Trimestral (%)']
    valores    = [
        indicadores.get('pib_variacao_anual', 2.9),
        indicadores.get('pib_variacao_trimestral', 0.8),
    ]
    cores = [ACCENT2 if v >= 0 else WARN for v in valores]
    fig = go.Figure(go.Bar(
        x=categorias, y=valores,
        marker_color=cores, opacity=0.85,
        text=[f"{v:+.2f}%" for v in valores],
        textposition='outside',
        textfont=dict(size=14, color=WHITE),
        width=0.4,
    ))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    return _fig(fig, height=340, margin=dict(l=70, r=70, t=30, b=60),
                yaxis_title='Variação (%)',
                yaxis=dict(zeroline=True, zerolinecolor=MUTED))


def fig_componentes_score_macro(score_detalhe: dict, setor: str) -> go.Figure:
    """Radar/spider mostrando os 5 componentes do Score Macro para um setor."""
    categorias = [
        'Inadimplência<br>Setorial',
        'PIB<br>Crescimento',
        'SELIC<br>(risco juros)',
        'IPCA<br>(inflação)',
        'Indicador<br>Setorial',
    ]
    valores = [
        score_detalhe.get('comp_inadimp', 50),
        score_detalhe.get('comp_pib',     50),
        score_detalhe.get('comp_selic',   50),
        score_detalhe.get('comp_ipca',    50),
        score_detalhe.get('comp_setor',   50),
    ]
    # Fechar o radar
    categorias_c = categorias + [categorias[0]]
    valores_c    = valores + [valores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=valores_c, theta=categorias_c,
        fill='toself',
        fillcolor=f"rgba(0,196,239,0.15)",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=6, color=ACCENT),
    ))
    return _fig(fig, height=380, margin=dict(l=60, r=60, t=40, b=60),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor=BORDER, color=MUTED, tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor=BORDER, color=MUTED, tickfont=dict(size=10)),
                    bgcolor='rgba(0,0,0,0)',
                ))


# ─────────────────────────────────────────────────────────────────────────────
# PRODUTO 2 — Probabilidade ML vs Score FIDC
# ─────────────────────────────────────────────────────────────────────────────

def fig_divergencia_ml(df_full: pd.DataFrame) -> go.Figure:
    """
    Scatter: Score FIDC (eixo X) vs Probabilidade ML de ser bom (eixo Y).
    Cada ponto é um CNPJ. Quadrantes de divergência destacados.
    """
    if 'prob_ml_bom' not in df_full.columns:
        return go.Figure()

    df = df_full.copy()
    df['diverge'] = df['alerta_divergencia'].map({0: 'Consenso', 1: 'Divergência'})
    df['cor'] = df['diverge'].map({'Consenso': ACCENT, 'Divergência': WARN})

    fig = go.Figure()

    for grupo, cor in [('Consenso', ACCENT), ('Divergência', WARN)]:
        d = df[df['diverge'] == grupo]
        fig.add_trace(go.Scatter(
            x=d['score_fidc'], y=d['prob_ml_bom'],
            mode='markers', name=grupo,
            marker=dict(size=6, color=cor, opacity=0.65),
            customdata=d[['id_cnpj', 'rating_carteira']].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Score FIDC: %{x}<br>"
                "Prob. ML (bom): %{y:.1f}%<br>"
                "Rating: %{customdata[1]}<extra></extra>"
            ),
        ))

    # Linhas de corte
    fig.add_vline(x=700, line_dash="dash", line_color=MUTED, line_width=1)
    fig.add_vline(x=400, line_dash="dash", line_color=MUTED, line_width=1)
    fig.add_hline(y=60,  line_dash="dash", line_color=MUTED, line_width=1)
    fig.add_hline(y=40,  line_dash="dash", line_color=MUTED, line_width=1)

    # Anotações dos quadrantes de divergência
    fig.add_annotation(x=820, y=20, text="⚠️ Score alto<br>ML pessimista",
                       showarrow=False, font=dict(color=WARN, size=10),
                       bgcolor=CARD_BG, bordercolor=WARN, borderwidth=1)
    fig.add_annotation(x=200, y=80, text="⚠️ Score baixo<br>ML otimista",
                       showarrow=False, font=dict(color=WARN, size=10),
                       bgcolor=CARD_BG, bordercolor=WARN, borderwidth=1)

    return _fig(fig, height=440,
                margin=dict(l=70, r=30, t=30, b=70),
                xaxis=dict(title='Score FIDC (0–1000)', range=[0, 1050]),
                yaxis=dict(title='Probabilidade ML — bom (%)', range=[-5, 105]),
                legend=dict(orientation='h', y=1.06, font=dict(size=11)))


def fig_prob_ml_hist(df_full: pd.DataFrame) -> go.Figure:
    """Histograma da distribuição da probabilidade ML por rating."""
    if 'prob_ml_bom' not in df_full.columns:
        return go.Figure()

    fig = go.Figure()
    for rating, cor in RATING_COLOR.items():
        d = df_full[df_full['rating_carteira'] == rating]
        if d.empty:
            continue
        fig.add_trace(go.Histogram(
            x=d['prob_ml_bom'], name=rating,
            marker_color=cor, opacity=0.75,
            nbinsx=20,
        ))

    return _fig(fig, height=360, barmode='overlay',
                margin=dict(l=70, r=30, t=30, b=60),
                xaxis_title='Probabilidade ML — bom (%)',
                yaxis_title='Qtd de CNPJs',
                legend=dict(orientation='h', y=1.06, font=dict(size=10)))


# ─────────────────────────────────────────────────────────────────────────────
# INDICADOR DE RISCO SETORIAL — gráfico histórico com bandas
# ─────────────────────────────────────────────────────────────────────────────

def fig_risco_setorial(ind_risco: dict) -> go.Figure:
    """
    Gráfico de linha da inadimplência setorial histórica (24 meses)
    com faixa sombreada de referência (média ± 0.5σ) e ponto atual destacado.
    """
    if not ind_risco or ind_risco.get('df_historico') is None:
        return go.Figure()

    df   = ind_risco['df_historico'].copy()
    med  = ind_risco['media_24m']
    sup  = ind_risco['banda_sup']
    inf  = ind_risco['banda_inf']
    cor  = ind_risco['cor']
    tag  = ind_risco['tag']

    fig = go.Figure()

    # Faixa Regular (média ± 0.5σ)
    fig.add_trace(go.Scatter(
        x=pd.concat([df['data'], df['data'][::-1]]),
        y=[sup]*len(df) + [inf]*len(df),
        fill='toself',
        fillcolor='rgba(248,196,11,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Faixa Regular (±0.5σ)',
        showlegend=True,
    ))

    # Linha histórica
    fig.add_trace(go.Scatter(
        x=df['data'], y=df['valor'],
        mode='lines+markers',
        name='Inadimplência PJ (%)',
        line=dict(color=MUTED, width=2),
        marker=dict(size=4, color=MUTED),
    ))

    # Linha da média
    fig.add_hline(y=med, line_dash='dash', line_color=AMBER, line_width=1.5,
                  annotation_text=f'Média 24m: {med:.2f}%',
                  annotation_font_color=AMBER, annotation_position='right')

    # Ponto atual destacado
    ultimo = df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[ultimo['data']], y=[ultimo['valor']],
        mode='markers+text',
        marker=dict(size=14, color=cor, symbol='circle',
                    line=dict(color=WHITE, width=2)),
        text=[f"  {ultimo['valor']:.2f}% — {tag}"],
        textfont=dict(color=cor, size=11),
        textposition='middle right',
        name=f'Atual ({ind_risco["data_atual"]})',
        showlegend=True,
    ))

    return _fig(fig, height=380,
                margin=dict(l=70, r=120, t=30, b=60),
                xaxis_title='Período',
                yaxis_title='Inadimplência PJ (%)',
                legend=dict(orientation='h', y=1.08, font=dict(size=10)))