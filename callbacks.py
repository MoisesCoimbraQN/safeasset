import pandas as pd
# =============================================================================
# callbacks.py — SafeAsset
# Responsável por: toda a lógica reativa do Dash (callbacks)
# Conecta o frontend (layout.py) com os dados (pipeline.py) e gráficos (charts.py)
# =============================================================================

import base64, io
import numpy as np
import pandas as pd

from dash import dcc, html, dash_table, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

import pipeline as pl
import charts as ch
from layout import (card, kpi, section_title, build_rank_table,
                    ACCENT, ACCENT2, WARN, MUTED, BORDER, WHITE, NAVY, BLUE, CARD_BG)
# Aliases de compatibilidade — tema escuro
DARK   = WHITE    # texto claro sobre fundo escuro
AMBER  = '#F59E0B'  # âmbar — usado em alertas e destaques
BG_MAIN = CARD_BG # fundo dos cards
from pipeline import RATING_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — parse de upload CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_upload(contents, filename):
    """Decodifica conteúdo de upload Dash e retorna JSON string."""
    if not contents:
        return None
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df.to_json(date_format='iso', orient='split')
    except Exception:
        return None


def read_json(json_str):
    """Lê JSON string do dcc.Store para DataFrame."""
    return pd.read_json(io.StringIO(json_str), orient='split')


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO DOS CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def register_callbacks(app):
    """Registra todos os callbacks no app Dash."""


    # ── TOGGLE SIDEBAR ────────────────────────────────────────────────────
    @app.callback(
        Output('sidebar-container', 'style'),
        Output('sidebar-container', 'children'),
        Output('btn-toggle-sidebar', 'children'),
        Output('btn-toggle-sidebar', 'title'),
        Output('store-sidebar',      'data'),
        Input('btn-toggle-sidebar',  'n_clicks'),
        State('store-sidebar',       'data'),
        prevent_initial_call=True,
    )
    def toggle_sidebar(n, estado):
        from layout import build_sidebar as _bs

        BTN_STYLE = {
            'background': BLUE, 'color': ACCENT,
            'border': f'1px solid {BORDER}',
            'borderRadius': '50%', 'width': '22px', 'height': '22px',
            'fontSize': '10px', 'cursor': 'pointer', 'padding': '0',
            'marginTop': '8px',
        }

        if estado == 'open':
            # Recolher — sidebar vira barra fina com label vertical
            sidebar_content = html.Div(
                style={'display': 'flex', 'flexDirection': 'column',
                       'alignItems': 'center', 'paddingTop': '8px', 'width': '100%'},
                children=[
                    html.Div('PAINEL', style={
                        'color': MUTED, 'fontSize': '9px',
                        'writingMode': 'vertical-rl',
                        'textOrientation': 'mixed',
                        'letterSpacing': '2px',
                        'transform': 'rotate(180deg)',
                        'marginTop': '12px',
                    }),
                ]
            )
            sidebar_style = {
                'width': '18px', 'minWidth': '18px',
                'flexShrink': '0',
                'transition': 'width 0.2s ease, min-width 0.2s ease',
                'overflowX': 'hidden',
                'background': NAVY,
                'borderRight': f'1px solid {BORDER}',
            }
            return sidebar_style, sidebar_content, '▶', 'Expandir painel lateral', 'closed'

        else:
            # Expandir — sidebar completa
            sidebar_style = {
                'width': '280px', 'minWidth': '280px',
                'flexShrink': '0',
                'transition': 'width 0.2s ease, min-width 0.2s ease',
                'overflowX': 'hidden',
            }
            return sidebar_style, _bs(), '◀', 'Recolher painel lateral', 'open'

    # ── Upload base auxiliar (callback independente) ─────────────────────
    @app.callback(
        Output('store-raw-aux', 'data'),
        Output('aux-status',    'children'),
        Input('upload-aux', 'contents'),
        State('upload-aux', 'filename'),
        prevent_initial_call=True,
    )
    def handle_upload_aux(contents, filename):
        if not contents:
            return no_update, no_update
        print(f'[SafeAsset] Upload base auxiliar: {filename}')
        data = parse_upload(contents, filename)
        if data:
            print(f'[SafeAsset] Base auxiliar OK')
            return data, f'✅ {filename}'
        print('[SafeAsset] ERRO ao parsear base auxiliar')
        return no_update, '❌ Erro ao ler o arquivo'

    # ── Upload base boletos (callback independente) ────────────────────────
    @app.callback(
        Output('store-raw-bol', 'data'),
        Output('bol-status',    'children'),
        Input('upload-bol', 'contents'),
        State('upload-bol', 'filename'),
        prevent_initial_call=True,
    )
    def handle_upload_bol(contents, filename):
        if not contents:
            return no_update, no_update
        print(f'[SafeAsset] Upload base boletos: {filename}')
        data = parse_upload(contents, filename)
        if data:
            print(f'[SafeAsset] Base boletos OK')
            return data, f'✅ {filename}'
        print('[SafeAsset] ERRO ao parsear base boletos')
        return no_update, '❌ Erro ao ler o arquivo'

    # ── Buscar dados macro quando AMBOS os stores estiverem preenchidos ───
    @app.callback(
        Output('store-macro', 'data'),
        Input('store-raw-aux', 'data'),
        Input('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def handle_macro(aux_json, bol_json):
        if not aux_json or not bol_json:
            return no_update
        import json as _json
        try:
            from macro import buscar_indicadores
            macro_dados = buscar_indicadores()
            print(f"[SafeAsset] Macro OK — fonte: {macro_dados['fonte']}")
            return _json.dumps({
                'indicadores': macro_dados['indicadores'],
                'pib':         macro_dados['pib'],
                'data_coleta': macro_dados['data_coleta'],
                'fonte':       macro_dados['fonte'],
            })
        except Exception as e:
            # Mesmo com erro, retorna fallback para a aba Macro exibir algo
            print(f"[SafeAsset] Macro fallback por exceção: {e}")
            from macro import FALLBACK
            return _json.dumps({
                'indicadores': {k: v for k, v in FALLBACK.items()
                                if not isinstance(v, str)},
                'pib': {'pib_variacao_anual': FALLBACK['pib_variacao_anual'],
                        'pib_variacao_trimestral': FALLBACK['pib_variacao_trimestral']},
                'data_coleta': FALLBACK['data_referencia'],
                'fonte': 'fallback',
            })

    # ── Popular filtros UF/CNAE após upload ───────────────────────────────
    @app.callback(
        Output('flt-uf',   'options'),
        Output('flt-cnae', 'options'),
        Input('store-raw-aux', 'data'),
    )
    def update_filter_options(aux_json):
        if not aux_json:
            return [], []
        df = read_json(aux_json)
        ufs   = [{'label': u, 'value': u} for u in sorted(df['uf'].dropna().unique())]
        cnaes = [{'label': str(c), 'value': str(c)}
                 for c in sorted(df['cd_cnae_prin'].dropna().astype(str).unique())]
        return ufs, cnaes

    # ── Limpar filtros ─────────────────────────────────────────────────────
    @app.callback(
        Output('flt-cnpj', 'value'),
        Output('flt-uf',   'value'),
        Output('flt-cnae', 'value'),
        Output('flt-date', 'start_date'),
        Output('flt-date', 'end_date'),
        Input('btn-clear', 'n_clicks'),
        prevent_initial_call=True,
    )
    def clear_filters(_):
        return '', [], [], None, None

    # ── CALLBACK PRINCIPAL — executa pipeline e monta dashboard ───────────
    @app.callback(
        Output('main-content',   'children'),
        Output('loading-output', 'children'),
        Input('btn-run',         'n_clicks'),
        Input('btn-run-upload',  'n_clicks'),
        Input('btn-run-ml',      'n_clicks'),
        Input('btn-run-target',  'n_clicks'),
        Input('btn-run-fraud',   'n_clicks'),
        Input('store-raw-aux',   'data'),
        Input('store-raw-bol',   'data'),   # dispara quando bol é carregado
        Input('flt-cnpj',        'value'),
        Input('flt-uf',          'value'),
        Input('flt-cnae',        'value'),
        Input('flt-date',        'start_date'),
        Input('flt-date',        'end_date'),
        State('sl-test',  'value'),
        State('sl-trees', 'value'),
        State('sl-liq',   'value'),
        State('sl-mat',   'value'),
        State('sl-dup-thresh',  'value'),
        State('sl-emit-thresh', 'value'),
        prevent_initial_call=True,
    )
    def run_dashboard(run_clicks, run_upload_clicks, ml_clicks, target_clicks, fraud_clicks,
                      aux_json_input, bol_json,
                      cnpj_q, sel_ufs, sel_cnaes, date_from, date_to,
                      test_size, n_trees, liq_thresh, mat_thresh,
                      dup_thresh, emit_thresh):

        from dash.exceptions import PreventUpdate

        # Aguardar ambos os arquivos — mas não bloquear se um já estava no store
        if not aux_json_input and not bol_json:
            raise PreventUpdate

        if not aux_json_input:
            return html.Div([
                html.Div('📂', style={'fontSize': '48px', 'marginBottom': '12px'}),
                html.Div('Aguardando base auxiliar…', style={'color': MUTED, 'fontSize': '15px'}),
                html.Div('Faça o upload do arquivo base_auxiliar_fiap.csv',
                         style={'color': WARN, 'fontSize': '13px', 'marginTop': '6px'}),
            ], style={'textAlign': 'center', 'padding': '60px'}), ''

        if not bol_json:
            return html.Div([
                html.Div('📂', style={'fontSize': '48px', 'marginBottom': '12px'}),
                html.Div('Base auxiliar carregada ✅', style={'color': ACCENT2, 'fontSize': '15px'}),
                html.Div('Agora faça o upload do arquivo base_boletos_fiap.csv',
                         style={'color': MUTED, 'fontSize': '13px', 'marginTop': '6px'}),
            ], style={'textAlign': 'center', 'padding': '60px'}), ''

        try:
            df_aux = read_json(aux_json_input)
        except Exception as e:
            return html.Div(f'❌ Erro ao ler base auxiliar: {e}',
                            style={'color': WARN, 'padding': '40px'}), ''

        try:
            df_bol = read_json(bol_json)
        except Exception as e:
            return html.Div(f'❌ Erro ao ler base de boletos: {e}',
                            style={'color': WARN, 'padding': '40px'}), ''

        # ── Aplicar filtros globais ───────────────────────────────────────
        if cnpj_q:
            df_aux = df_aux[df_aux['id_cnpj'].astype(str).str.lower()
                            .str.contains(cnpj_q.strip().lower(), na=False)]
        if sel_ufs:
            df_aux = df_aux[df_aux['uf'].isin(sel_ufs)]
        if sel_cnaes:
            df_aux = df_aux[df_aux['cd_cnae_prin'].astype(str).isin(sel_cnaes)]
        if df_bol is not None and (date_from or date_to):
            ds = pd.to_datetime(df_bol['dt_emissao'], errors='coerce')
            if date_from: df_bol = df_bol[ds >= pd.Timestamp(date_from)]
            if date_to:   df_bol = df_bol[ds <= pd.Timestamp(date_to)]
            valid = set(df_bol['id_pagador'].tolist())
            if valid: df_aux = df_aux[df_aux['id_cnpj'].isin(valid)]

        if len(df_aux) < 10:
            return html.Div('⚠️ Filtro muito restritivo — poucos dados disponíveis.',
                            style={'color': WARN, 'padding': '40px', 'textAlign': 'center'}), ''

        # ── Executar pipeline ─────────────────────────────────────────────
        try:
            R = pl.run_pipeline(
                df_aux.copy(), df_bol.copy(),
                test_size          = test_size  or 0.2,
                n_estimators       = n_trees    or 300,
                liq_thresh         = liq_thresh or 0.65,
                mat_thresh         = mat_thresh or 800,
                pct_dup_thresh     = (dup_thresh  or 5)  / 100,
                n_emitentes_thresh = emit_thresh or 10,
            )
        except Exception as e:
            import traceback
            return html.Div([
                html.Div('❌ Erro ao executar o pipeline:',
                         style={'color': WARN, 'fontWeight': '700', 'marginBottom': '8px'}),
                html.Pre(traceback.format_exc(),
                         style={'color': MUTED, 'fontSize': '11px', 'background': '#0d1b2a',
                                'padding': '12px', 'borderRadius': '6px', 'overflow': 'auto'}),
            ], style={'padding': '24px'}), ''

        # ── Montar dashboard ──────────────────────────────────────────────
        dashboard = build_dashboard(R, liq_thresh or 0.65, mat_thresh or 800)
        return dashboard, ''

    # ── Filtros do ranking ─────────────────────────────────────────────────
    @app.callback(
        Output('rank-table-container', 'children'),
        Input('rank-cnpj',   'value'),
        Input('rank-uf',     'value'),
        Input('rank-cnae',   'value'),
        Input('rank-rating', 'value'),
        Input('rank-score',  'value'),
        State('store-raw-aux', 'data'),
        State('store-raw-bol', 'data'),
        State('sl-test',  'value'),
        State('sl-trees', 'value'),
        State('sl-liq',   'value'),
        State('sl-mat',   'value'),
        prevent_initial_call=True,
    )
    def update_rank_table(cnpj_q, sel_ufs, sel_cnaes, sel_ratings, score_range,
                          aux_json, bol_json, test_size, n_trees, liq_thresh, mat_thresh):
        if not aux_json or not bol_json:
            return html.Div('Dados não carregados.', style={'color': MUTED})
        df_aux = read_json(aux_json)
        df_bol = read_json(bol_json)
        R = pl.run_pipeline(df_aux.copy(), df_bol.copy(),
                             test_size=test_size or 0.2, n_estimators=n_trees or 300,
                             liq_thresh=liq_thresh or 0.65, mat_thresh=mat_thresh or 800)
        return build_rank_table(R['df_full'], cnpj_q, sel_ufs, sel_cnaes, sel_ratings, score_range)

    # ── CALLBACK MACRO — dispara quando aba é aberta ou store muda ──────
    @app.callback(
        Output('macro-content', 'children'),
        Input('tabs-main',      'value'),
        State('store-macro',    'data'),
        State('store-raw-aux',  'data'),
        State('store-raw-bol',  'data'),
        prevent_initial_call=True,
    )
    def update_macro_content(tab_ativa, macro_store_json, aux_json, bol_json):
        if tab_ativa != 'tab-macro':
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        if not aux_json or not bol_json:
            return html.Div(
                '🌐 Faça o upload dos dois arquivos CSV para carregar os dados macroeconômicos.',
                style={'color': MUTED, 'fontSize': '14px',
                       'textAlign': 'center', 'padding': '40px'}
            )
        # Se o store-macro ainda não chegou, buscar direto (fallback síncrono)
        if not macro_store_json:
            import json as _j
            try:
                from macro import buscar_indicadores
                _d = buscar_indicadores()
                macro_store_json = _j.dumps({
                    'indicadores': _d['indicadores'], 'pib': _d['pib'],
                    'data_coleta': _d['data_coleta'], 'fonte': _d['fonte'],
                })
            except Exception:
                from macro import FALLBACK
                macro_store_json = _j.dumps({
                    'indicadores': {k: v for k, v in FALLBACK.items()
                                    if not isinstance(v, str)},
                    'pib': {'pib_variacao_anual': FALLBACK['pib_variacao_anual'],
                            'pib_variacao_trimestral': FALLBACK['pib_variacao_trimestral']},
                    'data_coleta': FALLBACK['data_referencia'], 'fonte': 'fallback',
                })
        import json as _json
        from macro import enriquecer_perfil_com_macro, calcular_score_macro_setor, CNAE_SETOR
        print("[SafeAsset] update_macro_content iniciando...")
        try:
            dados  = _json.loads(macro_store_json)
            ind    = dados.get('indicadores', {})
            fonte  = dados.get('fonte', '—')
            coleta = dados.get('data_coleta', '—')

            # Reconstruir perfil CNAE a partir da base auxiliar
            df_aux = read_json(aux_json)
            from pipeline import calcular_perfil_cnae as _cpf
            # Criar df_full mínimo para calcular perfil CNAE
            from macro import enriquecer_perfil_com_macro, calcular_score_macro_setor, CNAE_SETOR
            import pandas as _pd

            # Formatar CNAE e agrupar
            from pipeline import _formatar_cnae
            df_aux = df_aux.copy()
            df_aux['cd_cnae_fmt'] = df_aux['cd_cnae_prin'].apply(_formatar_cnae)
            perfil = (df_aux.groupby('cd_cnae_fmt')
                      .size().reset_index(name='qtd_cnpjs'))

            # Calcular scores por setor
            scores = {}
            for cnae_fmt in perfil['cd_cnae_fmt'].dropna().unique():
                div   = str(cnae_fmt).split('.')[0]
                setor = CNAE_SETOR.get(div, 'servicos')
                if setor not in scores:
                    scores[setor] = calcular_score_macro_setor(ind, setor)

            # Enriquecer perfil com macro
            perfil['setor_macro'] = perfil['cd_cnae_fmt'].apply(
                lambda x: CNAE_SETOR.get(str(x).split('.')[0], 'servicos')
            )
            perfil['score_macro'] = perfil['setor_macro'].apply(
                lambda s: calcular_score_macro_setor(ind, s)['score']
            )
            perfil['nivel_macro'] = perfil['setor_macro'].apply(
                lambda s: calcular_score_macro_setor(ind, s)['nivel']
            )

            # Calcular Indicador de Risco Setorial
            df_bol = read_json(bol_json)
            from macro import calcular_indicador_risco_setorial
            try:
                ind_risco = calcular_indicador_risco_setorial(df_aux, df_bol)
            except Exception as _e:
                print(f"[SafeAsset] Erro ind_risco: {_e}")
                ind_risco = None

            def G(fig):
                return dcc.Graph(figure=fig, config={'displayModeBar': False})

            alerta = [] if fonte == 'api' else [
                html.Div([
                    html.Span('⚠️  ', style={'fontSize': '15px'}),
                    html.Span(
                        'Sem conexão com BCB/IBGE — exibindo dados de referência recentes.',
                        style={'fontSize': '11px'}),
                ], style={'background': '#2d1a00', 'border': '1px solid #F59E0B',
                          'borderRadius': '8px', 'padding': '8px 12px',
                          'color': '#F59E0B', 'marginBottom': '12px'}),
            ]

            print('[SafeAsset] Macro content montado com sucesso')
            return html.Div([
                *alerta,
                html.Div(f'Dados: {coleta}  ·  Origem: {fonte}',
                         style={'fontSize': '11px', 'color': MUTED,
                                'marginBottom': '16px', 'fontStyle': 'italic'}),

                # ── Indicador de Risco Setorial ───────────────────────────
                *([
                    card([
                        html.Div('Indicador de Risco Setorial',
                                 style={'fontSize': '16px', 'fontWeight': '700',
                                        'color': WHITE, 'marginBottom': '4px'}),
                        html.Div(
                            f'Setor predominante: {ind_risco["setor_label"]} '
                            f'({ind_risco["pct_valor"]:.1f}% do valor da carteira) · '
                            f'Série BCB {ind_risco["codigo_serie"]} · '
                            f'Referência: últimos 24 meses',
                            style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),

                        dbc.Row([
                            dbc.Col([
                                # Tag principal
                                html.Div([
                                    html.Span(ind_risco['emoji'] + '  ',
                                              style={'fontSize': '24px'}),
                                    html.Span(ind_risco['tag'],
                                              style={'fontSize': '22px', 'fontWeight': '800',
                                                     'color': ind_risco['cor']}),
                                ], style={'marginBottom': '12px'}),
                                html.Div(ind_risco['interpretacao'],
                                         style={'fontSize': '13px', 'color': WHITE,
                                                'lineHeight': '1.6', 'marginBottom': '16px'}),
                                dbc.Row([
                                    dbc.Col(kpi('Inadimplência Atual',
                                        f'{ind_risco["valor_atual"]:.2f}%',
                                        f'Referência: {ind_risco["data_atual"]}',
                                        ind_risco['cor']), width=4),
                                    dbc.Col(kpi('Média Histórica 24m',
                                        f'{ind_risco["media_24m"]:.2f}%',
                                        'Base de comparação BCB',
                                        ACCENT), width=4),
                                    dbc.Col(kpi('Z-score',
                                        f'{ind_risco["z_score"]:+.2f}σ',
                                        f'Faixa Regular: ±0.5σ',
                                        AMBER), width=4),
                                ], className='g-2'),
                            ], width=5),
                            dbc.Col([
                                html.Div('Série Histórica — Inadimplência PJ do Setor',
                                         style={'fontSize': '12px', 'color': MUTED,
                                                'marginBottom': '6px'}),
                                G(ch.fig_risco_setorial(ind_risco)),
                            ], width=7),
                        ], className='g-3'),
                    ]),
                ] if ind_risco else [
                    html.Div('Indicador de Risco Setorial não disponível.',
                             style={'color': MUTED, 'fontSize': '13px', 'padding': '12px'}),
                ]),

                html.Hr(style={'borderColor': BORDER, 'margin': '8px 0 20px'}),

                # KPIs
                dbc.Row([
                    dbc.Col(kpi('SELIC Meta',    f'{ind.get("selic_meta","—")}%',            'taxa básica de juros', WARN),   width=2),
                    dbc.Col(kpi('IPCA 12m',      f'{ind.get("ipca_acumulado_12m","—")}%',    'inflação acumulada',   WARN),   width=2),
                    dbc.Col(kpi('Câmbio BRL/USD',f'R$ {ind.get("cambio_brl_usd","—")}',      'cotação atual',        ACCENT), width=2),
                    dbc.Col(kpi('Inadimp. PJ',   f'{ind.get("inadimplencia_pj_total","—")}%','% carteira BCB',       WARN),   width=2),
                    dbc.Col(kpi('PIB Anual',     f'{ind.get("pib_variacao_anual","—")}%',    'variação IBGE',        ACCENT2),width=2),
                    dbc.Col(kpi('Spread PJ',     f'{ind.get("spread_pj","—")} p.p.',         'spread médio PJ',      WARN),   width=2),
                ], className='g-3', style={'marginBottom': '20px'}),

                # Score macro + PIB
                dbc.Row([
                    dbc.Col([
                        card([
                            html.Div('Score Macroeconômico por Setor da Carteira',
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '4px'}),
                            html.Div('Inadimplência setorial (40%) · PIB (25%) · SELIC (15%) · IPCA (10%) · Indicador setorial (10%)',
                                     style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '10px'}),
                            G(ch.fig_score_macro_setores(scores)),
                        ]),
                    ], width=7),
                    dbc.Col([
                        card([
                            html.Div('Variação do PIB (IBGE)',
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_pib_variacao(ind)),
                        ]),
                    ], width=5),
                ], className='g-3'),

                # Tabela setorial
                card([
                    html.Div('Ranking Setorial — Score FIDC × Contexto Macroeconômico',
                             style={'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '12px'}),
                    dash_table.DataTable(
                        data=(perfil
                              .rename(columns={'cd_cnae_fmt': 'CNAE', 'qtd_cnpjs': 'CNPJs',
                                               'setor_macro': 'Grupo Macro',
                                               'score_macro': 'Score Macro',
                                               'nivel_macro': 'Contexto'})
                              .sort_values('Score Macro')
                              .to_dict('records')),
                        columns=[{'name': c, 'id': c}
                                 for c in ['CNAE', 'CNPJs', 'Grupo Macro', 'Score Macro', 'Contexto']],
                        page_size=12, sort_action='native',
                        style_table={'overflowX': 'auto'},
                        style_cell={'backgroundColor': CARD_BG, 'color': WHITE,
                                    'border': f'1px solid {BORDER}', 'fontFamily': 'Space Grotesk',
                                    'fontSize': '12px', 'padding': '8px 12px', 'textAlign': 'left'},
                        style_header={'backgroundColor': BLUE, 'fontWeight': '700',
                                      'color': ACCENT, 'border': f'1px solid {BORDER}'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{Contexto} = "Favorável"'},   'color': '#00cc70', 'fontWeight': '600'},
                            {'if': {'filter_query': '{Contexto} = "Neutro"'},      'color': '#F59E0B', 'fontWeight': '600'},
                            {'if': {'filter_query': '{Contexto} = "Atenção"'},     'color': '#ff6b35', 'fontWeight': '600'},
                            {'if': {'filter_query': '{Contexto} = "Desfavorável"'},'color': '#EF4444', 'fontWeight': '600'},
                            {'if': {'row_index': 'odd'}, 'backgroundColor': '#0d1b2a'},
                        ],
                    ),
                ]),
            ])

        except Exception:
            import traceback
            err = traceback.format_exc()
            print(f"[SafeAsset] ERRO no update_macro_content:\n{err}")
            return html.Div([
                html.Div('❌ Erro ao carregar dados macro:',
                         style={'color': WARN, 'fontWeight': '700', 'marginBottom': '8px'}),
                html.Pre(err,
                         style={'color': MUTED, 'fontSize': '10px', 'background': '#0d1b2a',
                                'padding': '12px', 'borderRadius': '6px', 'overflow': 'auto',
                                'maxHeight': '300px'}),
            ], style={'padding': '24px'})



# ─────────────────────────────────────────────────────────────────────────────
# MONTAGEM DO DASHBOARD (chamado dentro do callback principal)
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(R: dict, liq_thresh: float, mat_thresh: float):
    """
    Monta todo o conteúdo do dashboard a partir dos resultados do pipeline.
    Usa charts.py para gerar os gráficos e layout.py para os componentes visuais.
    """
    df_full      = R['df_full']
    df_bol       = R['df_bol']
    ml           = R['ml']
    best_name    = R['best_name']
    feat_imp     = R['feat_imp']
    corr_mat     = R['corr_matrix']
    fraude       = R['fraude']
    fraude_stats = fraude['stats']
    fraude_cnpj  = fraude['fraude_por_cnpj']
    resumo_dup   = fraude['resumo_duplicatas']
    perfil_cnae  = R['perfil_cnae']

    # macro vazio por padrão — preenchido pelo update_macro_content separadamente
    # Usado apenas para o semáforo setorial no Resumo (se já disponível no R)
    macro = R.get('macro') or {}

    # ── KPIs principais ───────────────────────────────────────────────────
    total       = len(df_full)
    rating_ct   = df_full['rating_carteira'].value_counts()
    n_excel     = rating_ct.get('A — Excelente', 0)
    score_med   = df_full['score_fidc'].mean()
    best_auc    = ml[best_name]['auc']
    boas_pct    = df_full['target'].mean() * 100
    n_suspeitos = fraude_stats['cnpjs_suspeitos']

    kpi_row = dbc.Row([
        dbc.Col(kpi('Total CNPJs',    f'{total:,}',        'base analisada'),                        width=2),
        dbc.Col(kpi('Rating A',       f'{n_excel:,}',      f'{n_excel/total*100:.0f}%',    ACCENT2), width=2),
        dbc.Col(kpi('Score Médio',    f'{score_med:.0f}',  '/ 1000',                       ACCENT),  width=2),
        dbc.Col(kpi('Carteiras Boas', f'{boas_pct:.1f}%',  'target = 1'),                            width=2),
        dbc.Col(kpi('AUC-ROC',        f'{best_auc:.4f}',   best_name.split()[0],           ACCENT2), width=2),
        dbc.Col(kpi('Suspeitos',      f'{n_suspeitos:,}',  'CNPJs c/ flag fraude',         WARN),    width=2),
    ], className='g-3', style={'marginBottom': '20px'})

    # ── TAB STYLE ─────────────────────────────────────────────────────────
    tab_style = {'color': MUTED, 'background': CARD_BG}
    tab_sel   = {'color': NAVY,  'background': ACCENT, 'fontWeight': '700'}

    def G(fig, cfg=None):
        """Wrapper para dcc.Graph com config padrão."""
        return dcc.Graph(figure=fig, config=cfg or {'displayModeBar': False})

    # ── TABS ──────────────────────────────────────────────────────────────
    tabs = dcc.Tabs(
        id='tabs-main',
        value='tab-resumo',
        style={'marginBottom': '16px'},
        colors={'border': BORDER, 'primary': ACCENT, 'background': CARD_BG},
        children=[

            # ── EDA ───────────────────────────────────────────────────────
            # ── RESUMO INDICATIVO ─────────────────────────────────────────
            dcc.Tab(label='📋 Resumo', value='tab-resumo',
                    style=tab_style,
                    selected_style={'color': NAVY, 'background': ACCENT2, 'fontWeight': '700'},
              children=[html.Div(style={'padding': '24px'}, children=[

                html.Div([
                    html.Div('Resumo Indicativo de Carteira',
                             style={'fontSize': '22px', 'fontWeight': '700', 'color': WHITE}),
                    html.Div('Priorização para o analista de crédito — visão consolidada dos 4 indicadores principais',
                             style={'fontSize': '13px', 'color': MUTED, 'marginTop': '4px'}),
                ], style={'marginBottom': '24px'}),

                # ── Veredicto geral ───────────────────────────────────────
                html.Div([
                    html.Span(
                        '✅ CARTEIRA RECOMENDÁVEL' if df_full['score_fidc'].mean() >= 600
                        else ('⚠️ CARTEIRA COM RESSALVAS' if df_full['score_fidc'].mean() >= 400
                              else '🚨 CARTEIRA DE ALTO RISCO'),
                        style={
                            'background': ACCENT2 if df_full['score_fidc'].mean() >= 600
                                          else (AMBER if df_full['score_fidc'].mean() >= 400 else WARN),
                            'color': NAVY, 'fontWeight': '800', 'fontSize': '15px',
                            'padding': '8px 24px', 'borderRadius': '24px',
                        }
                    ),
                ], style={'marginBottom': '28px'}),

                # ════════════════════════════════════════════════════════
                # BLOCO 0 — Recomendação pelo Target
                # ════════════════════════════════════════════════════════
                card([
                    html.Div('0 — Recomendação segundo o Target',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': ACCENT, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {ACCENT}',
                                    'paddingLeft': '10px'}),
                    html.Div('Classificação binária por regras de negócio Núclea (liquidez + materialidade + atraso)',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi(
                            'Carteiras Boas',
                            f'{int(df_full["target"].sum()):,}',
                            f'{df_full["target"].mean()*100:.1f}% da base — target = 1',
                            ACCENT2), width=4),
                        dbc.Col(kpi(
                            'Carteiras Ruins',
                            f'{int((df_full["target"]==0).sum()):,}',
                            f'{(df_full["target"]==0).mean()*100:.1f}% da base — target = 0',
                            WARN), width=4),
                        dbc.Col(kpi(
                            'Total Analisado',
                            f'{len(df_full):,}',
                            'CNPJs na carteira', ACCENT), width=4),
                    ], className='g-3'),
                ]),

                # ════════════════════════════════════════════════════════
                # BLOCO 1 — Score Final de Qualidade
                # ════════════════════════════════════════════════════════
                card([
                    html.Div('1 — Score Final de Qualidade (0–1000)',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': ACCENT2, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {ACCENT2}',
                                    'paddingLeft': '10px'}),
                    html.Div('Fórmula ponderada: Liquidez (35%) + Materialidade (25%) + Quantidade (15%) + Liq.3m (10%) + Atraso (8%) + Inadimplência (7%)',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi('Score Médio',
                            f'{df_full["score_fidc"].mean():.0f}',
                            f'Mín: {df_full["score_fidc"].min():.0f}  ·  Máx: {df_full["score_fidc"].max():.0f}',
                            ACCENT), width=3),
                        *[dbc.Col(kpi(
                            rating.split('—')[0].strip(),
                            f'{int((df_full["rating_carteira"]==rating).sum()):,}',
                            f'{(df_full["rating_carteira"]==rating).mean()*100:.1f}%',
                            RATING_COLOR.get(rating, WHITE)), width=1)
                          for rating in ['A — Excelente','B — Bom','C — Risco Moderado','D — Risco Elevado','E — Alto Risco']
                          if rating in df_full['rating_carteira'].values],
                    ], className='g-2'),
                ]),

                # ════════════════════════════════════════════════════════
                # BLOCO 2 — Score ML Auxiliar (contínuo)
                # ════════════════════════════════════════════════════════
                *([card([
                    html.Div('2 — Score ML Auxiliar (indicador contínuo de risco)',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': '#a78bfa', 'marginBottom': '16px',
                                    'borderLeft': '4px solid #a78bfa',
                                    'paddingLeft': '10px'}),
                    html.Div(f'Modelo: {best_name.split()[0]}  ·  AUC {ml[best_name]["auc"]:.4f}  ·  '
                             f'Quanto maior, menor o risco segundo o comportamento dos boletos',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi(
                            'Score ML Médio',
                            f'{df_full["prob_ml_bom"].mean():.1f}%',
                            f'Mediana: {df_full["prob_ml_bom"].median():.1f}%',
                            '#a78bfa'), width=3),
                        dbc.Col(kpi(
                            'Score ML Alto (≥ 40%)',
                            f'{int((df_full["prob_ml_bom"] >= 40).sum()):,}',
                            f'{(df_full["prob_ml_bom"] >= 40).mean()*100:.1f}% — sinal positivo',
                            ACCENT2), width=3),
                        dbc.Col(kpi(
                            'Score ML Baixo (< 20%)',
                            f'{int((df_full["prob_ml_bom"] < 20).sum()):,}',
                            f'{(df_full["prob_ml_bom"] < 20).mean()*100:.1f}% — sinal de atenção',
                            WARN), width=3),
                        dbc.Col(kpi(
                            'AUC-ROC',
                            f'{ml[best_name]["auc"]:.4f}',
                            f'{best_name.split()[0]} — poder preditivo',
                            '#a78bfa'), width=3),
                    ], className='g-3'),
                    html.Div([
                        html.Span('ℹ️  ', style={'fontSize': '14px'}),
                        html.Span(
                            'O Score ML é um indicador relativo — não classifica bom/ruim diretamente. '
                            'Scores mais altos indicam padrões de pagamento mais saudáveis nos boletos. '
                            'Use em conjunto com o Score FIDC para uma visão complementar.',
                            style={'fontSize': '11px', 'color': MUTED}),
                    ], style={'marginTop': '12px', 'padding': '8px 12px',
                              'background': '#0a1e30', 'borderRadius': '6px',
                              'border': f'1px solid {BORDER}'}),
                ])] if 'prob_ml_bom' in df_full.columns else []),

                # ════════════════════════════════════════════════════════
                # BLOCO 3 — Divergência Score vs ML
                # ════════════════════════════════════════════════════════
                *([card([
                    html.Div('3 — Sinais de Divergência: Score FIDC × Score ML',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': WARN, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {WARN}',
                                    'paddingLeft': '10px'}),
                    html.Div('CNPJs onde os dois sistemas apontam direções opostas — '
                             'requerem análise adicional do analista antes da aquisição',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi(
                            'Sinais de Atenção',
                            f'{int(df_full["alerta_divergencia"].sum()):,}',
                            f'{df_full["alerta_divergencia"].mean()*100:.1f}% da carteira',
                            WARN), width=3),
                        dbc.Col(kpi(
                            'Score Alto + ML Baixo',
                            f'{int(((df_full["score_fidc"]>=800)&(df_full["prob_ml_bom"]<30)).sum()):,}',
                            'Score ≥ 800 · ML < 30% — investigar',
                            WARN), width=3),
                        dbc.Col(kpi(
                            'Score Baixo + ML Alto',
                            f'{int(((df_full["score_fidc"]<300)&(df_full["prob_ml_bom"]>=70)).sum()):,}',
                            'Score < 300 · ML ≥ 70% — revisar',
                            ACCENT2), width=3),
                        dbc.Col(kpi(
                            'Sem Divergência',
                            f'{int((df_full["alerta_divergencia"]==0).sum()):,}',
                            f'{(df_full["alerta_divergencia"]==0).mean()*100:.1f}% — sinais alinhados',
                            ACCENT), width=3),
                    ], className='g-3'),
                    html.Div([
                        html.Span('ℹ️  ', style={'fontSize': '14px'}),
                        html.Span(
                            'Divergência esperada: Score FIDC e Score ML medem dimensões distintas. '
                            'O Score FIDC usa regras de negócio (liquidez e materialidade). '
                            'O Score ML usa padrões de boletos. A divergência é um sinal de investigação, não de erro.',
                            style={'fontSize': '11px', 'color': MUTED}),
                    ], style={'marginTop': '12px', 'padding': '8px 12px',
                              'background': '#0a1e30', 'borderRadius': '6px',
                              'border': f'1px solid {BORDER}'}),
                ])] if 'alerta_divergencia' in df_full.columns else []),

                # ── Recomendação textual ──────────────────────────────────
                html.Div(style={
                    'background': '#0a1e30', 'border': f'1px solid {BORDER}',
                    'borderLeft': f'4px solid {ACCENT2}',
                    'borderRadius': '10px', 'padding': '16px 20px', 'marginTop': '8px',
                }, children=[
                    html.Div('📌 Recomendação Automática',
                             style={'fontSize': '13px', 'fontWeight': '700',
                                    'color': ACCENT2, 'marginBottom': '8px'}),
                    html.Div([
                        html.Span(f"A carteira contém {len(df_full):,} CNPJs com score médio de "
                                  f"{df_full['score_fidc'].mean():.0f}/1000. "),
                        html.Span(
                            f"{int(df_full['target'].sum()):,} CNPJs "
                            f"({df_full['target'].mean()*100:.1f}%) são classificados como boas carteiras pelo target. "),
                        html.Span(
                            f"{int((df_full['rating_carteira'].isin(['A — Excelente','B — Bom'])).sum()):,} CNPJs "
                            f"possuem rating A ou B pelo score de negócio. "),
                        *([html.Span(
                            f"{int(df_full['alerta_divergencia'].sum()):,} CNPJs apresentam sinais divergentes entre "
                            "score de negócio e score ML — recomenda-se análise individual antes da aquisição."
                        )] if 'alerta_divergencia' in df_full.columns else []),
                    ], style={'fontSize': '13px', 'color': WHITE, 'lineHeight': '1.8'}),
                ]),

              ])]),

            dcc.Tab(label='📊 EDA', value='tab-eda', style=tab_style, selected_style=tab_sel,
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('2–3. Análise Exploratória', 'Distribuições, nulos, volumes temporais'),
                card([html.Div('Nulos por Coluna (%)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_nulos(df_full))]),
                card([html.Div('Distribuição dos Scores de Crédito', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_scores(df_full))]),
                card([html.Div('Índices de Liquidez', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_liquidez(df_full))]),
                card([html.Div('Atraso Médio por UF (Top 12)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_atraso_uf(df_full))]),
                card([html.Div('Volume de Boletos por Mês de Emissão', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_volume_mensal(df_bol))]),
                card([html.Div('Tipos de Baixa dos Boletos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_tipos_baixa(df_bol))]),
                card([html.Div('Distribuição do Atraso Real', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_atraso_real(df_bol))]),

                # ── Análise setorial por CNAE ─────────────────────────────
                html.Hr(style={'borderColor': BORDER, 'margin': '24px 0 16px'}),
                section_title('Análise Setorial por CNAE',
                    'Denominação oficial CNAE 2.0 · Top 7 setores da carteira'),
                card([
                    html.Div('Top 7 setores por volume de CNPJs na carteira',
                             style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                    G(ch.fig_cnpjs_por_cnae(perfil_cnae, top_n=7)),
                ]),
                card([
                    html.Div('Score médio vs volume de CNPJs por setor  (tamanho da bolha = % suspeitos)',
                             style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                    G(ch.fig_score_por_cnae(perfil_cnae, top_n=7)),
                ]),

              ])]),

            # ── ANÁLISE DE FRAUDE ─────────────────────────────────────────
            dcc.Tab(label='🚨 Fraude', value='tab-fraude', style=tab_style, selected_style={**tab_sel, 'background': WARN, 'color': NAVY},
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('2B. Detecção de Boletos Duplicados',
                    f'Threshold: ≥ {fraude_stats["pct_dup_thresh"]*100:.0f}% duplicados  OU  ≥ {fraude_stats["n_emit_thresh"]} emitentes distintos'),

                # KPIs de fraude
                dbc.Row([
                    dbc.Col(kpi('Total Duplicatas',    f'{fraude_stats["total_duplicatas"]:,}',
                                f'{fraude_stats["pct_duplicatas"]:.1f}% da base', WARN), width=3),
                    dbc.Col(kpi('Dup. por ID',         f'{fraude_stats["total_dup_id"]:,}',
                                'mesmo id_boleto repetido', WARN), width=3),
                    dbc.Col(kpi('Dup. por Conteúdo',   f'{fraude_stats["total_dup_conteudo"]:,}',
                                'valor+venc+pagador+benef', WARN), width=3),
                    dbc.Col(kpi('CNPJs Suspeitos',      f'{fraude_stats["cnpjs_suspeitos"]:,}',
                                f'de {fraude_stats["total_cnpjs_com_boleto"]:,} c/ boletos', WARN), width=3),
                ], className='g-3', style={'marginBottom': '20px'}),

                dbc.Row([
                    dbc.Col([
                        card([html.Div('Tipos de duplicata detectados', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                              G(ch.fig_duplicatas_tipo(fraude_stats))]),
                    ], width=5),
                    dbc.Col([
                        card([html.Div('Concentração de emitentes por pagador', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                              G(ch.fig_emitentes_por_pagador(fraude_cnpj))]),
                    ], width=7),
                ], className='g-3'),

                card([html.Div('Top CNPJs por % de boletos duplicados  (laranja = flag de risco ativado)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                      G(ch.fig_top_cnpjs_suspeitos(fraude_cnpj))]),

                card([html.Div('Score FIDC vs % duplicados — CNPJs suspeitos em destaque', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                      G(ch.fig_scatter_fraude(df_full))]),

                card([html.Div('Grupos de boletos duplicados com maior incidência', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                      G(ch.fig_resumo_duplicatas(resumo_dup))]),

                # Tabela de CNPJs suspeitos
                card([
                    html.Div('CNPJs com flag de risco de fraude', style={'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '12px', 'color': WARN}),
                    *([] if fraude_cnpj[fraude_cnpj['flag_risco_fraude'] == 1].empty else [
                        dash_table.DataTable(
                            data=fraude_cnpj[fraude_cnpj['flag_risco_fraude'] == 1]
                                .sort_values('bol_pct_duplicado', ascending=False)
                                [['id_pagador', 'bol_qtd_total', 'bol_qtd_dup_total',
                                  'bol_pct_duplicado', 'bol_n_emitentes', 'motivo_alerta']]
                                .rename(columns={
                                    'id_pagador': 'CNPJ', 'bol_qtd_total': 'Total Bol.',
                                    'bol_qtd_dup_total': 'Duplicatas', 'bol_pct_duplicado': '% Dup.',
                                    'bol_n_emitentes': 'Emitentes', 'motivo_alerta': 'Motivo'})
                                .assign(**{'% Dup.': lambda d: (d['% Dup.'] * 100).round(1).astype(str) + '%'})
                                .to_dict('records'),
                            columns=[{'name': c, 'id': c} for c in
                                     ['CNPJ', 'Total Bol.', 'Duplicatas', '% Dup.', 'Emitentes', 'Motivo']],
                            page_size=15, sort_action='native',
                            style_table={'overflowX': 'auto'},
                            style_cell={'backgroundColor': CARD_BG, 'color': WHITE,
                                        'border': f'1px solid {BORDER}', 'fontFamily': 'Space Grotesk',
                                        'fontSize': '12px', 'padding': '8px 12px', 'textAlign': 'left'},
                            style_header={'backgroundColor': BLUE, 'fontWeight': '700',
                                          'color': WARN, 'border': f'1px solid {BORDER}', 'fontSize': '12px'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{Motivo} contains "Duplicatas" && {Motivo} contains "Emitentes"'},
                                 'backgroundColor': '#2d0a0a', 'color': '#ff6b6b', 'fontWeight': '600'},
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#0d1b2a'},
                            ],
                        )
                    ]),
                    html.Div('Nenhum CNPJ suspeito encontrado com os thresholds atuais.',
                             style={'color': ACCENT2, 'fontSize': '13px', 'padding': '12px 0'})
                    if fraude_cnpj[fraude_cnpj['flag_risco_fraude'] == 1].empty else html.Div(),
                ]),
              ])]),


            # ── TARGET & CORRELAÇÃO ───────────────────────────────────────
            dcc.Tab(label='🎯 Target & Correlação', style=tab_style, selected_style=tab_sel,
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('5–6. Target e Correlação',
                    f'Regra 2 de 3: score_quantidade ≥ {int(mat_thresh)} · score_materialidade ≥ {int(mat_thresh)} · liquidez_3m ≥ {liq_thresh}'),
                card([html.Div('Distribuição do Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_target_pizza(df_full))]),
                card([html.Div('Score Materialidade v2 por Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_boxplot_target(df_full, 'score_materialidade_v2', 'Score Materialidade v2'))]),
                card([html.Div('Liquidez Sacado (1m) por Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_boxplot_target(df_full, 'sacado_indice_liquidez_1m', 'Índice de Liquidez (1m)'))]),
                card([html.Div('Correlação com o Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_correlacao_target(corr_mat))]),
                card([html.Div('Matriz de Correlação Completa (Pearson)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_heatmap_correlacao(corr_mat))]),
              ])]),

            # ── MODELAGEM ─────────────────────────────────────────────────
            dcc.Tab(label='🤖 Modelagem', value='tab-ml', style=tab_style, selected_style=tab_sel,
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('7–8. Modelagem Preditiva', 'Pipeline: Imputer → Scaler → Modelo · Cross-Validation k=5'),
                card([
                    html.Div('Comparação de Modelos — Métricas', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '12px'}),
                    dash_table.DataTable(
                        data=[{'Modelo': n, 'CV AUC (treino)': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
                               'Test AUC': f"{r['auc']:.4f}", 'Test AP': f"{r['ap']:.4f}"}
                              for n, r in ml.items()],
                        columns=[{'name': c, 'id': c} for c in ['Modelo', 'CV AUC (treino)', 'Test AUC', 'Test AP']],
                        style_table={'overflowX': 'auto'},
                        style_cell={'backgroundColor': CARD_BG, 'color': WHITE, 'border': f'1px solid {BORDER}',
                                    'fontFamily': 'Space Grotesk', 'fontSize': '13px', 'padding': '12px 16px', 'textAlign': 'left'},
                        style_header={'backgroundColor': BLUE, 'fontWeight': '700', 'color': ACCENT,
                                      'border': f'1px solid {BORDER}', 'fontSize': '13px'},
                        style_data_conditional=[{'if': {'filter_query': f'{{Modelo}} = "{best_name}"'},
                                                 'backgroundColor': '#0d2b1a', 'color': ACCENT2, 'fontWeight': '600'}],
                    ),
                ]),
                card([html.Div('AUC-ROC por Fold — Cross-Validation (k=5)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_cv_folds(ml))]),
                card([html.Div('CV AUC vs Test AUC — comparação entre modelos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_cv_vs_test(ml))]),
                card([html.Div(f'Importância das Features — {best_name}', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_feature_importance(feat_imp, best_name))]),
                html.Hr(style={'borderColor': BORDER, 'margin': '20px 0'}),
                section_title('Avaliação dos Modelos', 'Curvas ROC e Matrizes de Confusão'),
                card([html.Div('Curvas ROC — Comparação entre Modelos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_roc(ml))]),
                card([html.Div('Matrizes de Confusão — por Modelo', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_confusion_matrix(ml))]),
              ])]),


            # ── SCORE FINAL ───────────────────────────────────────────────
            dcc.Tab(label='🥇 Score Final', value='tab-score', style=tab_style, selected_style=tab_sel,
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('11. Score Final de Qualidade',
                    'Fórmula composta: Liquidez (35%) + Materialidade (25%) + Quantidade (15%) + Liq.3m (10%) + Atraso (8%) + Inadimplência (7%)'),
                card([html.Div('Distribuição do Score FIDC', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_score_histograma(df_full))]),
                card([html.Div('CNPJs por Rating de Carteira', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_rating_barras(df_full))]),
                card([html.Div('Score FIDC Médio por UF', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_score_por_uf(df_full))]),
                card([html.Div('Liquidez Sacado (1m) vs Score FIDC', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_scatter(df_full, 'sacado_indice_liquidez_1m', 'Liquidez Sacado (1m)'))]),
                card([html.Div('Score Materialidade v2 vs Score FIDC', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_scatter(df_full, 'score_materialidade_v2', 'Score Materialidade v2'))]),
                card([html.Div('Atraso Médio (dias) vs Score FIDC', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_scatter(df_full, 'media_atraso_dias', 'Média de Atraso (dias)'))]),

                # ── Produto 2: ML como segunda opinião ────────────────────
                *([
                    html.Hr(style={'borderColor': BORDER, 'margin': '24px 0 16px'}),
                    section_title('Produto 2 — Segunda Opinião do Modelo ML',
                        f'Score FIDC = regra de negócio Núclea  ·  Prob. ML = padrões aprendidos nos boletos ({best_name.split()[0]})'),
                    dbc.Row([
                        dbc.Col(kpi('Divergências',
                            f'{int(df_full["alerta_divergencia"].sum()):,}',
                            f'{df_full["alerta_divergencia"].mean()*100:.1f}% da carteira', WARN), width=3),
                        dbc.Col(kpi('Prob. ML Média',
                            f'{df_full["prob_ml_bom"].mean():.1f}%',
                            'carteira geral', ACCENT), width=3),
                        dbc.Col(kpi('Score Alto + ML Pessimista',
                            f'{int(((df_full["score_fidc"]>=700)&(df_full["prob_ml_bom"]<40)).sum()):,}',
                            'score ≥ 800, ML < 30%', WARN), width=3),
                        dbc.Col(kpi('Score Baixo + ML Otimista',
                            f'{int(((df_full["score_fidc"]<400)&(df_full["prob_ml_bom"]>=60)).sum()):,}',
                            'score < 300, ML ≥ 70%', ACCENT2), width=3),
                    ], className='g-3', style={'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col([card([
                            html.Div('Score FIDC vs Probabilidade ML — quadrantes de divergência',
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '4px'}),
                            html.Div('CNPJs em laranja merecem atenção especial — os dois sistemas discordam.',
                                     style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_divergencia_ml(df_full)),
                        ])], width=8),
                        dbc.Col([card([
                            html.Div('Distribuição da Prob. ML por Rating',
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_prob_ml_hist(df_full)),
                        ])], width=4),
                    ], className='g-3'),
                ] if 'prob_ml_bom' in df_full.columns else []),

                # Ranking com filtros
                card([
                    html.Div('🏆 Ranking de CNPJs', style={'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col([dcc.Input(id='rank-cnpj', placeholder='Buscar CNPJ...', debounce=True,
                                           style={'width': '100%', 'background': '#0d1b2a', 'border': f'1px solid {BORDER}',
                                                  'borderRadius': '6px', 'padding': '9px 12px', 'color': WHITE,
                                                  'fontFamily': "'JetBrains Mono',monospace", 'fontSize': '12px'})], width=3),
                        dbc.Col([dcc.Dropdown(id='rank-uf',
                                              options=[{'label': u, 'value': u} for u in sorted(df_full['uf'].dropna().unique())],
                                              placeholder='UF', multi=True, className='dark-dropdown')], width=2),
                        dbc.Col([dcc.Dropdown(id='rank-cnae',
                                              options=[{'label': str(c), 'value': str(c)} for c in sorted(df_full['cd_cnae_prin'].dropna().astype(str).unique())],
                                              placeholder='CNAE', multi=True, className='dark-dropdown')], width=3),
                        dbc.Col([dcc.Dropdown(id='rank-rating',
                                              options=[{'label': r, 'value': r} for r in list(RATING_COLOR.keys())],
                                              placeholder='Rating', multi=True, className='dark-dropdown')], width=2),
                        dbc.Col([dcc.RangeSlider(id='rank-score', min=0, max=1000, step=50, value=[0, 1000],
                                                  marks={0: '0', 500: '500', 1000: '1000'},
                                                  tooltip={'always_visible': False})], width=2),
                    ], className='g-2', style={'marginBottom': '16px'}),

                    html.Div(id='rank-table-container',
                             children=[build_rank_table(df_full)]),

                    html.Br(),
                    html.A('⬇️ Download CSV completo', id='download-link',
                           download='resultado_fidc.csv',
                           href='data:text/csv;charset=utf-8,' +
                                df_full[['id_cnpj', 'uf', 'cd_cnae_prin', 'score_fidc', 'rating_carteira', 'target']]
                                .sort_values('score_fidc', ascending=False).to_csv(index=False),
                           target='_blank',
                           style={'color': ACCENT, 'fontSize': '13px', 'textDecoration': 'none',
                                  'border': f'1px solid {ACCENT}', 'borderRadius': '6px',
                                  'padding': '9px 20px', 'display': 'inline-block', 'marginTop': '8px'}),
                ]),
              ])]),

            # ── CONTEXTO MACROECONÔMICO ───────────────────────────────────
            dcc.Tab(label='🌐 Macro', value='tab-macro', style=tab_style,
                    selected_style={'color': NAVY, 'background': '#2563EB', 'fontWeight': '700'},
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('Contexto Macroeconômico — Camada 1',
                    'Fontes: BCB SGS · IBGE SIDRA · Dados buscados automaticamente no upload'),
                html.Div(id='macro-content',
                    children=html.Div(
                        '🌐 Faça o upload dos dois arquivos CSV para carregar os dados macroeconômicos.',
                        style={'color': MUTED, 'fontSize': '14px',
                               'textAlign': 'center', 'padding': '40px'}
                    )
                ),
              ])]),
        ]
    )

    return html.Div([kpi_row, tabs])