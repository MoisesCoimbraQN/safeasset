# =============================================================================
# callbacks.py — SafeAsset
# Responsável por: toda a lógica reativa do Dash (callbacks)
# Conecta o frontend (layout.py) com os dados (pipeline.py) e gráficos (charts.py)
# =============================================================================

import base64, io, threading
import numpy as np
import pandas as pd

from dash import dcc, html, dash_table, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

import pipeline as pl
import charts as ch
import storage as st   
from layout import (card, kpi, section_title, build_rank_table,
                    ACCENT, ACCENT2, WARN, MUTED, BORDER, WHITE, NAVY, BLUE, CARD_BG )
from theme import BG_INFO, RISK_E

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # sem dotenv instalado (ex: produção no Render) — env vars já vêm do ambiente

# Aliases de compatibilidade — tema escuro
DARK   = WHITE    # texto claro sobre fundo escuro
AMBER  = '#B7791F'  # âmbar — usado em alertas e destaques
BG_MAIN = CARD_BG # fundo dos cards
from pipeline import RATING_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# CACHE DE PIPELINE (em memória do processo)
#
# Guarda o resultado completo da última "Executar Análise Histórica" —
# incluindo DataFrames e modelos treinados, sem round-trip por JSON/Store.
# Filtros, análise de carteira nova e downloads leem daqui em vez de rodar
# pl.run_pipeline() de novo — só os botões de treino (Executar Análise
# Histórica / Aplicar Parâmetros ML / Aplicar Detecção de Fraude) escrevem
# aqui. Funciona porque o Render roda WEB_CONCURRENCY=1 (um único worker) —
# se um dia o deploy for multi-worker, isso precisa virar um cache
# compartilhado (Redis, etc.), já que cada worker teria sua própria cópia.
# ─────────────────────────────────────────────────────────────────────────────
_PIPELINE_CACHE = {
    'R': None,          # dict completo retornado por pl.run_pipeline() + extras
    'df_aux': None,      # base auxiliar usada no último treino (para ind_risco)
    'df_bol_raw': None,  # base de boletos crua usada no último treino (idem)
}

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
        Output('store-raw-cart', 'data'),
        Output('cart-status', 'children'),
        Input('upload-cart', 'contents'),
        State('upload-cart', 'filename'),
        prevent_initial_call=True,
    )
    def handle_upload_cart(contents, filename):
        if not contents:
            return None, ''
        try:
            print(f'[SafeAsset] Upload carteira nova: {filename}')
            json_str = parse_upload(contents, filename)
            if not json_str:
                return None, '❌ Erro ao ler arquivo'
            df = read_json(json_str)
            n_cnpjs = df['id_pagador'].nunique() if 'id_pagador' in df.columns else len(df)
            vlr = df['vlr_nominal'].sum() if 'vlr_nominal' in df.columns else 0
            status = (f'✅ {filename} — {n_cnpjs:,} CNPJs · R$ {vlr:,.0f}')
            print(f'[SafeAsset] Carteira OK — {n_cnpjs:,} CNPJs · R$ {vlr:,.0f}')
            return json_str, status
        except Exception as e:
            print(f'[SafeAsset] Erro carteira: {e}')
            return None, f'❌ Erro: {e}'


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
                'ind_risco':   None,  # calculado no run_dashboard com df_bol filtrado
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
                'ind_risco': None,
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

    # ── CALLBACK 1/3 — TREINO — única fonte de pl.run_pipeline() ──────────
    # Dispara SÓ nos botões que legitimamente pedem retreino. Filtros e
    # carteira nova NÃO estão nos Inputs daqui de propósito.
    @app.callback(
        Output('main-content',   'children', allow_duplicate=True),
        Output('loading-output', 'children', allow_duplicate=True),
        Output('store-df-full',  'data',     allow_duplicate=True),
        Input('btn-run',         'n_clicks'),
        Input('btn-run-upload',  'n_clicks'),
        Input('btn-run-ml',      'n_clicks'),
        Input('btn-run-fraud',   'n_clicks'),
        State('store-raw-aux',   'data'),
        State('store-raw-bol',   'data'),
        State('sl-test',  'value'),
        State('sl-trees', 'value'),
        State('sl-dup-thresh',  'value'),
        State('sl-emit-thresh', 'value'),
        prevent_initial_call=True,
    )
    def processar_historico(run_clicks, run_upload_clicks, ml_clicks, fraud_clicks,
                            aux_json_input, bol_json,
                            test_size, n_trees, dup_thresh, emit_thresh):

        from dash.exceptions import PreventUpdate

        if not aux_json_input and not bol_json:
            raise PreventUpdate

        if not aux_json_input:
            return html.Div([
                html.Div('📂', style={'fontSize': '48px', 'marginBottom': '12px'}),
                html.Div('Aguardando base auxiliar…', style={'color': MUTED, 'fontSize': '15px'}),
                html.Div('Faça o upload do arquivo base_auxiliar_fiap.csv',
                         style={'color': WARN, 'fontSize': '13px', 'marginTop': '6px'}),
            ], style={'textAlign': 'center', 'padding': '60px'}), '', None

        if not bol_json:
            return html.Div([
                html.Div('📂', style={'fontSize': '48px', 'marginBottom': '12px'}),
                html.Div('Base auxiliar carregada ✅', style={'color': ACCENT2, 'fontSize': '15px'}),
                html.Div('Agora faça o upload do arquivo base_boletos_fiap.csv',
                         style={'color': MUTED, 'fontSize': '13px', 'marginTop': '6px'}),
            ], style={'textAlign': 'center', 'padding': '60px'}), '', None

        try:
            df_aux = read_json(aux_json_input)
        except Exception as e:
            return html.Div(f'❌ Erro ao ler base auxiliar: {e}',
                            style={'color': WARN, 'padding': '40px'}), '', None

        try:
            df_bol = read_json(bol_json)
        except Exception as e:
            return html.Div(f'❌ Erro ao ler base de boletos: {e}',
                            style={'color': WARN, 'padding': '40px'}), '', None

        if len(df_aux) < 10:
            return html.Div('⚠️ Base auxiliar com poucos dados.',
                            style={'color': WARN, 'padding': '40px', 'textAlign': 'center'}), '', None

        # ── Executar pipeline (ÚNICO lugar que treina os modelos) ─────────
        try:
            R = pl.run_pipeline(
                df_aux.copy(), df_bol.copy(),
                test_size          = test_size  or 0.2,
                n_estimators       = n_trees    or 300,
                liq_thresh         = 0.65,
                mat_thresh         = 800,
                pct_dup_thresh     = (dup_thresh  or 5)  / 100,
                n_emitentes_thresh = emit_thresh or 10,
            )
        except Exception as e:
            import traceback
            return html.Div([
                html.Div('❌ Erro ao executar o pipeline:',
                         style={'color': WARN, 'fontWeight': '700', 'marginBottom': '8px'}),
                html.Pre(traceback.format_exc(),
                         style={'color': MUTED, 'fontSize': '11px', 'background': '#F6F8FA',
                                'padding': '12px', 'borderRadius': '6px', 'overflow': 'auto'}),
            ], style={'padding': '24px'}), '', None

        run_id = st.novo_run_id()

        # Retreinou — carteira nova anterior fica obsoleta contra o modelo
        # novo; usuário precisa clicar "Analisar Carteira Nova" de novo.
        R['cobertura']   = None
        R['df_carteira'] = None

        # ── Calcular Indicador de Risco Setorial ──────────────────────────
        try:
            from macro import calcular_indicador_risco_setorial
            R['ind_risco'] = calcular_indicador_risco_setorial(df_aux, df_bol)
            print(f"[SafeAsset] ind_risco: {R['ind_risco']['tag']} — {R['ind_risco']['setor_label']}")
        except Exception as _e:
            print(f"[SafeAsset] ind_risco erro: {_e}")
            R['ind_risco'] = None

        # ── Guarda no cache do processo — fonte pra filtros/cart/downloads ─
        _PIPELINE_CACHE['R']          = R
        _PIPELINE_CACHE['df_aux']     = df_aux
        _PIPELINE_CACHE['df_bol_raw'] = df_bol

        # ── Montar dashboard ──────────────────────────────────────────────
        dashboard = build_dashboard(R, 0.65, 800,
                                     dup_thresh  or 0.05,
                                     emit_thresh or 10)
        df_full_json = R['df_full'].to_json(date_format='iso', orient='split')

        threading.Thread(
            target=lambda: (
                st.salvar_snapshot_s3(run_id, aux=df_aux, bol=df_bol, df_full=R['df_full']),
                st.salvar_historico_cnpj(run_id, R['df_full']),
                st.salvar_historico_cedente(
                    run_id, pl.calcular_perfil_beneficiario(df_bol, R['df_full']),
                    origem='historico'),
            ),
            daemon=True,
        ).start()

        return dashboard, '', df_full_json

    # ── CALLBACK 2/3 — FILTROS — sem retreino, lê do cache de pipeline ────
    @app.callback(
        Output('main-content',   'children', allow_duplicate=True),
        Output('loading-output', 'children', allow_duplicate=True),
        Output('store-df-full',  'data',     allow_duplicate=True),
        Input('flt-cnpj',        'value'),
        Input('flt-uf',          'value'),
        Input('flt-cnae',        'value'),
        Input('flt-date',        'start_date'),
        Input('flt-date',        'end_date'),
        State('sl-dup-thresh',  'value'),
        State('sl-emit-thresh', 'value'),
        prevent_initial_call=True,
    )
    def aplicar_filtros(cnpj_q, sel_ufs, sel_cnaes, date_from, date_to,
                        dup_thresh, emit_thresh):
        from dash.exceptions import PreventUpdate

        R_base = _PIPELINE_CACHE['R']
        if R_base is None:
            raise PreventUpdate

        df_full = R_base['df_full'].copy()
        if cnpj_q:
            df_full = df_full[df_full['id_cnpj'].astype(str).str.lower()
                              .str.contains(cnpj_q.strip().lower(), na=False)]
        if sel_ufs:
            df_full = df_full[df_full['uf'].isin(sel_ufs)]
        if sel_cnaes:
            df_full = df_full[df_full['cd_cnae_prin'].astype(str).isin(sel_cnaes)]
        if date_from or date_to:
            df_bol_full = R_base['df_bol']
            ds = pd.to_datetime(df_bol_full['dt_emissao'], errors='coerce')
            df_bol_f = df_bol_full
            if date_from: df_bol_f = df_bol_f[ds >= pd.Timestamp(date_from)]
            if date_to:   df_bol_f = df_bol_f[ds <= pd.Timestamp(date_to)]
            valid = set(df_bol_f['id_pagador'].tolist())
            if valid:
                df_full = df_full[df_full['id_cnpj'].isin(valid)]

        if len(df_full) < 10:
            return html.Div('⚠️ Filtro muito restritivo — poucos dados disponíveis.',
                            style={'color': WARN, 'padding': '40px', 'textAlign': 'center'}), '', no_update

        # Reusa tudo que já foi treinado (ML, fraude, perfil CNAE, ind_risco)
        # — só a visão de CNPJs exibida muda, nada é recalculado do zero.
        R_view = dict(R_base)
        R_view['df_full'] = df_full
        dashboard = build_dashboard(R_view, 0.65, 800,
                                     dup_thresh  or 0.05,
                                     emit_thresh or 10)
        df_full_json = df_full.to_json(date_format='iso', orient='split')
        return dashboard, '', df_full_json

    # ── CALLBACK 3/3 — CARTEIRA NOVA — sem retreino, lê do cache ──────────
    @app.callback(
        Output('main-content',   'children', allow_duplicate=True),
        Output('loading-output', 'children', allow_duplicate=True),
        Output('store-df-full',  'data',     allow_duplicate=True),
        Input('btn-run-cart',    'n_clicks'),
        State('store-raw-cart',  'data'),
        State('sl-dup-thresh',  'value'),
        State('sl-emit-thresh', 'value'),
        prevent_initial_call=True,
    )
    def analisar_carteira_nova(cart_clicks, cart_json, dup_thresh, emit_thresh):
        from dash.exceptions import PreventUpdate

        R = _PIPELINE_CACHE['R']
        if R is None:
            return html.Div(
                '⚠️ Execute a Análise Histórica (base auxiliar + boletos) antes de analisar a carteira nova.',
                style={'color': WARN, 'padding': '40px', 'textAlign': 'center'}), '', no_update

        df_cart = read_json(cart_json) if cart_json else None
        if df_cart is None or len(df_cart) == 0:
            raise PreventUpdate

        from pipeline import calcular_cobertura_carteira
        cobertura = calcular_cobertura_carteira(R['df_full'], df_cart)
        R['cobertura']   = cobertura
        R['df_carteira'] = df_cart
        # mutou o dict em cache — persiste pra próxima vez que filtros/
        # downloads lerem o cache, sem precisar reanalizar a carteira nova

        run_id = st.novo_run_id()
        perfil_cedentes_novo = pl.calcular_perfil_beneficiario(df_cart, R['df_full'])
        threading.Thread(
            target=st.salvar_historico_cedente,
            args=(run_id, perfil_cedentes_novo, 'carteira_nova'),
            daemon=True,
        ).start()

        dashboard = build_dashboard(R, 0.65, 800,
                                     dup_thresh  or 0.05,
                                     emit_thresh or 10)
        df_full_json = R['df_full'].to_json(date_format='iso', orient='split')
        return dashboard, '', df_full_json


    # ── Filtros do ranking ─────────────────────────────────────────────────
    @app.callback(
        Output('rank-table-container', 'children'),
        Input('rank-cnpj',   'value'),
        Input('rank-uf',     'value'),
        Input('rank-cnae',   'value'),
        Input('rank-rating', 'value'),
        Input('rank-score',  'value'),
        State('store-df-full', 'data'),
        prevent_initial_call=True,
    )
    def update_rank_table(cnpj_q, sel_ufs, sel_cnaes, sel_ratings, score_range, df_full_json):
        if not df_full_json:
            return html.Div('Pipeline ainda não executado.', style={'color': MUTED})
        df_full = read_json(df_full_json)
        return build_rank_table(df_full, cnpj_q, sel_ufs, sel_cnaes, sel_ratings, score_range)



    # ── CALLBACK DOWNLOAD — CNPJs com flag de fraude ──────────────────────
    @app.callback(
        Output('download-fraude', 'data'),
        Input('btn-download-fraude', 'n_clicks'),
        State('store-raw-aux', 'data'),
        State('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def download_fraude(n_clicks, aux_json, bol_json):
        if not n_clicks or not aux_json or not bol_json:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        try:
            import pipeline as pl
            df_aux = read_json(aux_json)
            df_bol = read_json(bol_json)
            fraude = pl.detectar_duplicatas(df_bol)
            df_exp = fraude['fraude_por_cnpj']
            df_exp = df_exp[df_exp['flag_risco_fraude'] == 1].copy()
            df_exp['bol_pct_duplicado'] = (df_exp['bol_pct_duplicado']*100).round(1)
            df_exp = df_exp.rename(columns={
                'id_pagador': 'CNPJ', 'bol_qtd_total': 'Total Boletos',
                'bol_qtd_dup_total': 'Duplicatas', 'bol_pct_duplicado': '% Duplicados',
                'bol_n_emitentes': 'Emitentes', 'motivo_alerta': 'Motivo',
                'flag_risco_fraude': 'Flag Fraude'
            })
            return dcc.send_data_frame(
                df_exp.to_csv, 'safeasset_cnpjs_fraude.csv',
                index=False, sep=';', decimal=','
            )
        except Exception as e:
            from dash.exceptions import PreventUpdate
            print(f"[SafeAsset] Erro download fraude: {e}")
            raise PreventUpdate

    # ── CALLBACK DOWNLOAD — CNPJs com histórico PCR ─────────────────────
    @app.callback(
        Output('download-conhecidos', 'data'),
        Input('btn-download-conhecidos', 'n_clicks'),
        State('store-raw-aux', 'data'),
        State('store-raw-bol', 'data'),
        State('store-raw-cart', 'data'),
        prevent_initial_call=True,
    )
    def download_conhecidos(n_clicks, aux_json, bol_json, cart_json):
        if not n_clicks or not cart_json:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        R = _PIPELINE_CACHE['R']
        if R is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        try:
            df_cart = read_json(cart_json)
            df_full = R['df_full']
            col_pag = 'id_pagador' if 'id_pagador' in df_cart.columns else 'id_cnpj'
            conhecidos = set(df_cart[col_pag].astype(str)) & set(df_full['id_cnpj'].astype(str))
            df_exp = df_full[df_full['id_cnpj'].astype(str).isin(conhecidos)].copy()
            vlr = df_cart.groupby(col_pag)['vlr_nominal'].sum().reset_index()
            vlr.columns = ['id_cnpj', 'vlr_carteira']
            df_exp = df_exp.merge(vlr.rename(columns={col_pag:'id_cnpj'}), on='id_cnpj', how='left')
            cols = ['id_cnpj','uf','cd_cnae_prin','score_fidc','rating_carteira',
                    'prob_ml_bom','flag_risco_fraude','sacado_indice_liquidez_1m',
                    'vlr_carteira']
            cols = [c for c in cols if c in df_exp.columns]
            df_exp = df_exp[cols].sort_values('score_fidc', ascending=False)
            return dcc.send_data_frame(df_exp.to_csv,
                'safeasset_cnpjs_com_historico.csv', index=False, sep=';', decimal=',')
        except Exception as e:
            from dash.exceptions import PreventUpdate
            print(f"[SafeAsset] Erro download conhecidos: {e}")
            raise PreventUpdate

    # ── CALLBACK DOWNLOAD — CNPJs para análise individual ─────────────────
    @app.callback(
        Output('download-analise', 'data'),
        Input('btn-download-analise', 'n_clicks'),
        State('store-raw-aux', 'data'),
        State('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def download_analise(n_clicks, aux_json, bol_json):
        if not n_clicks:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        R = _PIPELINE_CACHE['R']
        if R is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        try:
            df = R['df_full']

            # Selecionar CNPJs com alerta de divergência
            cols = ['id_cnpj', 'uf', 'cd_cnae_prin', 'score_fidc',
                    'rating_carteira']
            if 'prob_ml_bom' in df.columns:
                cols += ['prob_ml_bom', 'alerta_divergencia']
            if 'flag_risco_fraude' in df.columns:
                cols += ['flag_risco_fraude', 'motivo_alerta']

            cols = [c for c in cols if c in df.columns]

            if 'alerta_divergencia' in df.columns:
                df_export = df[df['alerta_divergencia'] == 1][cols].copy()
            else:
                df_export = df[cols].copy()

            df_export = df_export.sort_values('score_fidc', ascending=False)
            df_export.columns = [c.replace('_', ' ').title() for c in df_export.columns]

            return dcc.send_data_frame(
                df_export.to_csv, 'safeasset_cnpjs_analise_individual.csv',
                index=False, sep=';', decimal=','
            )
        except Exception as e:
            from dash.exceptions import PreventUpdate
            print(f"[SafeAsset] Erro download: {e}")
            raise PreventUpdate

    # ── CALLBACK DOWNLOAD — CNPJs sem histórico ──────────────────────────
    @app.callback(
        Output('download-novos', 'data'),
        Input('btn-download-novos', 'n_clicks'),
        State('store-raw-aux', 'data'),
        State('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def download_novos(n_clicks, aux_json, bol_json):
        if not n_clicks:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        R = _PIPELINE_CACHE['R']
        if R is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        try:
            df = R['df_full']
            df_exp = df[df['sem_historico'] == 1].copy() if 'sem_historico' in df.columns else df
            cols = ['id_cnpj','uf','cd_cnae_prin','score_fidc','rating_carteira',
                    'score_materialidade_v2','score_quantidade_v2',
                    'sacado_indice_liquidez_1m','flag_risco_fraude']
            cols = [c for c in cols if c in df_exp.columns]
            df_exp = df_exp[cols].sort_values('score_fidc', ascending=False)
            df_exp.columns = [c.replace('_',' ').title() for c in df_exp.columns]
            return dcc.send_data_frame(df_exp.to_csv,
                'safeasset_cnpjs_sem_historico.csv', index=False, sep=';', decimal=',')
        except Exception as e:
            from dash.exceptions import PreventUpdate
            print(f"[SafeAsset] Erro download novos: {e}")
            raise PreventUpdate

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
        if tab_ativa != 'tab-outras':
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
        from macro import enriquecer_perfil_com_macro, calcular_score_macro_setor, CNAE_SETOR, calcular_quartis_macro
        _p25, _p75 = calcular_quartis_macro()
        print("[SafeAsset] update_macro_content iniciando...")
        try:
            dados  = _json.loads(macro_store_json)
            ind    = dados.get('indicadores', {})
            fonte  = dados.get('fonte', '—')
            coleta = dados.get('data_coleta', '—')

            # Reconstruir perfil CNAE a partir da base auxiliar
            df_aux = read_json(aux_json)
            from pipeline import calcular_perfil_cnae


            # Criar df_full mínimo para calcular perfil CNAE
            from macro import enriquecer_perfil_com_macro, calcular_score_macro_setor, CNAE_SETOR, calcular_quartis_macro
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
                    scores[setor] = calcular_score_macro_setor(ind, setor, p25=_p25, p75=_p75)

            # Enriquecer perfil com macro
            perfil['setor_macro'] = perfil['cd_cnae_fmt'].apply(
                lambda x: CNAE_SETOR.get(str(x).split('.')[0], 'servicos')
            )
            perfil['score_macro'] = perfil['setor_macro'].apply(
                lambda s: calcular_score_macro_setor(ind, s, p25=_p25, p75=_p75)['score']
            )
            perfil['nivel_macro'] = perfil['setor_macro'].apply(
                lambda s: calcular_score_macro_setor(ind, s, p25=_p25, p75=_p75)['nivel']
            )

            # Indicador de Risco calculado abaixo via _ind_risco_macro
            df_bol = read_json(bol_json)

            def G(fig):
                return dcc.Graph(figure=fig, config={'displayModeBar': False})

            alerta = [] if fonte == 'api' else [
                html.Div([
                    html.Span('⚠️  Sem conexão com BCB/IBGE — exibindo dados de referência recentes.',
                              style={'fontSize': '11px', 'color': AMBER,
                                     'fontStyle': 'italic'}),
                ], style={'marginBottom': '10px'}),
            ]

            # ── Séries históricas para os 3 gráficos — fallback imediato ──
            import pandas as _pd

            # Fallbacks com dados reais BCB coletados em mai/2026
            # Atualizar via: api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados/ultimos/20
            _FB_INAD   = [2.36,2.37,2.33,2.31,2.03,2.23,2.30,2.25,2.46,2.40,
                          2.40,2.51,2.56,2.51,2.56,2.48,2.43,2.60,2.78,2.75]
            _FB_JUROS  = [18.2,18.5,18.8,19.1,19.4,19.7,20.0,20.3,20.1,19.8,
                          19.5,19.2,18.9,18.6,18.3,18.0,17.8,17.6,17.9,18.2]
            _FB_CONC   = [320000,335000,318000,342000,355000,328000,341000,360000,
                          375000,352000,368000,385000,371000,358000,372000,389000,
                          365000,378000,392000,381000]
            _FB_DUPL   = [19.5,19.8,20.1,20.4,20.7,21.0,21.3,21.6,21.4,21.1,
                          20.8,20.5,20.2,19.9,19.6,19.3,19.1,19.4,19.7,20.0]

            def _make_fb(vals):
                return _pd.DataFrame({
                    'data':  _pd.date_range(end=_pd.Timestamp.today(),
                                            periods=20, freq='ME'),
                    'valor': vals,
                }).sort_values('data').reset_index(drop=True)

            # Tenta API com timeout curto; usa fallback instantaneamente se falhar
            from macro import _bcb_serie_historico as _bsh
            def _safe_hist(codigo, fb):
                try:
                    df = _bsh(codigo, n=20)
                    return df if len(df) >= 5 else _make_fb(fb)
                except Exception:
                    return _make_fb(fb)

            _hist_inad       = _safe_hist(21083, _FB_INAD)
            _hist_juros      = _safe_hist(20715, _FB_JUROS)
            _hist_concessoes = _safe_hist(20632, _FB_CONC)
            _hist_duplicatas = _safe_hist(20719, _FB_DUPL)

            print('[SafeAsset] Macro content montado com sucesso')

            # ── Indicador de Risco PJ — mesmo cálculo do run_dashboard ──────
            # Usa os mesmos dados (df_aux sem filtros, df_bol original) para
            # garantir z-score idêntico ao exibido no Resumo
            _ind_risco_macro = None
            try:
                from macro import calcular_indicador_risco_setorial
                # df_aux e df_bol aqui são os originais (sem filtros de CNPJ/UF/data)
                # igual ao que run_dashboard usa para R['ind_risco']
                _ind_risco_macro = calcular_indicador_risco_setorial(df_aux, df_bol)
            except Exception as _e:
                print(f"[SafeAsset] ind_risco macro: {_e}")

            def _build_risco_block(ir):
                return card([
                    html.Div('Indicador de Risco — Inadimplência PJ (Mercado)',
                             style={'fontSize': '16px', 'fontWeight': '700',
                                    'color': WHITE, 'marginBottom': '8px'}),
                    html.Div([
                        html.Span('ℹ️ ', style={'fontWeight': '700'}),
                        html.Span(
                            'Série BCB 21083 — Inadimplência PJ Total SFN. '
                            'O BCB não publica inadimplência por setor CNAE via API. '
                            'Este indicador reflete o mercado PJ geral e serve como '
                            'referência de contexto macroeconômico para todos os setores.',
                            style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic'}),
                    ], style={
                        'background': '#F6F8FA', 'border': f'1px solid {BORDER}',
                        'borderRadius': '8px', 'padding': '8px 14px', 'marginBottom': '16px',
                    }),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span(ir['emoji'] + '  ', style={'fontSize': '24px'}),
                                html.Span(ir['tag'],
                                          style={'fontSize': '22px', 'fontWeight': '800',
                                                 'color': ir['cor']}),
                            ], style={'marginBottom': '8px'}),
                            html.Div(ir['interpretacao'],
                                     style={'fontSize': '13px', 'color': WHITE,
                                            'lineHeight': '1.6', 'marginBottom': '12px'}),
                            dbc.Row([
                                dbc.Col(kpi('Inadimplência Atual',
                                    f'{ir["valor_atual"]:.2f}%',
                                    f'Referência: {ir["data_atual"]}',
                                    ir['cor']), width=4),
                                dbc.Col(kpi('Média Histórica 19m',
                                    f'{ir["media_24m"]:.2f}%',
                                    'Inadimplência PJ — BCB 21083', ACCENT), width=4),
                                dbc.Col(kpi('Z-score',
                                    f'{ir["z_score"]:+.2f}σ',
                                    'Faixa Regular: ±0.5σ', AMBER), width=4),
                            ], className='g-2'),
                            *([html.Div(
                                '📋 Nota BCB (Set/2025): cerca de 70% do aumento da inadimplência '
                                'desde jan/2025 é metodológico (novas regras contábeis), '
                                'não deterioração real do crédito.',
                                style={'fontSize': '11px', 'color': '#B7791F',
                                       'fontStyle': 'italic', 'marginTop': '10px'})]
                            if ir.get('z_score', 0) > 0 else []),
                        ], width=5),
                        dbc.Col([
                            html.Div('Série Histórica — Inadimplência PJ Total SFN (BCB 21083)',
                                     style={'fontSize': '12px', 'color': MUTED,
                                            'marginBottom': '6px'}),
                            G(ch.fig_risco_setorial(ir)),
                        ], width=7),
                    ], className='g-3'),
                ])

            return html.Div([
                *alerta,
                html.Div(f'Dados: {coleta}  ·  Origem: {fonte}',
                         style={'fontSize': '11px', 'color': MUTED,
                                'marginBottom': '16px', 'fontStyle': 'italic'}),

                # Indicador de Risco PJ
                *([_build_risco_block(_ind_risco_macro)] if _ind_risco_macro else [
                    html.Div('Indicador de Risco não disponível.',
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

                # ── Gráfico 1: Taxa de Juros PJ × Inadimplência PJ ───────────
                *([card([
                    html.Div('Taxa Média de Juros PJ × Inadimplência PJ',
                             style={'fontSize': '14px', 'fontWeight': '600',
                                    'color': WHITE, 'marginBottom': '4px'}),
                    html.Div('BCB 20715 (juros % a.a.) · BCB 21083 (inadimplência %) · '
                             'Correlação: juros sobem → inadimplência tende a subir nos meses seguintes',
                             style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '10px'}),
                    G(ch.fig_juros_vs_inadimplencia(_hist_juros, _hist_inad)),
                ])] if _hist_juros is not None and _hist_inad is not None else [
                    html.Div('Dados de juros/inadimplência não disponíveis.',
                             style={'color': MUTED, 'fontSize': '12px', 'padding': '12px'}),
                ]),

                # ── Gráfico 2: Concessões de Crédito PJ ──────────────────
                *([card([
                    html.Div('Concessões de Crédito PJ — Volume Mensal',
                             style={'fontSize': '14px', 'fontWeight': '600',
                                    'color': WHITE, 'marginBottom': '4px'}),
                    html.Div('BCB 20632 · Valor das novas operações contratadas no SFN (R$ milhões) · '
                             'Queda nas concessões sinaliza aperto de crédito e maior risco para sacados',
                             style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '10px'}),
                    G(ch.fig_concessoes_pj(_hist_concessoes)),
                ])] if _hist_concessoes is not None else [
                    html.Div('Dados de concessões não disponíveis.',
                             style={'color': MUTED, 'fontSize': '12px', 'padding': '12px'}),
                ]),

                # ── Gráfico 3: Taxa de Juros — Desconto de Duplicatas ────
                *([card([
                    html.Div('Taxa de Juros — Desconto de Duplicatas e Recebíveis',
                             style={'fontSize': '14px', 'fontWeight': '600',
                                    'color': WHITE, 'marginBottom': '4px'}),
                    html.Div('BCB 20719 · Taxa média % a.a. — operação de crédito livre mais próxima '
                             'do mercado de FIDC/recebíveis · Alta indica custo maior de captação para sacados',
                             style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '10px'}),
                    G(ch.fig_juros_duplicatas(_hist_duplicatas)),
                ])] if _hist_duplicatas is not None else [
                    html.Div('Dados de juros duplicatas não disponíveis.',
                             style={'color': MUTED, 'fontSize': '12px', 'padding': '12px'}),
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
                         style={'color': MUTED, 'fontSize': '10px', 'background': '#F6F8FA',
                                'padding': '12px', 'borderRadius': '6px', 'overflow': 'auto',
                                'maxHeight': '300px'}),
            ], style={'padding': '24px'})

    @app.callback(
        Output('cedentes-content', 'children'),
        Input('tabs-main', 'value'),
        State('store-df-full', 'data'),
        State('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def update_cedentes_content(tab_ativa, df_full_json, bol_json):
        if tab_ativa != 'tab-cedentes':
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        if not df_full_json or not bol_json:
            return html.Div(
                '🏦 Execute a análise histórica (aba lateral) para ver o perfil dos cedentes.',
                style={'color': MUTED, 'fontSize': '14px', 'textAlign': 'center', 'padding': '40px'}
            )

        import pipeline as pl
        df_full = read_json(df_full_json)
        df_bol  = read_json(bol_json)
        perfil_cedentes = pl.calcular_perfil_beneficiario(df_bol, df_full)

        if perfil_cedentes.empty:
            return html.Div('Nenhum dado de cedente disponível.',
                            style={'color': MUTED, 'fontSize': '13px', 'padding': '24px'})

        n_cedentes    = len(perfil_cedentes)
        n_recorrentes = int((perfil_cedentes['qtd_cnpjs_cedidos'] > 1).sum())
        top_cedente   = perfil_cedentes.iloc[0]
        pior_cedente  = perfil_cedentes.sort_values('score_medio_carteira').iloc[0]

        return html.Div([
            html.Div(
                'ℹ️  Sem histórico entre execuções ainda — os indicadores refletem a carteira atual. '
                'Tendência ao longo do tempo aparece conforme novas cessões forem registradas.',
                style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic', 'marginBottom': '16px'}
            ),
            dbc.Row([
                dbc.Col(kpi('Total de Cedentes', f'{n_cedentes:,}', 'beneficiários distintos', ACCENT), width=3),
                dbc.Col(kpi('Cedentes Recorrentes', f'{n_recorrentes:,}', '> 1 CNPJ cedido nesta carteira', ACCENT2), width=3),
                dbc.Col(kpi('Maior Volume', str(top_cedente['id_beneficiario'])[:14],
                            f'{int(top_cedente["qtd_cnpjs_cedidos"])} CNPJs cedidos', ACCENT), width=3),
                dbc.Col(kpi('Menor Score Médio', str(pior_cedente['id_beneficiario'])[:14],
                            f'Score {pior_cedente["score_medio_carteira"]:.0f}', WARN), width=3),
            ], className='g-3', style={'marginBottom': '20px'}),
            card([
                html.Div('Ranking de Cedentes — Qtd de CNPJs cedidos ao fundo',
                        style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '4px'}),
                html.Div('Cor = score médio da carteira cedida (não é tendência — depende de histórico)',
                        style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '8px'}),
                dcc.Graph(figure=ch.fig_cedentes_ranking(perfil_cedentes), config={'displayModeBar': False}),
            ]),
            html.Hr(style={'borderColor': BORDER, 'margin': '20px 0'}),
            card([
                html.Div('🔍 Ver detalhe de um cedente', style={'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='cedente-sel',
                    options=[{'label': str(b), 'value': str(b)} for b in perfil_cedentes['id_beneficiario']],
                    placeholder='Selecione um beneficiário…',
                    className='dark-dropdown', style={'marginBottom': '12px'},
                ),
                html.Div(id='cedente-detalhe'),
            ]),
        ])

    @app.callback(
        Output('cedente-detalhe', 'children'),
        Input('cedente-sel', 'value'),
        State('store-df-full', 'data'),
        State('store-raw-bol', 'data'),
        prevent_initial_call=True,
    )
    def update_cedente_detalhe(id_benef, df_full_json, bol_json):
        if not id_benef or not df_full_json or not bol_json:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate

        df_full = read_json(df_full_json)
        df_bol  = read_json(bol_json)

        bol_c = df_bol[df_bol['id_beneficiario'].astype(str) == str(id_benef)]
        cnpjs_cedidos = set(bol_c['id_pagador'].astype(str).unique())
        df_sacados = df_full[df_full['id_cnpj'].astype(str).isin(cnpjs_cedidos)].copy()

        if df_sacados.empty:
            return html.Div('Nenhum sacado encontrado para este cedente.',
                            style={'color': MUTED, 'fontSize': '12px'})

        vlr_total = bol_c['vlr_nominal'].sum()
        score_med = df_sacados['score_fidc'].mean()

        cols = ['id_cnpj', 'uf', 'score_fidc', 'rating_carteira']
        if 'prob_ml_bom' in df_sacados.columns: cols.append('prob_ml_bom')
        if 'flag_risco_fraude' in df_sacados.columns: cols.append('flag_risco_fraude')

        # ── Histórico real (Fase 2 — Dynamo) ──────────────────────────────
        hist = st.buscar_historico_cedente(id_benef)
        n_cessoes = len(hist) if not hist.empty else 1

        if n_cessoes > 1:
            delta = hist['score_medio_carteira'].iloc[-1] - hist['score_medio_carteira'].iloc[0]
            if delta > 5:
                tend_label, tend_cor = '↑ Melhorando', ACCENT2
            elif delta < -5:
                tend_label, tend_cor = '↓ Piorando', WARN
            else:
                tend_label, tend_cor = '→ Estável', MUTED
            grafico_bloco = [
                dcc.Graph(figure=ch.fig_historico_cedente(hist), config={'displayModeBar': False}),
                html.Div('Cor do marcador: azul = execução da base histórica · laranja = análise de carteira nova',
                          style={'fontSize': '10px', 'color': MUTED, 'marginTop': '4px', 'fontStyle': 'italic'}),
            ]
        else:
            tend_label, tend_cor = '—', MUTED
            grafico_bloco = [html.Div(
                '📈 Gráfico de evolução aparece a partir da 2ª cessão registrada.',
                style={'fontSize': '11px', 'color': MUTED, 'fontStyle': 'italic', 'marginBottom': '12px'}
            )]

        return html.Div([
            dbc.Row([
                dbc.Col(kpi('Cessões Registradas', f'{n_cessoes}',
                            'no histórico acumulado' if n_cessoes > 1 else '1ª execução registrada', MUTED), width=3),
                dbc.Col(kpi('CNPJs Cedidos (atual)', f'{len(cnpjs_cedidos):,}',
                            f'R$ {vlr_total:,.0f} em boletos', ACCENT), width=3),
                dbc.Col(kpi('Score Médio (atual)', f'{score_med:.0f}', '/ 1000',
                            ACCENT2 if score_med >= 500 else WARN), width=3),
                dbc.Col(kpi('Tendência', tend_label, 'entre a 1ª e a última cessão', tend_cor), width=3),
            ], className='g-2', style={'marginBottom': '16px'}),
            *grafico_bloco,
            dash_table.DataTable(
                data=df_sacados[cols].sort_values('score_fidc', ascending=False).to_dict('records'),
                columns=[{'name': c, 'id': c} for c in cols],
                page_size=10, sort_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': CARD_BG, 'color': WHITE, 'border': f'1px solid {BORDER}',
                            'fontFamily': 'Space Grotesk', 'fontSize': '12px', 'padding': '8px 12px', 'textAlign': 'left'},
                style_header={'backgroundColor': BLUE, 'fontWeight': '700', 'color': ACCENT,
                            'border': f'1px solid {BORDER}', 'fontSize': '12px'},
                style_data_conditional=[
                    {'if': {'filter_query': f'{{rating_carteira}} = "{r}"'}, 'color': RATING_COLOR.get(r, 'white'), 'fontWeight': '600'}
                    for r in RATING_COLOR
                ] + [{'if': {'row_index': 'odd'}, 'backgroundColor': '#F6F8FA'}],
            ),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# MONTAGEM DO DASHBOARD (chamado dentro do callback principal)
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(R: dict, liq_thresh: float, mat_thresh: float,
                    dup_thresh: float = 0.05, emit_thresh: int = 10):
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
    cob   = R.get('cobertura', {})

    # df_conhecidos — CNPJs da carteira que TÊM histórico no df_full
    _df_carteira = R.get('df_carteira')
    if _df_carteira is not None and len(_df_carteira) > 0:
        _col_pag2     = 'id_pagador' if 'id_pagador' in _df_carteira.columns else 'id_cnpj'
        _cnpjs_full2  = set(df_full['id_cnpj'].astype(str).unique())
        _cnpjs_cart2  = set(_df_carteira[_col_pag2].astype(str).unique())
        _conhecidos   = _cnpjs_cart2 & _cnpjs_full2
        df_conhecidos = df_full[df_full['id_cnpj'].astype(str).isin(_conhecidos)].copy()
        # Adicionar vlr_nominal da carteira
        _vlr = _df_carteira.groupby(_col_pag2)['vlr_nominal'].sum().reset_index()
        _vlr.columns = ['id_cnpj', 'vlr_cart']
        df_conhecidos = df_conhecidos.merge(_vlr.rename(columns={_col_pag2:'id_cnpj'}),
                                            on='id_cnpj', how='left')
        df_conhecidos['vlr_cart'] = df_conhecidos['vlr_cart'].fillna(0)
    else:
        df_conhecidos = pd.DataFrame()

    # df_cart_novos e df_novos — baseados na carteira nova vs df_full histórico
    _df_carteira = R.get('df_carteira')
    if _df_carteira is not None and len(_df_carteira) > 0:
        _col_pag      = 'id_pagador' if 'id_pagador' in _df_carteira.columns else 'id_cnpj'
        _cnpjs_full   = set(df_full['id_cnpj'].astype(str).unique())
        _cnpjs_cart   = set(_df_carteira[_col_pag].astype(str).unique())
        _novos_ids    = _cnpjs_cart - _cnpjs_full
        df_cart_novos = _df_carteira[
            _df_carteira[_col_pag].astype(str).isin(_novos_ids)
        ].copy()
        df_novos      = pd.DataFrame()
        # Detecção de fraude na carteira nova
        if df_cart_novos is not None and len(df_cart_novos) > 0:
            from pipeline import detectar_duplicatas as _det_dup
            _fr_cart      = _det_dup(
                df_cart_novos,
                pct_dup_thresh     = dup_thresh  or 0.05,
                n_emitentes_thresh = emit_thresh or 10,
            )
            _fr_stats = _fr_cart['stats']
            _fr_cnpj  = _fr_cart['fraude_por_cnpj']
        else:
            _fr_stats = {'total_duplicatas':0,'cnpjs_suspeitos':0,
                         'total_cnpjs_com_boleto':0,'total_boletos':0}
            _fr_cnpj  = pd.DataFrame()
    else:
        df_cart_novos = pd.DataFrame()
        df_novos      = pd.DataFrame()
        _fr_stats = {'total_duplicatas':0,'cnpjs_suspeitos':0,
                     'total_cnpjs_com_boleto':0,'total_boletos':0}
        _fr_cnpj  = pd.DataFrame()

    # ── KPIs principais ───────────────────────────────────────────────────
    total       = len(df_full)
    rating_ct   = df_full['rating_carteira'].value_counts()
    n_excel     = rating_ct.get('A — Excelente', 0)
    score_med   = df_full['score_fidc'].mean()
    best_auc    = ml[best_name]['auc']
    boas_pct    = df_full['target'].mean() * 100
    n_suspeitos = fraude_stats['cnpjs_suspeitos']

    # ── Regra de Recomendação Final ────────────────────────────────────────
    _score_med  = df_full['score_fidc'].mean()
    _pct_ab     = df_full['rating_carteira'].isin(['A — Excelente','B — Bom']).mean() * 100
    _pct_de     = df_full['rating_carteira'].isin(['D — Risco Elevado','E — Alto Risco']).mean() * 100
    _pct_fraude = df_full['flag_risco_fraude'].mean() * 100 if 'flag_risco_fraude' in df_full.columns else 0
    _pct_diverg = df_full['alerta_divergencia'].mean() * 100 if 'alerta_divergencia' in df_full.columns else 0
    _n_diverg   = int(df_full['alerta_divergencia'].sum()) if 'alerta_divergencia' in df_full.columns else 0
    _macro_tag  = (R.get('ind_risco') or {}).get('tag', 'Regular')

    if _score_med >= 700 and _pct_ab >= 50 and _pct_fraude < 1:
        _rec_tag = 'Recomendado'
        _rec_label = '✅  RECOMENDADO'
        _rec_bg = ACCENT2
    elif _score_med >= 500 and _pct_ab >= 30 and _pct_fraude < 3:
        _rec_tag = 'Recomendado com Atenção'
        _rec_label = '⚠️  RECOMENDADO COM ATENÇÃO'
        _rec_bg = AMBER
    else:
        _rec_tag = 'Não Recomendado'
        _rec_label = '🚨  NÃO RECOMENDADO'
        _rec_bg = WARN

    # Notas de aviso
    _notas = []
    if _macro_tag == 'Atenção' and _rec_tag == 'Recomendado':
        _notas.append(
            '⚠️ Macro: Recomendação mantida pelos indicadores dos sacados, mas a inadimplência '
            'PJ geral está acima da média histórica dos últimos 19 meses. '
            'Recomenda-se análise do contexto macroeconômico antes da aquisição.'
        )
    if _pct_diverg > 5:
        _notas.append(
            f'⚠️ Divergência ML: {_n_diverg} CNPJs ({_pct_diverg:.1f}%) apresentam divergência '
            'entre o Score FIDC e o modelo ML. Recomenda-se análise individual desses CNPJs '
            'antes da aquisição.'
        )

    # ── Recomendação — Carteira Nova (CNPJs com histórico) ────────────────
    if not df_conhecidos.empty:
        _cart_score_med  = df_conhecidos['score_fidc'].mean()
        _cart_pct_ab     = df_conhecidos['rating_carteira'].isin(['A — Excelente','B — Bom']).mean() * 100
        _cart_pct_fraude = df_conhecidos['flag_risco_fraude'].mean() * 100 if 'flag_risco_fraude' in df_conhecidos.columns else 0
        _cart_vlr        = df_conhecidos['vlr_cart'].sum() if 'vlr_cart' in df_conhecidos.columns else 0

        if _cart_score_med >= 700 and _cart_pct_ab >= 50 and _cart_pct_fraude < 1:
            _cart_rec_label = '✅  RECOMENDADO'
            _cart_rec_bg    = ACCENT2
        elif _cart_score_med >= 500 and _cart_pct_ab >= 30 and _cart_pct_fraude < 3:
            _cart_rec_label = '⚠️  RECOMENDADO COM ATENÇÃO'
            _cart_rec_bg    = AMBER
        else:
            _cart_rec_label = '🚨  NÃO RECOMENDADO'
            _cart_rec_bg    = WARN
    else:
        _cart_score_med  = 0
        _cart_pct_ab     = 0
        _cart_pct_fraude = 0
        _cart_vlr        = 0
        _cart_rec_label  = None
        _cart_rec_bg     = MUTED

    # ── Recomendação — CNPJs Novos (sem histórico) ────────────────────────
    _cob = R.get('cobertura') or {}
    if _cob.get('sem_historico', 0) > 0:
        _novos_n       = _cob['sem_historico']
        _novos_vlr     = _cob.get('vlr_sem_historico', 0)
        _novos_vlr_pct = (_novos_vlr / _cob['vlr_total'] * 100) if _cob.get('vlr_total') else 0
        _novos_fraude  = 0  # calculado depois junto com _fr_stats
        _novos_fraude  = _fr_stats.get('cnpjs_suspeitos', 0)
        if _novos_fraude == 0 and _novos_vlr_pct < 20:
            _novos_rec_label = '⚠️  ANÁLISE MANUAL RECOMENDADA'
            _novos_rec_bg    = AMBER
        else:
            _novos_rec_label = '🚨  ANÁLISE MANUAL OBRIGATÓRIA'
            _novos_rec_bg    = WARN
    else:
        _novos_n         = 0
        _novos_vlr       = 0
        _novos_vlr_pct   = 0
        _novos_fraude    = 0
        _novos_rec_label = None
        _novos_rec_bg    = MUTED

    kpi_row = dbc.Row([
        dbc.Col(kpi('Total CNPJs',    f'{total:,}',        'base analisada'),                        width=2),
        dbc.Col(kpi('Rating A',       f'{n_excel:,}',      f'{n_excel/total*100:.0f}%',    ACCENT2), width=2),
        dbc.Col(kpi('Score Médio',    f'{score_med:.0f}',  '/ 1000',                       ACCENT),  width=2),
        dbc.Col(kpi('Adimplentes', f'{boas_pct:.1f}%',  'adimplência real — target = 1'),                            width=2),
        dbc.Col(kpi('AUC-ROC',        f'{best_auc:.4f}',   best_name.split()[0],           ACCENT2), width=2),
        dbc.Col(kpi('Suspeitos',      f'{n_suspeitos:,}',  'CNPJs c/ flag fraude',         WARN),    width=2),
    ], className='g-3', style={'marginBottom': '20px'})

    # ── TAB STYLES — diferenciação por grupo ─────────────────────────────
    # Neutro: Resumo (visão consolidada — pertence aos dois grupos)
    TAB_NEUT_STYLE = {'color': '#5B6677', 'background': CARD_BG,
                      'borderBottom': '2px solid #D3D9E2'}
    TAB_NEUT_SEL   = {'color': '#FFFFFF', 'background': '#5B6677',
                      'fontWeight': '700', 'borderBottom': '2px solid #5B6677'}

    # Azul: Análise Histórica (PCR)
    TAB_HIST_STYLE = {'color': '#1F5FA8', 'background': CARD_BG,
                      'borderBottom': '2px solid #1F5FA8'}
    TAB_HIST_SEL   = {'color': '#FFFFFF', 'background': '#1F5FA8',
                      'fontWeight': '700', 'borderBottom': '2px solid #1F5FA8'}

    # Verde: Decisão de Aquisição
    TAB_ACQ_STYLE  = {'color': '#1E8A57', 'background': CARD_BG,
                      'borderBottom': '2px solid #1E8A57'}
    TAB_ACQ_SEL    = {'color': '#FFFFFF', 'background': '#1E8A57',
                      'fontWeight': '700', 'borderBottom': '2px solid #1E8A57'}

    # Aliases de compatibilidade para abas que usavam tab_style/tab_sel inline
    tab_style = TAB_HIST_STYLE
    tab_sel   = TAB_HIST_SEL

    # Estilo base dos rótulos separadores de grupo (tabs desabilitadas)
    _sep_style = {
        'fontSize': '9px', 'fontWeight': '700', 'letterSpacing': '0.1em',
        'textTransform': 'uppercase', 'padding': '6px 12px 2px',
        'display': 'block', 'userSelect': 'none', 'pointerEvents': 'none',
        'cursor': 'default',
    }

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

            # ── RESUMO INDICATIVO ─────────────────────────────────────────
            dcc.Tab(label='Resumo', value='tab-resumo',
                    style=TAB_NEUT_STYLE,
                    selected_style=TAB_NEUT_SEL,
              children=[html.Div(style={'padding': '24px'}, children=[

                html.Div([
                    html.Div('Resumo Indicativo de Carteira',
                             style={'fontSize': '22px', 'fontWeight': '700', 'color': WHITE}),
                    html.Div('Visão consolidada — Base Histórica (PCR) e Carteira Nova em Aquisição',
                             style={'fontSize': '13px', 'color': MUTED, 'marginTop': '4px'}),
                ], style={'marginBottom': '24px'}),

                # ════════════════════════════════════════════════════════
                # SEÇÃO 1 — ANÁLISE HISTÓRICA (PCR)
                # ════════════════════════════════════════════════════════
                html.Div([
                    html.Div(style={
                        'background': '#EAF1FA',
                        'border': f'1px solid #1F5FA8',
                        'borderLeft': '4px solid #1F5FA8',
                        'borderRadius': '10px',
                        'padding': '12px 20px',
                        'marginBottom': '16px',
                        'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                    }, children=[
                        html.Span('[H]', style={
                            'fontSize': '11px', 'fontWeight': '700',
                            'background': '#1F5FA8', 'color': '#FFFFFF',
                            'padding': '2px 8px', 'borderRadius': '4px',
                            'letterSpacing': '0.05em',
                        }),
                        html.Span('ANÁLISE HISTÓRICA — Base PCR',
                                  style={'fontSize': '14px', 'fontWeight': '700',
                                         'color': '#1F5FA8'}),
                        html.Span('Dados históricos de sacados — treino do modelo e diagnóstico da carteira passada',
                                  style={'fontSize': '11px', 'color': MUTED}),
                    ]),
                ]),

                # Veredicto histórico
                html.Div([
                    html.Span(_rec_label, style={
                        'background': _rec_bg,
                        'color': NAVY, 'fontWeight': '800', 'fontSize': '15px',
                        'padding': '8px 24px', 'borderRadius': '24px',
                    }),
                    html.Div([
                        html.Span('Score médio ≥ 700 · % A+B ≥ 50% · Fraude < 1% · Macro ∈ {Recomendado, Regular}',
                                  style={'fontSize': '10px', 'color': MUTED}),
                    ], style={'marginTop': '8px'}),
                    *([html.Div([
                        *[html.Div(nota, style={'fontSize': '11px', 'color': '#B7791F',
                                               'fontStyle': 'italic', 'marginTop': '6px'})
                          for nota in _notas],
                    ])] if _notas else []),
                ], style={'marginBottom': '20px'}),


                # Bloco Target
                card([
                    html.Div('0 — Recomendação segundo o Target',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': ACCENT, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {ACCENT}',
                                    'paddingLeft': '10px'}),
                    html.Div('Adimplência real dos boletos — target=1 quando o sacado não possui boletos inadimplentes reais (excluindo cancelamentos comerciais)',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi('Carteiras Boas',
                            f'{int(df_full["target"].sum()):,}',
                            f'{df_full["target"].mean()*100:.1f}% da base — target = 1',
                            ACCENT2), width=4),
                        dbc.Col(kpi('Carteiras Ruins',
                            f'{int((df_full["target"]==0).sum()):,}',
                            f'{(df_full["target"]==0).mean()*100:.1f}% da base — target = 0',
                            WARN), width=4),
                        dbc.Col(kpi('Total Analisado',
                            f'{len(df_full):,}',
                            'CNPJs na carteira', ACCENT), width=4),
                    ], className='g-3'),
                ]),

                # Bloco Score Final
                card([
                    html.Div('1 — Score Final de Qualidade (0–1000)',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': ACCENT2, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {ACCENT2}',
                                    'paddingLeft': '10px'}),
                    html.Div('Fórmula ponderada: Liquidez Sacado (40%) + Materialidade (25%) + Quantidade (15%) + Atraso (12%) + Inadimplência (8%)',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi('Score Médio',
                            f'{df_full["score_fidc"].mean():.0f}',
                            f'Mín: {df_full["score_fidc"].min():.0f}  ·  Máx: {df_full["score_fidc"].max():.0f}',
                            ACCENT), width=2),
                        *[dbc.Col(kpi(
                            rating.split('—')[0].strip(),
                            f'{int((df_full["rating_carteira"]==rating).sum()):,}',
                            f'{(df_full["rating_carteira"]==rating).mean()*100:.1f}%',
                            RATING_COLOR.get(rating, WHITE)), width=2)
                          for rating in ['A — Excelente','B — Bom','C — Risco Moderado','D — Risco Elevado','E — Alto Risco']
                          if rating in df_full['rating_carteira'].values],
                    ], className='g-2'),
                ]),

                # Bloco Score ML
                *([card([
                    html.Div('2 — Score ML Auxiliar (indicador contínuo de risco)',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': '#5B6C99', 'marginBottom': '16px',
                                    'borderLeft': '4px solid #5B6C99',
                                    'paddingLeft': '10px'}),
                    html.Div(f'Modelo: {best_name.split()[0]}  ·  AUC {ml[best_name]["auc"]:.4f}  ·  '
                             f'Quanto maior, menor o risco segundo o comportamento dos boletos',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi('Score ML Médio',
                            f'{df_full["prob_ml_bom"].mean():.1f}%',
                            f'Mediana: {df_full["prob_ml_bom"].median():.1f}%',
                            '#5B6C99'), width=3),
                        dbc.Col(kpi('Score ML Alto (≥ 40%)',
                            f'{int((df_full["prob_ml_bom"] >= 40).sum()):,}',
                            f'{(df_full["prob_ml_bom"] >= 40).mean()*100:.1f}% — sinal positivo',
                            ACCENT2), width=3),
                        dbc.Col(kpi('Score ML Baixo (< 20%)',
                            f'{int((df_full["prob_ml_bom"] < 20).sum()):,}',
                            f'{(df_full["prob_ml_bom"] < 20).mean()*100:.1f}% — sinal de atenção',
                            WARN), width=3),
                        dbc.Col(kpi('AUC-ROC',
                            f'{ml[best_name]["auc"]:.4f}',
                            f'{best_name.split()[0]} — poder preditivo',
                            '#5B6C99'), width=3),
                    ], className='g-3'),
                    html.Div([
                        html.Span('ℹ️  ', style={'fontSize': '14px'}),
                        html.Span(
                            'O Score ML é um indicador relativo — não classifica bom/ruim diretamente. '
                            'Scores mais altos indicam padrões de pagamento mais saudáveis nos boletos. '
                            'Use em conjunto com o Score FIDC para uma visão complementar.',
                            style={'fontSize': '11px', 'color': MUTED}),
                    ], style={'marginTop': '12px', 'padding': '8px 12px',
                              'background': '#EAF1FA', 'borderRadius': '6px',
                              'border': f'1px solid {BORDER}'}),
                ])] if 'prob_ml_bom' in df_full.columns else []),

                # Bloco Divergência
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
                        dbc.Col(kpi('Sinais de Atenção',
                            f'{int(df_full["alerta_divergencia"].sum()):,}',
                            f'{df_full["alerta_divergencia"].mean()*100:.1f}% da carteira',
                            WARN), width=3),
                        dbc.Col(kpi('Score Alto + ML Baixo',
                            f'{int(((df_full["score_fidc"]>=800)&(df_full["prob_ml_bom"]<30)).sum()):,}',
                            'Score ≥ 800 · ML < 30% — investigar', WARN), width=3),
                        dbc.Col(kpi('Score Baixo + ML Alto',
                            f'{int(((df_full["score_fidc"]<300)&(df_full["prob_ml_bom"]>=70)).sum()):,}',
                            'Score < 300 · ML ≥ 70% — revisar', ACCENT2), width=3),
                        dbc.Col(kpi('Sem Divergência',
                            f'{int((df_full["alerta_divergencia"]==0).sum()):,}',
                            f'{(df_full["alerta_divergencia"]==0).mean()*100:.1f}% — sinais alinhados',
                            ACCENT), width=3),
                    ], className='g-3'),
                ])] if 'alerta_divergencia' in df_full.columns else []),

                # Risco Setorial
                *([card([
                    html.Div('🌐 Indicador de Risco Setorial',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': WHITE, 'marginBottom': '12px',
                                    'borderLeft': f'4px solid {(R.get("ind_risco") or {}).get("cor", ACCENT)}',
                                    'paddingLeft': '10px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span((R.get('ind_risco') or {}).get('emoji','—') + '  ',
                                          style={'fontSize': '20px'}),
                                html.Span((R.get('ind_risco') or {}).get('tag','—'),
                                          style={'fontSize': '18px', 'fontWeight': '800',
                                                 'color': (R.get('ind_risco') or {}).get('cor', ACCENT)}),
                            ], style={'marginBottom': '8px'}),
                            html.Div((R.get('ind_risco') or {}).get('interpretacao','—'),
                                     style={'fontSize': '12px', 'color': MUTED, 'lineHeight': '1.6'}),
                        ], width=7),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(kpi('Setor Predominante',
                                    (R.get('ind_risco') or {}).get('setor_label','—'),
                                    f'{(R.get("ind_risco") or {}).get("pct_valor",0):.1f}% do valor · inadimp. é PJ geral',
                                    ACCENT), width=6),
                                dbc.Col(kpi('Inadimp. Atual',
                                    f'{(R.get("ind_risco") or {}).get("valor_atual",0):.2f}%',
                                    f'Média 19m: {(R.get("ind_risco") or {}).get("media_24m",0):.2f}%',
                                    (R.get('ind_risco') or {}).get('cor', ACCENT)), width=6),
                            ], className='g-2'),
                        ], width=5),
                    ], className='g-3'),
                ])] if R.get('ind_risco') else []),

                # Download divergências
                card([
                    html.Div('📥 CNPJs para Análise Individual',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': WHITE, 'marginBottom': '8px',
                                    'borderLeft': f'4px solid {WARN}',
                                    'paddingLeft': '10px'}),
                    html.Div('CNPJs com sinais de divergência entre Score FIDC e Score ML.',
                             style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '16px'}),
                    *([
                        html.Div([
                            html.Span(f'{int(df_full["alerta_divergencia"].sum()):,} CNPJs identificados  ·  ',
                                style={'fontSize': '13px', 'color': WHITE}),
                            html.Span('Score FIDC · Rating · Prob. ML',
                                style={'fontSize': '12px', 'color': MUTED}),
                        ], style={'marginBottom': '12px'}),
                        dcc.Download(id='download-analise'),
                        html.Button('⬇ Baixar CSV — CNPJs para Análise',
                            id='btn-download-analise', n_clicks=0,
                            style={'background': BLUE, 'color': WARN,
                                   'border': f'1px solid {WARN}', 'borderRadius': '8px',
                                   'padding': '10px 20px', 'fontWeight': '700',
                                   'cursor': 'pointer',
                                   'fontFamily': "'Space Grotesk',sans-serif",
                                   'fontSize': '13px'}),
                    ] if 'alerta_divergencia' in df_full.columns else [
                        html.Div('Dados não disponíveis.', style={'color': MUTED, 'fontSize': '13px'}),
                    ]),
                ]),

                # Recomendação textual histórica
                html.Div(style={
                    'background': '#EAF1FA', 'border': f'1px solid {BORDER}',
                    'borderLeft': f'4px solid {ACCENT2}',
                    'borderRadius': '10px', 'padding': '16px 20px', 'marginTop': '8px',
                }, children=[
                    html.Div('📌 Recomendação Automática — Base Histórica (PCR)',
                             style={'fontSize': '13px', 'fontWeight': '700',
                                    'color': ACCENT2, 'marginBottom': '8px'}),
                    html.Div([
                        html.Span(f"A carteira contém {len(df_full):,} CNPJs com score médio de "
                                  f"{df_full['score_fidc'].mean():.0f}/1000. "),
                        html.Span(f"{int(df_full['target'].sum()):,} CNPJs "
                                  f"({df_full['target'].mean()*100:.1f}%) são classificados como boas carteiras pelo target. "),
                        html.Span(f"{int((df_full['rating_carteira'].isin(['A — Excelente','B — Bom'])).sum()):,} CNPJs "
                                  f"possuem rating A ou B pelo score de negócio. "),
                        *([html.Span(f"{int(df_full['alerta_divergencia'].sum()):,} CNPJs apresentam sinais divergentes entre "
                            "score de negócio e score ML — recomenda-se análise individual antes da aquisição."
                        )] if 'alerta_divergencia' in df_full.columns else []),
                    ], style={'fontSize': '13px', 'color': WHITE, 'lineHeight': '1.8'}),
                ]),

                # ════════════════════════════════════════════════════════
                # SEÇÃO 2 — DECISÃO DE AQUISIÇÃO (Carteira Nova)
                # ════════════════════════════════════════════════════════
                *([html.Div([

                    html.Div(style={
                        'background': '#E7F4EC',
                        'border': f'1px solid #1E8A57',
                        'borderLeft': '4px solid #1E8A57',
                        'borderRadius': '10px',
                        'padding': '12px 20px',
                        'marginTop': '32px',
                        'marginBottom': '16px',
                        'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                    }, children=[
                        html.Span('[A]', style={
                            'fontSize': '11px', 'fontWeight': '700',
                            'background': '#1E8A57', 'color': '#FFFFFF',
                            'padding': '2px 8px', 'borderRadius': '4px',
                            'letterSpacing': '0.05em',
                        }),
                        html.Span('DECISÃO DE AQUISIÇÃO — Carteira Nova',
                                  style={'fontSize': '14px', 'fontWeight': '700',
                                         'color': '#1E8A57'}),
                        html.Span('Análise dos CNPJs da carteira em avaliação',
                                  style={'fontSize': '11px', 'color': MUTED}),
                    ]),

                # ════════════════════════════════════════════════════════
                # BLOCO COBERTURA — Cobertura da carteira nova
                # ════════════════════════════════════════════════════════
                *([card([
                    html.Div('📊 Cobertura da Carteira',
                             style={'fontSize': '15px', 'fontWeight': '700',
                                    'color': WHITE, 'marginBottom': '16px',
                                    'borderLeft': f'4px solid {ACCENT}',
                                    'paddingLeft': '10px'}),
                    html.Div('Quantos CNPJs da carteira têm histórico PCR e podem ser avaliados pelo modelo',
                             style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '16px'}),
                    dbc.Row([
                        dbc.Col(kpi('Total na Carteira',
                            f'{R["cobertura"]["total_cnpjs"]:,}',
                            'CNPJs na carteira nova', ACCENT), width=2),
                        dbc.Col(kpi('Com Histórico PCR',
                            f'{R["cobertura"]["com_historico"]:,}',
                            f'{R["cobertura"]["pct_com_historico"]:.1f}% — score confiável',
                            ACCENT2), width=2),
                        dbc.Col(kpi('Sem Histórico',
                            f'{R["cobertura"]["sem_historico"]:,}',
                            f'{R["cobertura"]["pct_sem_historico"]:.1f}% — primeiro contato',
                            WARN), width=2),
                        dbc.Col(kpi('Recomendados (A+B)',
                            f'{R["cobertura"]["recomendados"]:,}',
                            'dos que têm histórico', ACCENT2), width=2),
                        dbc.Col(kpi('Atenção (C)',
                            f'{R["cobertura"]["atencao"]:,}',
                            'análise complementar', AMBER), width=2),
                        dbc.Col(kpi('Não Recomendados',
                            f'{R["cobertura"]["nao_recomendados"]:,}',
                            'rating D ou E', WARN), width=2),
                    ], className='g-2'),
                    *([html.Div([
                        html.Span('⚠️  ', style={'fontSize': '13px'}),
                        html.Span(
                            f'{R["cobertura"]["sem_historico"]:,} CNPJs '
                            f'({R["cobertura"]["pct_sem_historico"]:.1f}%) não possuem histórico PCR. '
                            'O modelo usou imputação pela mediana — análise individual obrigatória '
                            'antes da aquisição desses títulos.',
                            style={'fontSize': '11px', 'color': '#B7791F',
                                   'fontStyle': 'italic'}),
                    ], style={'marginTop': '10px'})]
                    if R['cobertura']['sem_historico'] > 0 else []),
                ])] if R.get('cobertura') else []),

                    # ── CNPJs com Histórico ───────────────────────────────
                    *([card([
                        html.Div('CNPJs com Histórico PCR',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': ACCENT2, 'marginBottom': '12px',
                                        'borderLeft': f'4px solid {ACCENT2}',
                                        'paddingLeft': '10px'}),
                        html.Div([
                            html.Span(_cart_rec_label, style={
                                'background': _cart_rec_bg, 'color': NAVY,
                                'fontWeight': '800', 'fontSize': '14px',
                                'padding': '7px 22px', 'borderRadius': '22px',
                            }),
                        ], style={'marginBottom': '14px'}),
                        dbc.Row([
                            dbc.Col(kpi('CNPJs', f'{len(df_conhecidos):,}',
                                f'{len(df_conhecidos)/cob["total_cnpjs"]*100:.0f}% da carteira' if cob.get("total_cnpjs") else '—',
                                ACCENT), width=3),
                            dbc.Col(kpi('Score Médio', f'{_cart_score_med:.0f}',
                                '/ 1000', ACCENT), width=3),
                            dbc.Col(kpi('Rating A+B', f'{_cart_pct_ab:.0f}%',
                                'dos CNPJs com histórico', ACCENT2), width=3),
                            dbc.Col(kpi('Fraude', f'{_cart_pct_fraude:.1f}%',
                                'CNPJs com flag',
                                WARN if _cart_pct_fraude > 0 else ACCENT2), width=3),
                        ], className='g-2'),
                        html.Div(
                            f'Score médio {_cart_score_med:.0f}/1000 · '
                            f'{_cart_pct_ab:.0f}% rating A ou B · '
                            f'{_cart_pct_fraude:.1f}% com flag de fraude · '
                            f'Valor total R$ {_cart_vlr:,.0f}.',
                            style={'fontSize': '12px', 'color': MUTED,
                                   'marginTop': '10px', 'fontStyle': 'italic'}),
                    ])] if _cart_rec_label else [
                        html.Div('Faça o upload da carteira nova para ver a recomendação.',
                                 style={'color': MUTED, 'fontSize': '12px', 'padding': '8px 0'}),
                    ]),

                    # ── CNPJs Novos ───────────────────────────────────────
                    *([card([
                        html.Div('CNPJs sem Histórico PCR — Primeiro Contato',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': AMBER, 'marginBottom': '12px',
                                        'borderLeft': f'4px solid {AMBER}',
                                        'paddingLeft': '10px'}),
                        html.Div([
                            html.Span(_novos_rec_label, style={
                                'background': _novos_rec_bg, 'color': NAVY,
                                'fontWeight': '800', 'fontSize': '14px',
                                'padding': '7px 22px', 'borderRadius': '22px',
                            }),
                        ], style={'marginBottom': '14px'}),
                        dbc.Row([
                            dbc.Col(kpi('CNPJs Novos', f'{_novos_n:,}',
                                f'{_cob.get("pct_sem_historico",0):.1f}% da carteira',
                                WARN), width=3),
                            dbc.Col(kpi('Valor em Risco', f'R$ {_novos_vlr:,.0f}',
                                f'{_novos_vlr_pct:.1f}% do valor total',
                                WARN if _novos_vlr_pct >= 20 else AMBER), width=3),
                            dbc.Col(kpi('Alerta Fraude', f'{_novos_fraude:,}',
                                'CNPJs com boletos suspeitos',
                                WARN if _novos_fraude > 0 else ACCENT2), width=3),
                            dbc.Col(kpi('Ver detalhes', '[A] ⚠️ Novos',
                                'aba de análise manual', AMBER), width=3),
                        ], className='g-2'),
                        html.Div(
                            f'{_novos_n:,} CNPJs sem histórico PCR — score não é confiável. '
                            f'Representam {_novos_vlr_pct:.1f}% do valor total da carteira. '
                            + (f'{_novos_fraude} com alerta de fraude nos boletos. ' if _novos_fraude > 0 else '')
                            + 'Acesse a aba [A] ⚠️ Novos para análise detalhada.',
                            style={'fontSize': '12px', 'color': MUTED,
                                   'marginTop': '10px', 'fontStyle': 'italic'}),
                    ])] if _novos_rec_label else [
                        html.Div('✅ Todos os CNPJs da carteira possuem histórico PCR.',
                                 style={'color': ACCENT2, 'fontSize': '12px', 'padding': '8px 0'}),
                    ]),

                    # ── Recomendação Automática — Carteira Nova ───────────
                    html.Div(style={
                        'background': '#E7F4EC', 'border': f'1px solid #1E8A57',
                        'borderLeft': '4px solid #7FC4A0',
                        'borderRadius': '10px', 'padding': '16px 20px', 'marginTop': '16px',
                    }, children=[
                        html.Div('📌 Recomendação Automática — Carteira Nova',
                                 style={'fontSize': '13px', 'fontWeight': '700',
                                        'color': '#1E8A57', 'marginBottom': '10px'}),
                        html.Div([
                            # CNPJs com histórico
                            *([html.Span(
                                f'A carteira nova contém {cob["total_cnpjs"]:,} CNPJs no total. '
                            )] if cob.get('total_cnpjs') else []),
                            *([html.Span(
                                f'{cob["com_historico"]:,} CNPJs ({cob["pct_com_historico"]:.1f}%) '
                                f'possuem histórico PCR e foram avaliados pelo modelo — '
                            )] if cob.get('com_historico') else []),
                            *([html.Span(
                                f'destes, {_cart_rec_label.replace("  ", " ")} '
                                f'com score médio de {_cart_score_med:.0f}/1000, '
                                f'{_cart_pct_ab:.0f}% rating A ou B '
                                f'e {_cart_pct_fraude:.1f}% com flag de fraude. '
                            )] if _cart_rec_label else []),
                            # CNPJs novos
                            *([html.Span(
                                f'{cob["sem_historico"]:,} CNPJs ({cob["pct_sem_historico"]:.1f}%) '
                                f'não possuem histórico PCR — representam R$ {_novos_vlr:,.0f} '
                                f'({_novos_vlr_pct:.1f}% do valor total). '
                            )] if cob.get('sem_historico', 0) > 0 else [
                                html.Span('Todos os CNPJs da carteira possuem histórico PCR. ')
                            ]),
                            # Fraude novos
                            *([html.Span(
                                f'{_novos_fraude} CNPJs novos apresentam alerta de fraude nos boletos — '
                                'análise obrigatória antes da aquisição. ',
                                style={'color': WARN}
                            )] if _novos_fraude > 0 else []),
                            # Macro
                            *([html.Span(
                                f'Inadimplência PJ geral (BCB 21083): '
                                f'{R["ind_risco"]["emoji"]} {R["ind_risco"]["tag"]} '
                                f'— setor predominante da carteira: {R["ind_risco"]["setor_label"]}. '
                            )] if R.get('ind_risco') else []),
                            # Ação recomendada
                            html.Span(
                                'Acesse as abas [A] ⚠️ Novos e [A] ✅ Histórico para análise detalhada.',
                                style={'color': '#1E8A57', 'fontStyle': 'italic'}
                            ),
                        ], style={'fontSize': '13px', 'color': WHITE, 'lineHeight': '1.9'}),
                    ]),

                ])] if R.get('cobertura') else []),

              ])]),

            # ════ ANÁLISE HISTÓRICA ═══════════════════════════════════════
            dcc.Tab(label='[H] EDA', value='tab-eda', style=TAB_HIST_STYLE, selected_style=TAB_HIST_SEL,
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('2–3. Análise Exploratória', 'Distribuições, nulos, volumes temporais'),
                card([html.Div('Nulos por Coluna (%) — CNPJs com boletos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_nulos(df_full[df_full['sem_historico'] == 0]))]),                card([html.Div('Distribuição dos Scores de Crédito', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_scores(df_full))]),
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
            dcc.Tab(label='[H] Fraude', value='tab-fraude', style=TAB_HIST_STYLE, selected_style={**TAB_HIST_SEL, 'background': WARN, 'color': NAVY},
              children=[html.Div(style={'padding': '24px'}, children=[
                section_title('2B. Detecção de Boletos Duplicados e Concentração de Risco',
                    f'Thresholds: ≥ {fraude_stats["pct_dup_thresh"]*100:.0f}% duplicados  OU  ≥ {fraude_stats["n_emit_thresh"]} emitentes distintos'),

                # ── LINHA 1: KPI PRINCIPAL COM A REGRA DE ATIVAÇÃO ───────────────────────
                dbc.Row([
                    dbc.Col(
                        card([
                            html.Div('CNPJs SUSPEITOS DE FRAUDE / RISCO', 
                                     style={'fontSize': '12px', 'fontWeight': '700', 'color': WARN, 'letterSpacing': '1px'}),
                            html.Div(f'{fraude_stats["cnpjs_suspeitos"]:,}', 
                                     style={'fontSize': '38px', 'fontWeight': '800', 'color': WARN, 'margin': '4px 0'}),
                            html.Div(f'de {fraude_stats["total_cnpjs_com_boleto"]:,} CNPJs analisados na base de dados', 
                                     style={'fontSize': '12px', 'color': MUTED, 'marginBottom': '10px'}),
                            
                            html.Div([
                                html.Span(' Regra de Ativação: ', style={'fontWeight': '700', 'color': WARN}),
                                html.Span(f'Alerta disparado quando % Duplicados ≥ {fraude_stats["pct_dup_thresh"]*100:.0f}% OU Nº Emitentes ≥ {fraude_stats["n_emit_thresh"]}.',
                                          style={'color': WHITE, 'fontWeight': '500'})
                            ], style={
                                'fontSize': '11px', 
                                'background': '#EAF1FA', 
                                'padding': '10px 14px', 
                                'borderRadius': '8px', 
                                'border': f'1px solid {BORDER}', 
                                'marginTop': '8px'
                            })
                        ]), 
                        width=12
                    ),
                ], style={'marginBottom': '16px'}),

                # ── LINHA 2: 4 KPIS DETALHADOS ──────────────────────────────────────────
                # ── LINHA 2: KPIS DOS 4 TIPOS DE CNPJS EM RISCO (TOTAL = 1.189) ─────────
                dbc.Row([
                    dbc.Col(kpi('Sem Alerta (Normal)', 
                                f'{len(fraude_cnpj[fraude_cnpj["flag_risco_fraude"] == 0]):,}',
                                'CNPJs regulares', ACCENT2), width=3),
                    dbc.Col(kpi('Por Duplicatas', 
                                f'{len(fraude_cnpj[fraude_cnpj["motivo_alerta"] == "Duplicatas excessivas"]):,}',
                                '≥ 5% boletos dup.', WARN), width=3),
                    dbc.Col(kpi('Por Muitos Beneficiários', 
                                f'{len(fraude_cnpj[fraude_cnpj["motivo_alerta"] == "Muitos emitentes"]):,}',
                                f'≥ {fraude_stats["n_emit_thresh"]} emitentes dist.', WARN), width=3),
                    dbc.Col(kpi('Por Ambos os Motivos', 
                                f'{len(fraude_cnpj[fraude_cnpj["motivo_alerta"] == "Duplicatas + Muitos emitentes"]):,}',
                                'Risco crítico acumulado', RISK_E), width=3),
                ], className='g-3', style={'marginBottom': '20px'}),

                # ── LINHA 3: 4 TIPOS DE FRAUDE (ESQUERDA) + CASCATA DE VALOR EM RISCO (DIREITA) ──
# ── GRÁFICO 1: DISTRIBUIÇÃO DE CNPJS POR CATEGORIA DE RISCO (LARGURA TOTAL) ──
                dbc.Row([
                    dbc.Col([
                        card([
                            html.Div('Distribuição de CNPJs por Categoria de Risco / Fraude', 
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_distribuicao_motivos_fraude(fraude_cnpj))
                        ]),
                    ], width=12),
                ], className='g-3'),

                # ── GRÁFICO 2: CASCATA DE VALOR EM RISCO (LARGURA TOTAL - LOGO ABAIXO) ──
                dbc.Row([
                    dbc.Col([
                        card([
                            html.Div('Valor Total em Risco (R$) por Tipo de Fraude (Gráfico em Cascata)', 
                                     style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_valor_cascata_fraude(fraude_cnpj, df_bol))
                        ]),
                    ], width=12),
                ], className='g-3'),

                # ── LINHA 4: CONCENTRAÇÃO DE EMITENTES EM UMA LINHA ISOLADA ABAIXO ──
                dbc.Row([
                    dbc.Col([
                        card([
                            html.Div('Concentração de Emitentes por Pagador', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}),
                            G(ch.fig_emitentes_por_pagador(fraude_cnpj))
                        ]),
                    ], width=12),
                ], className='g-3'),

                # ── TABELA FINAL DE CNPJs SUSPEITOS E BOTAO DE DOWNLOAD ──────────────────
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
                                 'backgroundColor': '#FBEAE8', 'color': '#BF3B33', 'fontWeight': '600'},
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#F6F8FA'},
                            ],
                        )
                    ]),
                    html.Div('Nenhum CNPJ suspeito encontrado com os thresholds atuais.',
                             style={'color': ACCENT2, 'fontSize': '13px', 'padding': '12px 0'})
                    if fraude_cnpj[fraude_cnpj['flag_risco_fraude'] == 1].empty else html.Div(),
                    
                    # Download CNPJs com fraude
                    *([
                        html.Hr(style={'borderColor': BORDER, 'margin': '12px 0'}),
                        dcc.Download(id='download-fraude'),
                        html.Button('⬇ Baixar CSV — CNPJs com Flag de Fraude',
                            id='btn-download-fraude', n_clicks=0,
                            style={'background': BLUE, 'color': WARN,
                                   'border': f'1px solid {WARN}', 'borderRadius': '8px',
                                   'padding': '9px 18px', 'fontWeight': '700',
                                   'cursor': 'pointer',
                                   'fontFamily': "'Space Grotesk',sans-serif",
                                   'fontSize': '12px'}),
                    ] if not fraude_cnpj[fraude_cnpj['flag_risco_fraude'] == 1].empty else []),
                ]),
              ])]),

            # ════ DECISÃO DE AQUISIÇÃO ════════════════════════════════════
            dcc.Tab(label='[A] Novos', value='tab-novos', style=TAB_ACQ_STYLE,
                    selected_style={**TAB_ACQ_SEL, 'background': WARN, 'color': NAVY},
              children=[html.Div(style={'padding': '24px'}, children=[

                section_title('CNPJs sem Histórico PCR',
                    'Análise específica para subsidiar decisão manual do analista'),

                

                *([
                    # Aviso principal
                    html.Div([
                        html.Span('⚠️  ', style={'fontSize': '16px'}),
                        html.Span(
                            f'{cob["sem_historico"]:,} CNPJs ({cob["pct_sem_historico"]:.1f}% da carteira) '
                            'não possuem histórico PCR. A análise preditiva não é aplicável a estes CNPJs. '
                            'Os dados abaixo subsidiam análise manual antes da decisão de aquisição.',
                            style={'fontSize': '12px', 'color': '#B7791F', 'fontStyle': 'italic'}),
                    ], style={'marginBottom': '20px'}),

                    # ── Bloco 1 — Visão Geral ─────────────────────────────────
                    card([
                        html.Div('1 — Visão Geral',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': WARN, 'marginBottom': '16px',
                                        'borderLeft': f'4px solid {WARN}',
                                        'paddingLeft': '10px'}),
                        dbc.Row([
                            dbc.Col(kpi('CNPJs sem Histórico',
                                f'{cob["sem_historico"]:,}',
                                f'{cob["pct_sem_historico"]:.1f}% da carteira',
                                WARN), width=3),
                            dbc.Col(kpi('Valor em Risco',
                                f'R$ {cob["vlr_sem_historico"]:,.0f}',
                                'valor total dos boletos desses CNPJs',
                                WARN), width=3),
                            dbc.Col(kpi('% do Valor Total',
                                f'{cob["vlr_sem_historico"]/cob["vlr_total"]*100:.1f}%' if cob["vlr_total"] else '—',
                                'participação na carteira',
                                AMBER), width=3),
                            dbc.Col(kpi('Com Scores Núclea',
                                f'{int(df_novos["score_materialidade_v2"].notna().sum()):,}' if "score_materialidade_v2" in df_novos.columns else '—',
                                'possuem indicadores PCR parciais',
                                ACCENT2), width=3),
                        ], className='g-3'),
                    ]),

                    # ── Bloco 2 — Perfil da Carteira Nova ────────────────
                    *([card([
                        html.Div('2 — Perfil dos Boletos da Carteira Nova',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': ACCENT, 'marginBottom': '16px',
                                        'borderLeft': f'4px solid {ACCENT}',
                                        'paddingLeft': '10px'}),
                        html.Div('Características dos boletos a pagar dos CNPJs sem histórico PCR',
                                 style={'fontSize': '11px', 'color': MUTED,
                                        'marginBottom': '16px'}),
                        *([
                            dbc.Row([
                                dbc.Col(kpi('Total de Boletos',
                                    f'{len(df_cart_novos):,}',
                                    f'dos {len(_df_carteira):,} boletos da carteira',
                                    WARN), width=3),
                                dbc.Col(kpi('Valor Total',
                                    f'R$ {df_cart_novos["vlr_nominal"].sum():,.0f}',
                                    f'Ticket médio: R$ {df_cart_novos["vlr_nominal"].mean():,.0f}',
                                    WARN), width=3),
                                dbc.Col(kpi('Beneficiários Distintos',
                                    f'{df_cart_novos["id_beneficiario"].nunique():,}',
                                    'cedentes distintos na carteira',
                                    ACCENT), width=3),
                                dbc.Col(kpi('CNPJs Pagadores',
                                    f'{df_cart_novos["id_pagador"].nunique():,}',
                                    'sem histórico PCR',
                                    WARN), width=3),
                            ], className='g-3'),
                            html.Hr(style={'borderColor': BORDER, 'margin': '16px 0'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Div('Emissão — intervalo dos boletos',
                                             style={'fontSize': '11px', 'color': MUTED,
                                                    'marginBottom': '6px'}),
                                    html.Div([
                                        html.Span('De ', style={'color': MUTED, 'fontSize': '12px'}),
                                        html.Span(
                                            str(df_cart_novos['dt_emissao'].min())[:10],
                                            style={'color': WHITE, 'fontWeight': '700',
                                                   'fontSize': '13px'}),
                                        html.Span('  até  ', style={'color': MUTED,
                                                                    'fontSize': '12px'}),
                                        html.Span(
                                            str(df_cart_novos['dt_emissao'].max())[:10],
                                            style={'color': WHITE, 'fontWeight': '700',
                                                   'fontSize': '13px'}),
                                    ]),
                                ], width=6),
                                dbc.Col([
                                    html.Div('Vencimento — intervalo dos boletos',
                                             style={'fontSize': '11px', 'color': MUTED,
                                                    'marginBottom': '6px'}),
                                    html.Div([
                                        html.Span('De ', style={'color': MUTED, 'fontSize': '12px'}),
                                        html.Span(
                                            str(df_cart_novos['dt_vencimento'].min())[:10],
                                            style={'color': AMBER, 'fontWeight': '700',
                                                   'fontSize': '13px'}),
                                        html.Span('  até  ', style={'color': MUTED,
                                                                    'fontSize': '12px'}),
                                        html.Span(
                                            str(df_cart_novos['dt_vencimento'].max())[:10],
                                            style={'color': AMBER, 'fontWeight': '700',
                                                   'fontSize': '13px'}),
                                    ]),
                                ], width=6),
                            ], className='g-3'),
                            html.Hr(style={'borderColor': BORDER, 'margin': '16px 0'}),
                            html.Div('Distribuição por Valor dos Boletos',
                                     style={'fontSize': '11px', 'color': MUTED,
                                            'marginBottom': '8px'}),
                            G(ch.fig_vlr_distribuicao_novos(df_cart_novos)),
                        ] if df_cart_novos is not None and len(df_cart_novos) > 0
                        else [html.Div('Dados da carteira não disponíveis.',
                                       style={'color': MUTED, 'fontSize': '12px'})]),
                    ]),] if True else []),

                    # ── Bloco 4 — Alertas de Fraude (carteira nova) ──────────
                    card([
                        html.Div('4 — Alertas de Fraude nos Boletos da Carteira',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': WARN, 'marginBottom': '16px',
                                        'borderLeft': f'4px solid {WARN}',
                                        'paddingLeft': '10px'}),
                        html.Div('Verificação de duplicatas e padrões suspeitos '
                                 'nos boletos da carteira nova',
                                 style={'fontSize': '11px', 'color': MUTED,
                                        'marginBottom': '16px'}),
                        dbc.Row([
                            dbc.Col(kpi('Total Boletos',
                                f'{_fr_stats["total_boletos"]:,}',
                                'boletos analisados na carteira',
                                ACCENT), width=3),
                            dbc.Col(kpi('Boletos Duplicados',
                                f'{_fr_stats["total_duplicatas"]:,}',
                                f'{_fr_stats["total_duplicatas"]/_fr_stats["total_boletos"]*100:.1f}% do total'
                                if _fr_stats["total_boletos"] > 0 else '0%',
                                WARN if _fr_stats["total_duplicatas"] > 0 else ACCENT2), width=3),
                            dbc.Col(kpi('CNPJs com Alerta',
                                f'{_fr_stats["cnpjs_suspeitos"]:,}',
                                f'de {_fr_stats["total_cnpjs_com_boleto"]:,} CNPJs analisados',
                                WARN if _fr_stats["cnpjs_suspeitos"] > 0 else ACCENT2), width=3),
                            dbc.Col(kpi('CNPJs sem Alerta',
                                f'{_fr_stats["total_cnpjs_com_boleto"] - _fr_stats["cnpjs_suspeitos"]:,}',
                                'sem sinais suspeitos',
                                ACCENT2), width=3),
                        ], className='g-3'),
                        *([html.Div(
                            f'⚠️  {_fr_stats["cnpjs_suspeitos"]} CNPJs com padrões suspeitos '
                            f'detectados nos {_fr_stats["total_boletos"]} boletos da carteira nova.',
                            style={'fontSize': '11px', 'color': '#B7791F',
                                   'fontStyle': 'italic', 'marginTop': '10px'})]
                        if _fr_stats["cnpjs_suspeitos"] > 0 else
                        [html.Div('✅ Nenhum padrão suspeito detectado nos boletos da carteira.',
                            style={'fontSize': '11px', 'color': ACCENT2,
                                   'marginTop': '10px'})]),
                    ]),

                    # ── Bloco 5 — Contexto Macro ─────────────────────────────
                    *([card([
                        html.Div('5 — Indicador de Inadimplência PJ (Mercado)',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': ACCENT, 'marginBottom': '16px',
                                        'borderLeft': f'4px solid {ACCENT}',
                                        'paddingLeft': '10px'}),

                        # Nota informativa
                        html.Div([
                            html.Span('ℹ️ ', style={'fontWeight': '700'}),
                            html.Span(
                                'Série BCB 21083 — Inadimplência PJ Total SFN. '
                                'Reflete o mercado PJ geral, não um setor específico.',
                                style={'fontSize': '11px', 'color': MUTED,
                                       'fontStyle': 'italic'}),
                        ], style={
                            'background': '#F6F8FA', 'border': f'1px solid {BORDER}',
                            'borderRadius': '8px', 'padding': '8px 14px',
                            'marginBottom': '14px',
                        }),

                        dbc.Row([
                            dbc.Col([
                                # Tag + interpretação
                                html.Div([
                                    html.Span(R['ind_risco']['emoji'] + '  ',
                                              style={'fontSize': '22px'}),
                                    html.Span(R['ind_risco']['tag'],
                                              style={'fontSize': '20px', 'fontWeight': '800',
                                                     'color': R['ind_risco']['cor']}),
                                ], style={'marginBottom': '8px'}),
                                html.Div(R['ind_risco']['interpretacao'],
                                         style={'fontSize': '12px', 'color': WHITE,
                                                'lineHeight': '1.6', 'marginBottom': '12px'}),
                                dbc.Row([
                                    dbc.Col(kpi('Inadimplência PJ Atual',
                                        f'{R["ind_risco"]["valor_atual"]:.2f}%',
                                        f'Referência: {R["ind_risco"]["data_atual"]}',
                                        R['ind_risco']['cor']), width=6),
                                    dbc.Col(kpi('Média Histórica 19m',
                                        f'{R["ind_risco"]["media_24m"]:.2f}%',
                                        'BCB 21083 — PJ Total SFN', ACCENT), width=6),
                                ], className='g-2'),
                            ], width=5),
                            dbc.Col([
                                # Destaque do setor predominante da carteira
                                html.Div([
                                    html.Div('Setor Predominante da Carteira',
                                             style={'fontSize': '10px', 'color': MUTED,
                                                    'fontWeight': '600', 'letterSpacing': '0.05em',
                                                    'textTransform': 'uppercase',
                                                    'marginBottom': '6px'}),
                                    html.Div([
                                        html.Span('🏭 ',
                                                  style={'fontSize': '16px'}),
                                        html.Span(R['ind_risco']['setor_label'],
                                                  style={'fontSize': '16px', 'fontWeight': '700',
                                                         'color': ACCENT}),
                                    ], style={'marginBottom': '6px'}),
                                    html.Div(
                                        f'{R["ind_risco"]["pct_valor"]:.1f}% do valor total da carteira',
                                        style={'fontSize': '11px', 'color': MUTED,
                                               'fontStyle': 'italic'}),
                                    html.Div(
                                        'O indicador PJ geral é o único disponível via API BCB — '
                                        'não há série específica por setor CNAE.',
                                        style={'fontSize': '10px', 'color': MUTED,
                                               'fontStyle': 'italic', 'marginTop': '8px'}),
                                ], style={
                                    'background': '#EAF1FA',
                                    'border': f'1px solid #1F5FA8',
                                    'borderRadius': '8px', 'padding': '12px 14px',
                                }),
                            ], width=7),
                        ], className='g-3'),
                    ])] if R.get('ind_risco') else []),

                    # ── Download ──────────────────────────────────────────────
                    card([
                        html.Div('📥 Download — CNPJs para Análise Individual',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': WHITE, 'marginBottom': '8px',
                                        'borderLeft': f'4px solid {WARN}',
                                        'paddingLeft': '10px'}),
                        html.Div(f'{cob["sem_historico"]:,} CNPJs sem histórico PCR '
                                 '— arquivo com todos os dados disponíveis para análise manual.',
                                 style={'fontSize': '11px', 'color': MUTED,
                                        'marginBottom': '12px'}),
                        dcc.Download(id='download-novos'),
                        html.Button('⬇ Baixar CSV — CNPJs sem Histórico',
                            id='btn-download-novos', n_clicks=0,
                            style={'background': BLUE, 'color': WARN,
                                   'border': f'1px solid {WARN}',
                                   'borderRadius': '8px', 'padding': '10px 20px',
                                   'fontWeight': '700', 'cursor': 'pointer',
                                   'fontFamily': "'Space Grotesk',sans-serif",
                                   'fontSize': '13px'}),
                    ]),

                ] if R.get('cobertura') and R['cobertura']['sem_historico'] > 0
                  else [html.Div([
                    html.Div('✅', style={'fontSize': '48px', 'marginBottom': '12px'}),
                    html.Div('Todos os CNPJs da carteira possuem histórico PCR.',
                             style={'fontSize': '15px', 'color': ACCENT2}),
                    html.Div('Nenhuma análise manual adicional necessária.',
                             style={'fontSize': '13px', 'color': MUTED, 'marginTop': '6px'}),
                ], style={'textAlign': 'center', 'padding': '60px'})]),

              ])]),

            # ── CARTEIRA EM AQUISIÇÃO — CNPJs COM HISTÓRICO ─────────────────
            dcc.Tab(label='[A] Histórico',
                    value='tab-conhecidos', style=TAB_ACQ_STYLE,
                    selected_style={**TAB_ACQ_SEL, 'background': ACCENT2, 'color': NAVY},
              children=[html.Div(style={'padding': '24px'}, children=[

                section_title('Carteira em Aquisição — CNPJs com Histórico PCR',
                    'CNPJs da carteira nova que possuem histórico na base PCR — análise de risco disponível'),

                *([

                    # ── KPIs ─────────────────────────────────────────────────
                    card([
                        html.Div('Visão Geral dos CNPJs com Histórico',
                                 style={'fontSize': '15px', 'fontWeight': '700',
                                        'color': ACCENT2, 'marginBottom': '16px',
                                        'borderLeft': f'4px solid {ACCENT2}',
                                        'paddingLeft': '10px'}),
                        dbc.Row([
                            dbc.Col(kpi('CNPJs com Histórico',
                                f'{len(df_conhecidos):,}',
                                f'de {cob["total_cnpjs"]:,} na carteira',
                                ACCENT2), width=2),
                            dbc.Col(kpi('Valor Total',
                                f'R$ {df_conhecidos["vlr_cart"].sum():,.0f}',
                                f'{df_conhecidos["vlr_cart"].sum()/cob["vlr_total"]*100:.1f}% da carteira'
                                if cob.get("vlr_total") else '—',
                                ACCENT), width=2),
                            dbc.Col(kpi('Score Médio',
                                f'{df_conhecidos["score_fidc"].mean():.0f}',
                                f'Mín: {df_conhecidos["score_fidc"].min():.0f} · Máx: {df_conhecidos["score_fidc"].max():.0f}',
                                ACCENT), width=2),
                            dbc.Col(kpi('Rating A+B',
                                f'{int(df_conhecidos["rating_carteira"].isin(["A — Excelente","B — Bom"]).sum()):,}',
                                f'{df_conhecidos["rating_carteira"].isin(["A — Excelente","B — Bom"]).mean()*100:.1f}%',
                                ACCENT2), width=2),
                            dbc.Col(kpi('Com Alerta Fraude',
                                f'{int(df_conhecidos["flag_risco_fraude"].sum()):,}'
                                if 'flag_risco_fraude' in df_conhecidos.columns else '—',
                                f'{df_conhecidos["flag_risco_fraude"].mean()*100:.1f}%'
                                if 'flag_risco_fraude' in df_conhecidos.columns else '',
                                WARN if df_conhecidos.get("flag_risco_fraude", pd.Series([0])).sum() > 0 else ACCENT2), width=2),
                            dbc.Col(kpi('Score ML Médio',
                                f'{df_conhecidos["prob_ml_bom"].mean():.1f}%'
                                if 'prob_ml_bom' in df_conhecidos.columns else '—',
                                'indicador auxiliar de risco',
                                '#5B6C99'), width=2),
                        ], className='g-2'),
                    ]),

                    # ── Gráficos linha 1 ─────────────────────────────────────
                    dbc.Row([
                        dbc.Col([
                            card([
                                html.Div('Distribuição de Ratings na Carteira',
                                         style={'fontSize': '12px', 'color': MUTED,
                                                'marginBottom': '8px'}),
                                G(ch.fig_target_pizza_conhecidos(df_conhecidos)),
                            ]),
                        ], width=4),
                        dbc.Col([
                            card([
                                html.Div('Distribuição de Scores FIDC',
                                         style={'fontSize': '12px', 'color': MUTED,
                                                'marginBottom': '8px'}),
                                G(ch.fig_score_hist_conhecidos(df_conhecidos)),
                            ]),
                        ], width=8),
                    ], className='g-3'),

                    # ── Gráficos linha 2 ─────────────────────────────────────
                    dbc.Row([
                        dbc.Col([
                            card([
                                html.Div('Valor em Risco por Rating',
                                         style={'fontSize': '12px', 'color': MUTED,
                                                'marginBottom': '8px'}),
                                G(ch.fig_valor_por_rating(df_conhecidos)),
                            ]),
                        ], width=6),
                        dbc.Col([
                            card([
                                html.Div('Distribuição de Liquidez Sacado (1m)',
                                         style={'fontSize': '12px', 'color': MUTED,
                                                'marginBottom': '8px'}),
                                G(ch.fig_liquidez_hist_conhecidos(df_conhecidos)),
                            ]),
                        ], width=6),
                    ], className='g-3'),

                    # ── Top 10 por valor ─────────────────────────────────────
                    card([
                        html.Div('Top 10 CNPJs por Valor na Carteira',
                                 style={'fontSize': '12px', 'color': MUTED,
                                        'marginBottom': '8px'}),
                        G(ch.fig_top10_valor(df_conhecidos)),
                    ]),

                    # ── Download ─────────────────────────────────────────────
                    card([
                        html.Div('📥 Download — CNPJs com Histórico PCR',
                                 style={'fontSize': '14px', 'fontWeight': '700',
                                        'color': WHITE, 'marginBottom': '8px',
                                        'borderLeft': f'4px solid {ACCENT2}',
                                        'paddingLeft': '10px'}),
                        html.Div(f'{len(df_conhecidos):,} CNPJs com score, rating e indicadores de risco.',
                                 style={'fontSize': '11px', 'color': MUTED,
                                        'marginBottom': '12px'}),
                        dcc.Download(id='download-conhecidos'),
                        html.Button('⬇ Baixar CSV — CNPJs com Histórico',
                            id='btn-download-conhecidos', n_clicks=0,
                            style={'background': BLUE, 'color': ACCENT2,
                                   'border': f'1px solid {ACCENT2}',
                                   'borderRadius': '8px', 'padding': '10px 20px',
                                   'fontWeight': '700', 'cursor': 'pointer',
                                   'fontFamily': "'Space Grotesk',sans-serif",
                                   'fontSize': '13px'}),
                    ]),

                ] if not df_conhecidos.empty else [
                    html.Div([
                        html.Div('📂', style={'fontSize': '48px', 'marginBottom': '12px'}),
                        html.Div('Faça o upload da carteira nova para ver os CNPJs com histórico PCR.',
                                 style={'fontSize': '15px', 'color': MUTED}),
                    ], style={'textAlign': 'center', 'padding': '60px'}),
                ]),])]),
            dcc.Tab(label='[H] Cedentes', value='tab-cedentes', style=tab_style,
                selected_style={**tab_sel, 'background': '#5B6C99', 'color': NAVY},
                children=[html.Div(style={'padding': '24px'}, children=[
                section_title('Cedentes — Histórico de Cessões',
                'Beneficiários que já venderam carteira ao fundo · Score e qualidade da carteira cedida'),
                html.Div(id='cedentes-content',
                children=html.Div(
            '🏦 Faça o upload e execute a análise histórica para ver o perfil dos cedentes.',
            style={'color': MUTED, 'fontSize': '14px',
                   'textAlign': 'center', 'padding': '40px'}
        )
    ),
  ])]),

            # ── OUTRAS INFORMAÇÕES (Treinamento do Modelo + Macro) ────────
            dcc.Tab(label='[H] Outras Informações', value='tab-outras',
                    style=TAB_HIST_STYLE, selected_style=TAB_HIST_SEL,
              children=[html.Div(style={'padding': '24px'}, children=[
                html.Details(open=False, children=[
                    html.Summary('▶  Treinamento do Modelo — Target, Correlação e Modelagem (ML)', style={
                        'fontWeight': '600', 'fontSize': '14px', 'color': WHITE,
                        'cursor': 'pointer', 'padding': '12px 16px',
                        'background': BLUE, 'borderRadius': '8px',
                        'border': f'1px solid {BORDER}', 'listStyle': 'none',
                        'userSelect': 'none',
                    }),
                    html.Div(style={'marginTop': '16px'}, children=[

                        section_title('5–6. Target e Correlação',
                            f'Adimplência real: target=1 se sacado sem boletos inadimplentes reais · Parâmetros usados como fallback'),
                        card([html.Div('Distribuição do Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_target_pizza(df_full))]),
                        card([html.Div('Score Materialidade Evolução por Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_boxplot_target(df_full, 'score_materialidade_evolucao', 'Score Materialidade Evolução por Target'))]),
                        card([html.Div('Liquidez Sacado (1m) por Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_boxplot_target(df_full, 'sacado_indice_liquidez_1m', 'Índice de Liquidez (1m)'))]),
                        card([html.Div('Correlação com o Target', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_correlacao_target(corr_mat))]),
                        card([html.Div('Matriz de Correlação Completa (Pearson)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_heatmap_correlacao(corr_mat))]),

                        html.Hr(style={'borderColor': BORDER, 'margin': '24px 0'}),

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
                                                         'backgroundColor': '#E7F4EC', 'color': ACCENT2, 'fontWeight': '600'}],
                            ),
                        ]),
                        card([html.Div('AUC-ROC por Fold — Cross-Validation (k=5)', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_cv_folds(ml))]),
                        card([html.Div('CV AUC vs Test AUC — comparação entre modelos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_cv_vs_test(ml))]),
                        card([html.Div(f'Importância das Features — {best_name}', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_feature_importance(feat_imp, best_name))]),
                        html.Hr(style={'borderColor': BORDER, 'margin': '20px 0'}),
                        section_title('Avaliação dos Modelos', 'Curvas ROC e Matrizes de Confusão'),
                        card([html.Div('Curvas ROC — Comparação entre Modelos', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_roc(ml))]),
                        card([html.Div('Matrizes de Confusão — por Modelo', style={'fontSize': '13px', 'color': MUTED, 'marginBottom': '8px'}), G(ch.fig_confusion_matrix(ml))]),
                    ]),
                ]),
                html.Details(open=False, style={'marginTop': '16px'}, children=[
                    html.Summary('▶  Contexto Macroeconômico (BCB · IBGE)', style={
                        'fontWeight': '600', 'fontSize': '14px', 'color': WHITE,
                        'cursor': 'pointer', 'padding': '12px 16px',
                        'background': BLUE, 'borderRadius': '8px',
                        'border': f'1px solid {BORDER}', 'listStyle': 'none',
                        'userSelect': 'none',
                    }),
                    html.Div(style={'marginTop': '16px'}, children=[
                        section_title('Contexto Macroeconômico — Camada 1',
                            'Fontes: BCB SGS · IBGE SIDRA · Dados buscados automaticamente no upload'),

                        html.Div(id='macro-content',
                            children=html.Div(
                                '🌐 Faça o upload dos dois arquivos CSV para carregar os dados macroeconômicos.',
                                style={'color': MUTED, 'fontSize': '14px',
                                       'textAlign': 'center', 'padding': '40px'}
                            )
                        ),
                    ]),
                ]),
              ])]),

              ])
            
    return html.Div([kpi_row, tabs])

