# =============================================================================
# layout.py — SafeAsset
# Responsável por: estrutura HTML/CSS do Dash (sidebar, header, tabs)
# Tema: dark tech azul-noite
# =============================================================================

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from pipeline import RATING_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# PALETA — tema escuro original
# ─────────────────────────────────────────────────────────────────────────────
NAVY    = '#0a1628'
BLUE    = '#1e3a5f'
ACCENT  = '#00d4ff'
ACCENT2 = '#00ff88'
WARN    = '#ff6b35'
MUTED   = '#8892a4'
CARD_BG = '#111d2e'
BORDER  = '#1e3a5f'
WHITE   = '#e8f0fe'
DARK    = '#e8f0fe'
BG_PAGE = NAVY
BG_MAIN = CARD_BG

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES REUTILIZÁVEIS
# ─────────────────────────────────────────────────────────────────────────────

def card(children, style=None, className=''):
    s = {'background': CARD_BG, 'border': f'1px solid {BORDER}',
         'borderRadius': '12px', 'padding': '20px', 'marginBottom': '16px'}
    if style:
        s.update(style)
    return html.Div(children, style=s, className=className)


def kpi(label, value, delta=None, color=ACCENT):
    return html.Div([
        html.Div(label, style={'fontSize': '11px', 'color': MUTED,
                               'letterSpacing': '1px', 'textTransform': 'uppercase',
                               'marginBottom': '6px'}),
        html.Div(value, style={'fontSize': '28px', 'fontWeight': '700',
                               'color': color, 'fontFamily': "'JetBrains Mono', monospace"}),
        html.Div(delta, style={'fontSize': '12px', 'color': MUTED,
                               'marginTop': '4px'}) if delta else None,
    ], style={'background': CARD_BG, 'border': f'1px solid {BORDER}',
              'borderRadius': '10px', 'padding': '16px 20px', 'textAlign': 'center'})


def section_title(text, sub=None):
    return html.Div([
        html.Div(text, style={'fontSize': '18px', 'fontWeight': '600',
                              'color': WHITE, 'marginBottom': '4px'}),
        html.Div(sub,  style={'fontSize': '13px', 'color': MUTED}) if sub else None,
        html.Hr(style={'borderColor': BORDER, 'margin': '10px 0 18px'}),
    ])


def badge_step(num, label):
    return html.Span([
        html.Span(str(num), style={
            'background': ACCENT, 'color': NAVY, 'borderRadius': '50%',
            'width': '22px', 'height': '22px', 'display': 'inline-flex',
            'alignItems': 'center', 'justifyContent': 'center',
            'fontSize': '11px', 'fontWeight': '700', 'marginRight': '8px',
        }),
        label,
    ], style={'color': MUTED, 'fontSize': '12px', 'display': 'inline-flex',
              'alignItems': 'center', 'marginRight': '20px'})


def label_sm(text):
    return html.Label(text, style={'fontSize': '11px', 'color': MUTED,
                                   'fontWeight': '500', 'marginBottom': '4px',
                                   'display': 'block'})


def build_rank_table(df_full, cnpj_q=None, sel_ufs=None, sel_cnaes=None,
                     sel_ratings=None, score_range=None):
    df_v = df_full[['id_cnpj', 'uf', 'cd_cnae_prin', 'score_fidc',
                    'rating_carteira', 'target']].copy()
    df_v['rating_carteira'] = df_v['rating_carteira'].astype(str)
    sr = score_range or [0, 1000]
    if cnpj_q:
        df_v = df_v[df_v['id_cnpj'].str.lower().str.contains(cnpj_q.strip().lower(), na=False)]
    if sel_ufs:
        df_v = df_v[df_v['uf'].isin(sel_ufs)]
    if sel_cnaes:
        df_v = df_v[df_v['cd_cnae_prin'].astype(str).isin(sel_cnaes)]
    if sel_ratings:
        df_v = df_v[df_v['rating_carteira'].isin(sel_ratings)]
    df_v = df_v[(df_v['score_fidc'] >= sr[0]) & (df_v['score_fidc'] <= sr[1])]
    df_v = df_v.sort_values('score_fidc', ascending=False).head(300)
    return dash_table.DataTable(
        data=df_v.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df_v.columns],
        page_size=15, sort_action='native', filter_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': CARD_BG, 'color': WHITE,
                    'border': f'1px solid {BORDER}', 'fontFamily': 'Space Grotesk',
                    'fontSize': '12px', 'padding': '8px 12px', 'textAlign': 'left'},
        style_header={'backgroundColor': BLUE, 'fontWeight': '700',
                      'color': ACCENT, 'border': f'1px solid {BORDER}', 'fontSize': '12px'},
        style_data_conditional=[
            {'if': {'filter_query': f'{{rating_carteira}} = "{r}"'},
             'color': RATING_COLOR.get(r, 'white'), 'fontWeight': '600'}
            for r in RATING_COLOR
        ] + [{'if': {'row_index': 'odd'}, 'backgroundColor': '#0d1b2a'}],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar():
    return html.Div(style={'width': '100%'}, children=[

        # Upload
        card([
            html.Div('📁 Dados', style={'fontWeight': '600', 'marginBottom': '6px', 'fontSize': '14px'}),
            dcc.Upload(id='upload-aux', multiple=False,
                       style={'border': f'2px dashed {BORDER}', 'borderRadius': '8px',
                              'cursor': 'pointer', 'marginBottom': '6px'},
                       children=html.Div([
                           html.Div('⬆', style={'fontSize': '20px', 'color': ACCENT}),
                           html.Div('1. Base Auxiliar', style={'fontSize': '12px', 'fontWeight': '600'}),
                           html.Div('base_auxiliar_fiap.csv', style={'fontSize': '10px', 'color': MUTED}),
                       ], style={'textAlign': 'center', 'padding': '12px'})),
            html.Div(id='aux-status', style={'fontSize': '11px', 'color': ACCENT2, 'marginBottom': '6px'}),
            dcc.Upload(id='upload-bol', multiple=False,
                       style={'border': f'2px dashed {BORDER}', 'borderRadius': '8px',
                              'cursor': 'pointer', 'marginBottom': '6px'},
                       children=html.Div([
                           html.Div('⬆', style={'fontSize': '20px', 'color': ACCENT2}),
                           html.Div('2. Base Boletos', style={'fontSize': '12px', 'fontWeight': '600'}),
                           html.Div('base_boletos_fiap.csv', style={'fontSize': '10px', 'color': MUTED}),
                       ], style={'textAlign': 'center', 'padding': '12px'})),
            html.Div(id='bol-status', style={'fontSize': '11px', 'color': ACCENT2, 'marginBottom': '10px'}),
            html.Button('▶ Executar com meus arquivos', id='btn-run-upload', n_clicks=0,
                        style={'width': '100%', 'background': BLUE, 'color': ACCENT,
                               'border': f'1px solid {ACCENT}', 'borderRadius': '8px',
                               'padding': '9px', 'fontWeight': '700', 'cursor': 'pointer',
                               'fontFamily': "'Space Grotesk',sans-serif", 'fontSize': '12px',
                               'marginBottom': '12px'}),
        ]),

        # Parâmetros ML
        card([
            html.Div('⚙️ Parâmetros ML', style={'fontWeight': '600', 'marginBottom': '14px', 'fontSize': '14px'}),
            label_sm('Tamanho do teste'),
            dcc.Slider(id='sl-test', min=0.15, max=0.35, step=0.05, value=0.20,
                       marks={0.15:{'label':'15%','style':{'color':'#8892a4','fontSize':'10px'}},0.20:{'label':'20%','style':{'color':'#00d4ff','fontSize':'10px'}},0.25:{'label':'25%','style':{'color':'#8892a4','fontSize':'10px'}},0.30:{'label':'30%','style':{'color':'#8892a4','fontSize':'10px'}},0.35:{'label':'35%','style':{'color':'#8892a4','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
            html.Br(),
            label_sm('Árvores — Random Forest'),
            dcc.Slider(id='sl-trees', min=100, max=500, step=100, value=300,
                       marks={100:{'label':'100','style':{'color':'#8892a4','fontSize':'10px'}},300:{'label':'300','style':{'color':'#00d4ff','fontSize':'10px'}},500:{'label':'500','style':{'color':'#8892a4','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
            html.Br(),
            label_sm('Threshold Liquidez'),
            dcc.Slider(id='sl-liq', min=0.50, max=0.90, step=0.05, value=0.70,
                       marks={0.50:{'label':'0.5','style':{'color':'#8892a4','fontSize':'10px'}},0.70:{'label':'0.7','style':{'color':'#00d4ff','fontSize':'10px'}},0.90:{'label':'0.9','style':{'color':'#8892a4','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
            html.Br(),
            label_sm('Threshold Materialidade'),
            dcc.Slider(id='sl-mat', min=700, max=970, step=50, value=900,
                       marks={700:{'label':'700','style':{'color':'#8892a4','fontSize':'10px'}},850:{'label':'850','style':{'color':'#8892a4','fontSize':'10px'}},970:{'label':'970','style':{'color':'#00d4ff','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
        ]),

        # Detecção de Fraude
        card([
            html.Div('🚨 Detecção de Fraude', style={'fontWeight': '600', 'marginBottom': '4px', 'fontSize': '14px'}),
            html.Div('Thresholds para flag de risco',
                     style={'fontSize': '10px', 'color': WARN, 'marginBottom': '12px', 'fontStyle': 'italic'}),
            label_sm('% mín. boletos duplicados'),
            dcc.Slider(id='sl-dup-thresh', min=1, max=30, step=1, value=5,
                       marks={1:{'label':'1%','style':{'color':'#8892a4','fontSize':'10px'}},15:{'label':'15%','style':{'color':'#8892a4','fontSize':'10px'}},30:{'label':'30%','style':{'color':'#8892a4','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
            html.Br(),
            label_sm('Nº mín. de emitentes distintos'),
            dcc.Slider(id='sl-emit-thresh', min=2, max=30, step=1, value=10,
                       marks={2:{'label':'2','style':{'color':'#8892a4','fontSize':'10px'}},15:{'label':'15','style':{'color':'#8892a4','fontSize':'10px'}},30:{'label':'30','style':{'color':'#8892a4','fontSize':'10px'}}},
                       tooltip={'always_visible': False}),
            html.Button('🔄 Reaplicar ML + Fraude', id='btn-run', n_clicks=0,
                        style={'width': '100%', 'background': BLUE, 'color': ACCENT,
                               'border': f'1px solid {ACCENT}', 'borderRadius': '8px',
                               'padding': '8px', 'fontWeight': '700', 'cursor': 'pointer',
                               'fontFamily': "'Space Grotesk',sans-serif", 'fontSize': '12px',
                               'marginTop': '10px'}),
        ]),

        # Filtros globais
        card([
            html.Div('🔎 Filtros Globais', style={'fontWeight': '600', 'marginBottom': '4px', 'fontSize': '14px'}),
            html.Div('Aplicados automaticamente ao mudar',
                     style={'fontSize': '10px', 'color': ACCENT2, 'marginBottom': '12px', 'fontStyle': 'italic'}),
            label_sm('Buscar CNPJ'),
            dcc.Input(id='flt-cnpj', placeholder='ex: CNPJ-001', debounce=True,
                      style={'width': '100%', 'background': '#0d1b2a', 'border': f'1px solid {BORDER}',
                             'borderRadius': '6px', 'padding': '8px', 'color': WHITE,
                             'fontFamily': "'JetBrains Mono',monospace", 'fontSize': '12px',
                             'marginBottom': '10px'}),
            label_sm('UF(s)'),
            dcc.Dropdown(id='flt-uf', multi=True, placeholder='Todas — clique para filtrar',
                         style={'marginBottom': '10px'}, className='dark-dropdown'),
            label_sm('CNAE(s)'),
            dcc.Dropdown(id='flt-cnae', multi=True, placeholder='Todos — clique para filtrar',
                         style={'marginBottom': '10px'}, className='dark-dropdown'),
            label_sm('Período de emissão dos boletos'),
            dcc.DatePickerRange(id='flt-date', display_format='DD/MM/YYYY',
                                style={'width': '100%', 'fontSize': '11px'},
                                className='dark-datepicker'),
            html.Button('🗑 Limpar filtros', id='btn-clear', n_clicks=0,
                        style={'width': '100%', 'background': 'transparent',
                               'color': MUTED, 'border': f'1px solid {BORDER}',
                               'borderRadius': '6px', 'padding': '7px', 'cursor': 'pointer',
                               'fontFamily': "'Space Grotesk',sans-serif",
                               'fontSize': '12px', 'marginTop': '10px'}),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def build_header():
    return html.Div([
        html.Div([
            html.Div([
                html.Span('Safe',  style={'color': ACCENT,  'fontWeight': '700', 'fontSize': '26px'}),
                html.Span('Asset', style={'color': WHITE,   'fontWeight': '300', 'fontSize': '26px'}),
                html.Div('Aquisição de ativos de forma segura · FIAP + Núclea',
                         style={'color': MUTED, 'fontSize': '12px', 'marginTop': '2px'}),
            ]),
            html.Div([
                badge_step(i, s) for i, s in enumerate([
                    'Carga', 'Fraude', 'EDA', 'Feat.Eng',
                    'Target', 'Correlação', 'ML', 'Score', 'CNAE', 'Macro',
                ], 1)
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '6px'}),
        ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '16px 24px',
                  'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
    ], style={'background': BLUE, 'borderBottom': f'1px solid {BORDER}',
              'position': 'sticky', 'top': '0', 'zIndex': '100'})


# ─────────────────────────────────────────────────────────────────────────────
# TELA INICIAL
# ─────────────────────────────────────────────────────────────────────────────

def build_welcome():
    return html.Div([
        html.Div('📊', style={'fontSize': '64px', 'marginBottom': '16px'}),
        html.Div('SafeAsset', style={'fontSize': '32px', 'fontWeight': '700',
                                      'color': ACCENT, 'marginBottom': '8px'}),
        html.Div('Faça o upload dos dois CSVs na barra lateral e clique em ▶ Executar.',
                 style={'color': MUTED, 'fontSize': '14px', 'maxWidth': '400px', 'textAlign': 'center'}),
        html.Div([
            html.Span('Pipeline: ', style={'color': MUTED, 'fontSize': '12px'}),
            *[html.Span(f'Passo {i} → ', style={'color': ACCENT, 'fontSize': '12px'}) for i in range(1, 7)],
            html.Span('Score Final', style={'color': ACCENT2, 'fontSize': '12px', 'fontWeight': '700'}),
        ], style={'marginTop': '20px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
              'justifyContent': 'center', 'minHeight': '400px', 'textAlign': 'center'})


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def create_layout():
    return html.Div(
        style={'fontFamily': "'Space Grotesk', sans-serif",
               'backgroundColor': NAVY, 'minHeight': '100vh', 'color': WHITE},
        children=[
            dcc.Store(id='store-raw-aux'),
            dcc.Store(id='store-raw-bol'),
            dcc.Store(id='store-macro'),
            dcc.Store(id='store-sidebar', data='open'),

            build_header(),

            html.Div(
                style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px 24px'},
                children=[
                    # Linha principal — sidebar + conteúdo lado a lado
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'alignItems': 'flex-start',
                            'gap': '0px',
                            'width': '100%',
                        },
                        children=[
                            # ── Sidebar (recolhível) ───────────────────────
                            html.Div(
                                id='sidebar-container',
                                style={
                                    'width': '280px',
                                    'minWidth': '280px',
                                    'flexShrink': '0',
                                    'transition': 'width 0.2s ease, min-width 0.2s ease',
                                    'overflowX': 'hidden',
                                    'position': 'relative',
                                },
                                children=[build_sidebar()]
                            ),

                            # ── Botão toggle (entre sidebar e conteúdo) ────
                            html.Div(
                                html.Button(
                                    '◀',
                                    id='btn-toggle-sidebar',
                                    n_clicks=0,
                                    title='Recolher painel lateral',
                                    style={
                                        'background': BLUE, 'color': ACCENT,
                                        'border': f'1px solid {BORDER}',
                                        'borderRadius': '50%',
                                        'width': '22px', 'height': '22px',
                                        'fontSize': '10px', 'cursor': 'pointer',
                                        'padding': '0', 'flexShrink': '0',
                                        'marginTop': '8px',
                                    }
                                ),
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'alignItems': 'center',
                                    'padding': '0 6px',
                                    'flexShrink': '0',
                                }
                            ),

                            # ── Área de conteúdo principal ─────────────────
                            html.Div(
                                style={
                                    'flex': '1',
                                    'minWidth': '0',
                                    'overflow': 'hidden',
                                },
                                children=[
                                    html.Div(id='main-content',
                                             children=[build_welcome()]),
                                    dcc.Loading(
                                        id='loading', type='circle', color=ACCENT,
                                        children=html.Div(id='loading-output')
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )