# =============================================================================
# app.py — SafeAsset
# Ponto de entrada da aplicação — apenas inicializa e conecta os módulos
#
# Deploy local:
#   python app.py  →  http://localhost:8050
#
# Deploy Render.com:
#   A variável PORT é injetada automaticamente pelo Render
# =============================================================================

import os
import dash
import dash_bootstrap_components as dbc

from layout import create_layout
from callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700'
        '&family=JetBrains+Mono:wght@400;600&display=swap',
    ],
    suppress_callback_exceptions=True,
)
app.title = 'SafeAsset'
app.layout = create_layout()

register_callbacks(app)

# Expor 'server' para o gunicorn (Render usa gunicorn internamente)
server = app.server

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 8050))
    host  = '0.0.0.0'
    debug = os.environ.get('DASH_DEBUG', 'false').lower() == 'true'
    print('=' * 55)
    print('  SafeAsset — Aquisição de ativos de forma segura')
    print(f'  Acesse: http://localhost:{port}')
    print('=' * 55)
    app.run(host=host, port=port, debug=debug)