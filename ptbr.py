# =============================================================================
# ptbr.py — SafeAsset
# Formatação numérica brasileira: vírgula decimal, ponto como separador de
# milhar (o inverso do padrão americano que o Python/f-strings produzem).
#
# Como funciona: o código do app continua gerando números no padrão US
# (f'{x:,}' → '4,612'; f'{x:.1f}' → '97.7'). Em vez de editar centenas de
# f-strings, os textos passam por br_text() ANTES de ir para a tela:
#   • localizar(componente) — percorre a árvore do Dash e converte textos,
#     tabelas (DataTable) e configura o locale dos gráficos;
#   • br_text(str) — converte só "tokens numéricos" no formato US, ignorando
#     qualquer outra coisa (IDs, datas, CNPJs formatados, versões etc.).
# =============================================================================

import re

# Token numérico: começa e termina em dígito, com , e . no meio.
_RX_TOKEN = re.compile(r'\d[\d.,]*\d|\d')
# Só converte o que é, sem ambiguidade, número no padrão US:
_US_MILHAR  = re.compile(r'^\d{1,3}(,\d{3})+(\.\d+)?$')   # 1,234  /  1,234,567.89
_US_DECIMAL = re.compile(r'^\d+\.\d+$')                    # 97.7   /  0.8339


def _troca(token: str) -> str:
    if _US_MILHAR.match(token) or _US_DECIMAL.match(token):
        return token.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
    return token   # ex.: 12.345.678 (CNPJ), 10.10.2026, 3.14.3 → intocado


def br_text(s):
    """Converte números no padrão US dentro de um texto para o padrão pt-BR."""
    if not isinstance(s, str):
        return s
    return _RX_TOKEN.sub(lambda m: _troca(m.group(0)), s)


# Meses em português (strftime('%b') depende do locale do servidor)
_MESES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun',
          'jul', 'ago', 'set', 'out', 'nov', 'dez']


def mes_ano_pt(ts) -> str:
    """Timestamp → 'jul/2026' (em português, independente do locale)."""
    try:
        return f'{_MESES[ts.month - 1]}/{ts.year}'
    except Exception:
        return str(ts)


_FMT_FLOAT = {
    'locale': {'group': '.', 'decimal': ','},
    'nully': '', 'prefix': None, 'specifier': ',',
}


def _localizar_tabela(tbl):
    """DataTable: floats ganham formato pt-BR; textos já formatados são convertidos."""
    data = tbl.data or []
    cols = tbl.columns or []
    # converte textos pré-formatados (ex.: '66.7%', '0.7517 ± 0.0563')
    for row in data:
        for k, v in row.items():
            if isinstance(v, str):
                row[k] = br_text(v)
    # colunas cujo valor (primeiro não nulo) é float → formato numérico pt-BR
    for col in cols:
        cid = col.get('id')
        amostra = next((r.get(cid) for r in data
                        if r.get(cid) is not None and r.get(cid) == r.get(cid)), None)
        if isinstance(amostra, float):
            col['type'] = 'numeric'
            col['format'] = dict(_FMT_FLOAT)
    return tbl


def localizar(comp):
    """
    Percorre a árvore de componentes do Dash convertendo para pt-BR:
      • textos (children)            → br_text
      • dash_table.DataTable         → formato numérico + textos
      • dcc.Graph                    → config locale pt-BR (meses, separadores)
    Altera os componentes no lugar e devolve o próprio objeto.
    """
    from dash.development.base_component import Component
    if isinstance(comp, str):
        return br_text(comp)
    if isinstance(comp, (list, tuple)):
        return [localizar(c) for c in comp]
    if isinstance(comp, Component):
        nome = type(comp).__name__
        if nome == 'DataTable':
            return _localizar_tabela(comp)
        if nome == 'Graph':
            cfg = dict(getattr(comp, 'config', None) or {})
            cfg.setdefault('locale', 'pt-BR')
            comp.config = cfg
            return comp
        filhos = getattr(comp, 'children', None)
        if filhos is not None:
            comp.children = localizar(filhos)
        return comp
    return comp
