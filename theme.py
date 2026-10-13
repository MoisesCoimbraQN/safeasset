# =============================================================================
# theme.py — SafeAsset
# Fonte única da paleta de cores — importada por pipeline.py, layout.py e
# charts.py.
#
# TEMA CLARO — fundo neutro e cards brancos, para que os DADOS (números,
# barras, linhas) sejam o ponto de destaque, não o plano de fundo. Cores de
# dado são mais escuras/saturadas que no tema escuro, para ter contraste
# sobre branco. Poucas famílias de cor: um azul de marca, e uma rampa de risco
# (verde → azul → âmbar → laranja → vermelho) reaproveitada em rating, fraude
# e macro.
#
# Obs.: os NOMES das constantes foram mantidos do tema escuro por
# compatibilidade (WHITE = cor do texto principal, NAVY = fundo da página).
# =============================================================================

# ─── Base neutra ──────────────────────────────────────────────────────────
NAVY    = '#F2F4F7'   # fundo da página — cinza-azulado muito claro
BLUE    = '#E8ECF2'   # fundo de blocos secundários (header, painéis, resumos)
CARD_BG = '#FFFFFF'   # fundo dos cards — branco
BORDER  = '#D3D9E2'   # bordas
WHITE   = '#1A2332'   # TEXTO principal — grafite escuro (nome mantido p/ compat.)
MUTED   = '#5B6677'   # texto secundário

BG_PAGE = NAVY
BG_MAIN = CARD_BG
DARK    = WHITE

# ─── Destaque único ────────────────────────────────────────────────────────
ACCENT  = '#1F5FA8'   # azul de marca — ação e destaque
ACCENT2 = '#1E8A57'   # verde — "positivo/adimplente"

# ─── Rampa de risco (5 níveis, reaproveitada em rating, fraude e macro) ──
RISK_A = '#1E8A57'    # favorável / excelente / adimplente
RISK_B = '#1F5FA8'    # bom / neutro-positivo
RISK_C = '#B7791F'    # atenção / risco moderado
RISK_D = '#C2571A'    # elevado / desfavorável
RISK_E = '#BF3B33'    # alto risco / fraude / inadimplente

WARN   = RISK_D        # alias — compatibilidade
AMBER  = RISK_C         # alias — compatibilidade

RATING_COLOR = {
    'A — Excelente':      RISK_A,
    'B — Bom':            RISK_B,
    'C — Risco Moderado': RISK_C,
    'D — Risco Elevado':  RISK_D,
    'E — Alto Risco':     RISK_E,
}

# ─── Tons categóricos extras (séries com mais de 5 categorias) ───────────
EXTRA1 = '#5B6C99'   # azul-ardósia
EXTRA2 = '#8A7355'   # marrom-areia

# Cores de status do Score Macro — reaproveita a rampa de risco
MACRO_FAVORAVEL    = RISK_A
MACRO_NEUTRO        = RISK_B
MACRO_ATENCAO        = RISK_C
MACRO_DESFAVORAVEL = RISK_E

# ─── Tons de apoio ────────────────────────────────────────────────────────
ACCENT_LIGHT  = '#7FA6D4'   # tint do ACCENT — bordas/realces suaves
ACCENT2_LIGHT = '#7FC4A0'   # tint do ACCENT2

# ─── Fundos tingidos (caixas de alerta/destaque, bem claros) ──────────────
BG_INFO    = '#EAF1FA'   # tingido de azul
BG_SUCCESS = '#E7F4EC'   # tingido de verde
BG_DANGER  = '#FBEAE8'   # tingido de vermelho
DARKER     = '#F6F8FA'   # neutro — listras de tabela, inputs
