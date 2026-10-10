# =============================================================================
# theme.py — SafeAsset
# Fonte única da paleta de cores — importada por pipeline.py, layout.py e
# charts.py, para nunca mais haver hex duplicado/dessincronizado entre arquivos.
#
# Direção: dashboard executivo de crédito — poucas famílias de cor (não um
# arco-íris), mas cada uma com contraste suficiente pra ler números e texto
# de dado rapidamente contra o fundo escuro. "Profissional" != "apagado" —
# dashboards financeiros sérios (ex: terminais de trading) usam cores vivas
# com disciplina, não cores fracas.
# =============================================================================

# ─── Base neutra ──────────────────────────────────────────────────────────
NAVY    = '#0B0F14'   # fundo da página — grafite quase neutro
BLUE    = '#1A2029'   # fundo de blocos secundários (header, painéis)
CARD_BG = '#141920'   # fundo dos cards
BORDER  = '#2B333F'   # bordas — cinza-azulado neutro (levemente mais claro)
WHITE   = '#EDEFF2'   # texto principal — branco quase puro
MUTED   = '#9BA5B3'   # texto secundário — mais claro que antes, ainda discreto

BG_PAGE = NAVY
BG_MAIN = CARD_BG
DARK    = WHITE

# ─── Destaque único (substitui o antigo ciano+verde+roxo) ────────────────
ACCENT  = '#5B9BD5'   # azul-aço claro — única cor de marca/ação, bem legível
ACCENT2 = '#5FB38A'   # verde-esmeralda — "positivo/adimplente", mesma lógica

# ─── Rampa de risco (5 níveis, reaproveitada em rating, fraude e macro) ──
RISK_A = '#5FB38A'    # favorável / excelente / adimplente
RISK_B = '#5B9BD5'    # bom / neutro-positivo
RISK_C = '#D4A94E'    # atenção / risco moderado
RISK_D = '#D08A4E'    # elevado / desfavorável
RISK_E = '#D16760'    # alto risco / fraude / inadimplente

WARN   = RISK_D        # alias — mantém compatibilidade com o nome usado no código
AMBER  = RISK_C         # alias — idem

RATING_COLOR = {
    'A — Excelente':      RISK_A,
    'B — Bom':            RISK_B,
    'C — Risco Moderado': RISK_C,
    'D — Risco Elevado':  RISK_D,
    'E — Alto Risco':     RISK_E,
}

# ─── Tons categóricos extras (séries com mais de 5 categorias) ───────────
EXTRA1 = '#7D8FB3'   # azul-ardósia claro — substitui o antigo roxo (#a78bfa)
EXTRA2 = '#A6937F'   # marrom-areia claro — substitui o antigo rosa (#f472b6)

# Cores de status do Score Macro (Favorável/Neutro/Atenção/Desfavorável) —
# reaproveita a mesma rampa de risco em vez de uma paleta própria
MACRO_FAVORAVEL    = RISK_A
MACRO_NEUTRO        = RISK_B
MACRO_ATENCAO        = RISK_C
MACRO_DESFAVORAVEL = RISK_E

# ─── Tons claros de apoio (texto sobre fundo colorido, estado selecionado) ─
ACCENT_LIGHT  = '#9CC1E0'   # tint claro do ACCENT — texto/borda sobre fundo ACCENT
ACCENT2_LIGHT = '#9CD3B6'   # tint claro do ACCENT2 — idem, família verde

# ─── Fundos escuros tingidos (caixas de alerta/destaque, bem mais escuros
# que CARD_BG, na mesma família de cor do conteúdo) ────────────────────────
BG_INFO    = '#141B22'   # tingido de azul — blocos informativos (família Histórico)
BG_SUCCESS = '#131C17'   # tingido de verde — blocos de sucesso (família Aquisição)
BG_DANGER  = '#1C1514'   # tingido de vermelho — blocos de alerta/fraude
DARKER     = '#0E1319'   # neutro, mais escuro que CARD_BG — listras de tabela, inputs
