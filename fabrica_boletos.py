import pandas as pd
import numpy as np

# 1. Carregar a base de dados original
df = pd.read_csv('C:\\Users\\mcmoi\\OneDrive\\Fiap\\Desafio_Nuclea_Fiap\\Projeto_Fiap_Nuclea\\base_boletos_fiap.csv')

# 2. Alternar (swap) id_pagador e id_beneficiario
df['id_pagador'], df['id_beneficiario'] = df['id_beneficiario'].copy(), df['id_pagador'].copy()

# 3. Ajustar as datas para o período de 2026 mantendo as diferenças relativas
cols_data = ['dt_emissao', 'dt_vencimento', 'dt_pagamento']
for col in cols_data:
    df[col] = pd.to_datetime(df[col])

max_dt = df['dt_emissao'].max()
target_dt = pd.to_datetime('2026-09-30')
days_shift = (target_dt - max_dt).days

for col in cols_data:
    df[col] = df[col] + pd.Timedelta(days=days_shift)
    df[col] = df[col].dt.strftime('%Y-%m-%d')

# 4. Reajustar os valores dos boletos (variação aleatória entre -20% e +20%)
np.random.seed(42)
fator = np.random.uniform(0.8, 1.2, size=len(df))

df['vlr_nominal'] = (df['vlr_nominal'] * fator).round(2)

mask_baixa = df['vlr_baixa'].notnull()
df.loc[mask_baixa, 'vlr_baixa'] = (df.loc[mask_baixa, 'vlr_baixa'] * fator[mask_baixa]).round(2)

# 5. Salvar o novo ficheiro CSV
df.to_csv('C:\\Users\\mcmoi\\OneDrive\\Fiap\\Desafio_Nuclea_Fiap\\Projeto_Fiap_Nuclea\\base_boletos_fiap_atualizada.csv', index=False)
print("Ficheiro 'base_boletos_fiap_atualizada.csv' gerado com sucesso!")