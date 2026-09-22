# =============================================================================
# storage.py — SafeAsset
# Responsável por: persistência de snapshots (S3) e histórico temporal (DynamoDB)
# Sem dependências de Dash — se a AWS falhar, loga e segue sem quebrar o app
# =============================================================================

import os
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REGION        = os.environ.get('AWS_REGION', 'us-east-2')
S3_BUCKET     = os.environ.get('S3_BUCKET', 'safeasset-snapshots')
TABLE_CNPJ    = 'safeasset_cnpj_historico'
TABLE_CEDENTE = 'safeasset_cedente_historico'

_s3 = None
_dynamo = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client('s3', region_name=REGION)
    return _s3


def _dynamo_resource():
    global _dynamo
    if _dynamo is None:
        _dynamo = boto3.resource('dynamodb', region_name=REGION)
    return _dynamo


def _to_decimal(val):
    """Auxiliar para converter números/floats em Decimal seguro para o DynamoDB."""
    if pd.isna(val) or val is None:
        return Decimal('0')
    return Decimal(str(val))


def novo_run_id() -> str:
    """Timestamp ISO 8601 UTC — usado como sort key nas duas tabelas."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def salvar_snapshot_s3(run_id: str, **dfs) -> bool:
    """Sobe CSVs brutos pro S3. Uso: salvar_snapshot_s3(run_id, aux=df_aux, bol=df_bol, ...)"""
    try:
        s3 = _s3_client()
        for nome, df in dfs.items():
            if df is None or df.empty:
                continue
            body = df.to_csv(index=False).encode('utf-8')
            s3.put_object(Bucket=S3_BUCKET, Key=f'runs/{run_id}/{nome}.csv', Body=body)
        return True
    except Exception as e:
        print(f"[SafeAsset] storage.salvar_snapshot_s3 erro: {e}")
        return False


def salvar_historico_cnpj(run_id: str, df_full: pd.DataFrame) -> bool:
    """Grava o estado de cada CNPJ sacado nesta execução (histórico, base auxiliar+boletos)."""
    try:
        table = _dynamo_resource().Table(TABLE_CNPJ)
        with table.batch_writer() as batch:
            for _, r in df_full.iterrows():
                batch.put_item(Item={
                    'id_cnpj':           str(r['id_cnpj']),
                    'run_timestamp':     run_id,
                    'score_fidc':        int(r['score_fidc']),
                    'rating_carteira':   str(r['rating_carteira']),
                    'prob_ml_bom':       _to_decimal(r.get('prob_ml_bom', 0)),
                    'flag_risco_fraude': int(r.get('flag_risco_fraude', 0) or 0),
                })
        print(f"[SafeAsset] storage — {len(df_full)} CNPJs gravados no Dynamo (run {run_id})")
        return True
    except Exception as e:
        print(f"[SafeAsset] storage.salvar_historico_cnpj erro: {e}")
        return False


def salvar_historico_cedente(run_id: str, perfil_cedentes: pd.DataFrame,
                             origem: str = 'historico') -> bool:
    """Grava uma 'cessão' por beneficiário. origem: 'historico' ou 'carteira_nova'."""
    if perfil_cedentes is None or perfil_cedentes.empty:
        return False
    try:
        table = _dynamo_resource().Table(TABLE_CEDENTE)
        with table.batch_writer() as batch:
            for _, r in perfil_cedentes.iterrows():
                batch.put_item(Item={
                    'id_beneficiario':      str(r['id_beneficiario']),
                    'run_timestamp':        run_id,
                    'origem':               origem,
                    'qtd_cnpjs_cedidos':    int(r['qtd_cnpjs_cedidos']),
                    'vlr_total_cedido':     _to_decimal(r['vlr_total_cedido']),
                    'score_medio_carteira': _to_decimal(r['score_medio_carteira']),
                    'pct_rating_ab':        _to_decimal(r['pct_rating_ab']),
                    'pct_rating_de':        _to_decimal(r['pct_rating_de']),
                    'pct_flag_fraude':      _to_decimal(r['pct_flag_fraude']),
                    'prob_ml_media':        _to_decimal(r.get('prob_ml_media', 0)),
                })
        print(f"[SafeAsset] storage — {len(perfil_cedentes)} cedentes gravados "
              f"(origem={origem}, run {run_id})")
        return True
    except Exception as e:
        print(f"[SafeAsset] storage.salvar_historico_cedente erro: {e}")
        return False


def buscar_historico_cedente(id_beneficiario: str) -> pd.DataFrame:
    """Retorna todas as cessões registradas de um beneficiário, ordenadas no tempo."""
    try:
        table = _dynamo_resource().Table(TABLE_CEDENTE)
        resp  = table.query(
            KeyConditionExpression=Key('id_beneficiario').eq(str(id_beneficiario))
        )
        items = resp.get('Items', [])
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items).sort_values('run_timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"[SafeAsset] storage.buscar_historico_cedente erro: {e}")
        return pd.DataFrame()


def buscar_historico_cnpj(id_cnpj: str) -> pd.DataFrame:
    """Retorna todas as execuções registradas de um CNPJ sacado, ordenadas no tempo."""
    try:
        table = _dynamo_resource().Table(TABLE_CNPJ)
        resp  = table.query(
            KeyConditionExpression=Key('id_cnpj').eq(str(id_cnpj))
        )
        items = resp.get('Items', [])
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items).sort_values('run_timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"[SafeAsset] storage.buscar_historico_cnpj erro: {e}")
        return pd.DataFrame()