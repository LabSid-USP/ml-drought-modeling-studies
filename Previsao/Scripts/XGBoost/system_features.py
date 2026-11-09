"""Features de *estado sistêmico* para bacias parciais do Cantareira.

Escopo
------
Usado quando a bacia-alvo NÃO é o Cantareira inteiro (ex.: ``CT_CACH_ATIB``),
para dar ao modelo informação de armazenamento agregado a montante. Não é
necessário (nem usado) quando a bacia-alvo já é ``CT_TOTAL``, pois o estado
sistêmico já é a própria variável-alvo.
"""
from __future__ import annotations

import pandas as pd


def add_cantareira_system_storage_features(
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    *,
    target_basin: str = "CT_CACH_ATIB",
    upstream_basin: str = "CT_JAGJAC",
    out_upstream_col: str = "Vol_JAGJAC",
    out_total_col: str = "Vol_total",
) -> dict[str, pd.DataFrame]:
    """Adiciona features de estado sistêmico ao dataset de uma bacia-alvo.

    Para ``CT_CACH_ATIB`` (típico):

    * ``Vol_JAGJAC``: volume atual do reservatório a montante.
    * ``Vol_total`` : ``Vol_JAGJAC + Vol_CT_CACH_ATIB``.

    As features são calculadas no tempo ``t``, portanto compatíveis com a
    formulação supervisionada ``X(t) -> y(t+H)`` (não introduz vazamento).
    """
    out = {k: v.copy() for k, v in datasets_by_basin.items()}

    if target_basin not in out:
        raise KeyError(f"{target_basin!r} não existe em datasets_by_basin.")

    required = ["date", upstream_basin, target_basin]
    missing = [c for c in required if c not in volumes.columns]
    if missing:
        raise KeyError(f"Colunas ausentes em volumes: {missing}")

    v = volumes[required].copy()
    v["date"] = pd.to_datetime(v["date"])

    v = v.rename(columns={upstream_basin: out_upstream_col})
    v[out_total_col] = v[out_upstream_col] + v[target_basin]

    add_cols = v[["date", out_upstream_col, out_total_col]]

    out[target_basin] = out[target_basin].merge(
        add_cols,
        on="date",
        how="left",
    )

    return out
