"""Definição dos conjuntos de features para experimentos A/B/C.

Cada experimento adiciona um nível de informação ao modelo XGBoost:

* **A_core**: variáveis hidroclimáticas básicas (pr, eto, q, Vol), lags
  imediatos e features de calendário.
* **B_external_ops**: A + variáveis operacionais externas roteadas pela
  bacia em questão (túneis, transposição PBS, elevatória).
* **C_hydro_candidates**: A + um conjunto parcimonioso de candidatos
  específicos por bacia (índices padronizados, rolling means, TWS), pré-
  selecionados por análise de correlação.
"""
from __future__ import annotations

from typing import Dict, List


def _ops_for_basin(basin: str) -> List[str]:
    """Retorna as variáveis operacionais externas relevantes para a bacia.

    Coerente com ``OpsRouting`` em
    :mod:`Previsao.Scripts.feature_constructor`.

    * qout/qbal são *locais* à bacia (DataFrame wide).
    * qt5/qt6/qt7, qpbs_net, qelev são *globais* e roteadas conforme a
      topologia operacional do Sistema Cantareira.

    Convenções para bacias agregadas
    --------------------------------
    * ``CT_CACH_ATIB`` (CACH+ATIB): qt6 vira interno; restam qt7 (in),
      qt5 (out), qpbs_net (in/out).
    * ``CT_TOTAL`` (JAGJAC+CACH+ATIB+PCASTR): qt5/qt6/qt7 são internos;
      restam apenas qpbs_net (in) e qelev (out).
    """
    base = ["qout", "qbal"]

    if basin == "CT_JAGJAC":
        return base + ["qt7"]
    if basin == "CT_CACH":
        return base + ["qt7", "qt6"]
    if basin == "CT_ATIB":
        return base + ["qt6", "qpbs_net", "qt5"]
    if basin == "CT_PCASTR":
        return base + ["qt5", "qelev"]
    if basin in ("CT_CACH_ATIB", "CT_CACH+CT_ATIB"):
        return base + ["qt7", "qpbs_net", "qt5"]
    if basin == "CT_TOTAL":
        return base + ["qpbs_net", "qelev"]
    return base


def _add_lags(cols: List[str], lags: List[int] | None) -> List[str]:
    """Acrescenta sufixos ``_Lag{k}`` às colunas indicadas."""
    if not lags:
        return cols
    out = cols.copy()
    for c in cols:
        for k in lags:
            out.append(f"{c}_Lag{k}")
    return out


def get_experiments_abc(
    *,
    basin: str,
    include_vol_lags: bool = True,
    ops_lags: List[int] | None = None,
    include_is_dry: bool = True,
) -> Dict[str, List[str]]:
    """Retorna o mapa nome→lista-de-features para os experimentos A/B/C.

    Parameters
    ----------
    basin : str
        Código da bacia. Define o roteamento operacional (B) e o set de
        candidatos (C).
    include_vol_lags : bool
        Se True, adiciona ``Vol_Lag1`` à composição A.
    ops_lags : list[int] | None
        Lags adicionais a aplicar nas variáveis operacionais de B.
        Default: ``[1]``.
    include_is_dry : bool
        Se True, inclui o flag de estação seca (Abr–Set) gerado em
        :meth:`XGBoostModel.add_calendar_features`.

    Returns
    -------
    dict[str, list[str]]
        Mapa com chaves ``A_core_raw``, ``B_core_raw_plus_ops`` e
        ``C_core_raw_plus_candidates``.

    Notas
    -----
    Para ``CT_TOTAL``, o set C combina os melhores candidatos
    encontrados nos benchmarks de CT_JAGJAC (SRI12, eto_RollingMean12) e
    CT_CACH_ATIB (q_RollingMean6, eto_ExpandingMean), todos sobre o
    sistema agregado.
    """
    if ops_lags is None:
        ops_lags = [1]

    vol_lags = ["Vol_Lag1"] if include_vol_lags else []
    calendar = ["month_sin", "month_cos"] + (
        ["is_dry"] if include_is_dry else []
    )

    A = [
        "pr",
        "eto",
        "q",
        "Vol",
        "pr_Lag1",
        "eto_Lag1",
        "q_Lag1",
        *vol_lags,
        *calendar,
    ]

    ops_raw = _ops_for_basin(basin)
    B = A[:-len(calendar)] + _add_lags(ops_raw, ops_lags) + A[-len(calendar):]

    if basin == "CT_JAGJAC":
        C_add = ["SRI12", "eto_RollingMean12", "tws"]
    elif basin in ("CT_CACH_ATIB", "CT_CACH+CT_ATIB"):
        C_add = ["q_RollingMean6", "tws", "SRI12", "eto_ExpandingMean"]
    elif basin == "CT_TOTAL":
        # Combinação dos melhores candidatos dos benchmarks anteriores,
        # aplicada agora sobre o sistema agregado.
        C_add = [
            "SRI12",
            "eto_RollingMean12",
            "q_RollingMean6",
            "eto_ExpandingMean",
            "tws",
        ]
    else:
        C_add = ops_raw

    C = A[:-len(calendar)] + C_add + A[-len(calendar):]

    return {
        "A_core_raw": A,
        "B_core_raw_plus_ops": B,
        "C_core_raw_plus_candidates": C,
    }
