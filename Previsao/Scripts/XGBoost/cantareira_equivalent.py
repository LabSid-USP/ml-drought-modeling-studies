"""
Agregação de bacias do Sistema Cantareira em bacias equivalentes.

Este módulo oferece duas frentes complementares:

1. **Agregação de pares** (legado e mantida): ``BasinAggregationSpec`` +
   ``build_equivalent_cantareira_inputs()`` agregam duas bacias em uma
   equivalente. Uso típico: CT_CACH + CT_ATIB → CT_CACH_ATIB.

2. **Agregação multi-bacia** (CT_TOTAL): ``MultiBasinAggregationSpec`` +
   ``build_total_cantareira_inputs()`` agregam um número arbitrário de bacias
   em uma única equivalente. Uso típico: CT_JAGJAC + CT_CACH + CT_ATIB +
   CT_PCASTR → CT_TOTAL (todo o Sistema Cantareira).

Notas físicas
-------------
- Vazões em m³/s (Qnat, Qbal, Qout): soma é fisicamente significativa quando
  agregam-se sub-bacias em um nó único.
- Armazenamentos em hm³ (Vol): soma é direta.
- Variáveis em lâmina (mm) como Pr, ETo e TWS: somar **não** é correto;
  utilizar **média ponderada por área** das bacias incrementais.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import pandas as pd


AggMode = Literal["sum", "mean", "area_weighted_mean"]


# --------------------------------------------------------------------------- #
# Constantes — áreas de drenagem das bacias incrementais (km²)
# --------------------------------------------------------------------------- #
# Fonte primária: estimativa por feições oficiais georreferenciadas via QGIS.
# Fonte secundária: PDAA — Plano Diretor de Abastecimento de Água da RMSP.
# Os valores abaixo são os do QGIS (maior precisão geométrica).
CANTAREIRA_AREA_KM2: dict[str, float] = {
    "CT_JAGJAC": 1239.25,
    "CT_CACH": 392.38,
    "CT_ATIB": 315.07,
    "CT_PCASTR": 337.57,
}

# Áreas conforme PDAA (mantidas para conferência/auditoria documental).
CANTAREIRA_AREA_KM2_PDAA: dict[str, float] = {
    "CT_JAGJAC": 1230.0,
    "CT_CACH": 392.0,
    "CT_ATIB": 312.0,
    "CT_PCASTR": 338.0,
}


# --------------------------------------------------------------------------- #
# Agregação de PAR de bacias (legado: CT_CACH_ATIB)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BasinAggregationSpec:
    """Especificação para agregar duas bacias em uma equivalente.

    Parameters
    ----------
    basin_a, basin_b : str
        Códigos das bacias originais.
    new_basin : str
        Código da bacia equivalente resultante.
    pr_mode, eto_mode, tws_mode : AggMode
        Modo de agregação para variáveis em mm (default: mean ou
        area_weighted_mean quando áreas são fornecidas).
    qnat_mode, qbal_mode, qout_mode, vol_mode : AggMode
        Modo de agregação para fluxos (m³/s) e armazenamentos (hm³),
        tipicamente "sum".
    area_a, area_b : float | None
        Áreas das bacias originais (km² ou unidade consistente). Necessárias
        apenas quando algum modo for ``area_weighted_mean``.
    """
    basin_a: str = "CT_CACH"
    basin_b: str = "CT_ATIB"
    new_basin: str = "CT_CACH_ATIB"

    pr_mode: AggMode = "mean"
    eto_mode: AggMode = "mean"
    qnat_mode: AggMode = "sum"
    qbal_mode: AggMode = "sum"
    qout_mode: AggMode = "sum"
    vol_mode: AggMode = "sum"
    tws_mode: AggMode = "mean"

    area_a: float | None = None
    area_b: float | None = None


def default_cach_atib_spec_mm(
    new_basin: str = "CT_CACH_ATIB",
) -> BasinAggregationSpec:
    """Spec padrão para CT_CACH + CT_ATIB com Pr/ETo em mm.

    Usa **média ponderada por área** para variáveis em lâmina (Pr/ETo) e
    **soma** para fluxos (Q) e armazenamentos (Vol). Recomendado quando
    Pr/ETo são profundidades médias mensais por bacia em mm.
    """
    return BasinAggregationSpec(
        basin_a="CT_CACH",
        basin_b="CT_ATIB",
        new_basin=new_basin,
        pr_mode="area_weighted_mean",
        eto_mode="area_weighted_mean",
        qnat_mode="sum",
        qbal_mode="sum",
        qout_mode="sum",
        vol_mode="sum",
        tws_mode="area_weighted_mean",
        area_a=CANTAREIRA_AREA_KM2["CT_CACH"],
        area_b=CANTAREIRA_AREA_KM2["CT_ATIB"],
    )


def _standardize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    return out


def _agg_series(
    s_a: pd.Series,
    s_b: pd.Series,
    mode: AggMode,
    area_a: float | None = None,
    area_b: float | None = None,
) -> pd.Series:
    if mode == "sum":
        return s_a + s_b
    if mode == "mean":
        return 0.5 * (s_a + s_b)
    if mode == "area_weighted_mean":
        if area_a is None or area_b is None:
            raise ValueError(
                "area_a e area_b devem ser fornecidos para area_weighted_mean."
            )
        w = area_a + area_b
        return (s_a * area_a + s_b * area_b) / w
    raise ValueError(f"Modo de agregação não suportado: {mode}")


def aggregate_two_basins_wide(
    df: pd.DataFrame,
    spec: BasinAggregationSpec,
    *,
    mode: AggMode,
) -> pd.DataFrame:
    """Agrega duas colunas de bacia em uma nova coluna num DataFrame wide."""
    d = _standardize_monthly(df)
    if spec.basin_a not in d.columns or spec.basin_b not in d.columns:
        missing = [c for c in (spec.basin_a, spec.basin_b)
                   if c not in d.columns]
        raise KeyError(f"Colunas de bacia ausentes no DataFrame: {missing}")

    out = d.copy()
    out[spec.new_basin] = _agg_series(
        out[spec.basin_a],
        out[spec.basin_b],
        mode=mode,
        area_a=spec.area_a,
        area_b=spec.area_b,
    )
    return out


def build_equivalent_cantareira_inputs(
    *,
    pr: pd.DataFrame,
    eto: pd.DataFrame,
    qnat: pd.DataFrame,
    vol: pd.DataFrame,
    qbal: pd.DataFrame | None = None,
    qout: pd.DataFrame | None = None,
    tws: pd.DataFrame | None = None,
    keep_basins: list[str] | None = None,
    spec: BasinAggregationSpec | None = None,
    drop_original_pair: bool = True,
) -> dict[str, pd.DataFrame]:
    """Cria datasets equivalentes com uma bacia agregada (par → uma).

    Uso típico: agregar CT_CACH + CT_ATIB → CT_CACH_ATIB, mantendo
    CT_JAGJAC inalterado.

    Parameters
    ----------
    pr, eto, qnat, vol : pd.DataFrame
        DataFrames wide (date + colunas por bacia) das variáveis básicas.
    qbal, qout, tws : pd.DataFrame, opcional
        DataFrames wide adicionais a serem agregados.
    keep_basins : list[str], opcional
        Bacias que devem ser preservadas além da equivalente (default:
        ``["CT_JAGJAC"]``).
    spec : BasinAggregationSpec, opcional
        Configuração da agregação. Se None, usa o default
        (``CT_CACH + CT_ATIB``).
    drop_original_pair : bool
        Se True, remove (basin_a, basin_b) dos outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapa com chaves pr, eto, qnat, vol, qbal, qout, tws
        (quando fornecidos).
    """
    if spec is None:
        spec = BasinAggregationSpec()

    if keep_basins is None:
        keep_basins = ["CT_JAGJAC"]

    out: dict[str, pd.DataFrame] = {}

    pr2 = aggregate_two_basins_wide(pr, spec, mode=spec.pr_mode)
    eto2 = aggregate_two_basins_wide(eto, spec, mode=spec.eto_mode)
    qnat2 = aggregate_two_basins_wide(qnat, spec, mode=spec.qnat_mode)
    vol2 = aggregate_two_basins_wide(vol, spec, mode=spec.vol_mode)

    out["pr"] = pr2
    out["eto"] = eto2
    out["qnat"] = qnat2
    out["vol"] = vol2

    if qbal is not None:
        out["qbal"] = aggregate_two_basins_wide(
            qbal, spec, mode=spec.qbal_mode)
    if qout is not None:
        out["qout"] = aggregate_two_basins_wide(
            qout, spec, mode=spec.qout_mode)
    if tws is not None:
        out["tws"] = aggregate_two_basins_wide(tws, spec, mode=spec.tws_mode)

    keep_cols = ["date"] + keep_basins + [spec.new_basin]
    for k, df in list(out.items()):
        cols = [c for c in keep_cols if c in df.columns]
        out[k] = df[cols].copy()

        if drop_original_pair:
            out[k] = out[k].drop(
                columns=[c for c in (spec.basin_a, spec.basin_b)
                         if c in out[k].columns])

    return out


# --------------------------------------------------------------------------- #
# Agregação MULTI-BACIA (CT_TOTAL = todo o Sistema Cantareira)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MultiBasinAggregationSpec:
    """Especificação para agregar N bacias em uma equivalente única.

    Modos de agregação seguem a mesma convenção física da spec de par:
    soma para fluxos/armazenamentos, média ponderada por área para lâminas.

    Parameters
    ----------
    basins : tuple[str, ...]
        Códigos das bacias a serem agregadas (>= 2).
    new_basin : str
        Código da bacia equivalente resultante.
    areas_km2 : dict[str, float]
        Áreas de drenagem incrementais (km²) por bacia. Usadas pelas
        agregações em modo ``area_weighted_mean``. Deve conter todas as
        bacias em ``basins``.
    pr_mode, eto_mode, tws_mode : AggMode
        Modo de agregação para variáveis em mm (default:
        ``area_weighted_mean``).
    qnat_mode, qbal_mode, qout_mode, vol_mode : AggMode
        Modo de agregação para fluxos e armazenamentos (default: ``sum``).
    """
    basins: tuple[str, ...] = ("CT_JAGJAC", "CT_CACH", "CT_ATIB", "CT_PCASTR")
    new_basin: str = "CT_TOTAL"
    areas_km2: dict[str, float] = field(
        default_factory=lambda: dict(CANTAREIRA_AREA_KM2))

    pr_mode: AggMode = "area_weighted_mean"
    eto_mode: AggMode = "area_weighted_mean"
    tws_mode: AggMode = "area_weighted_mean"
    qnat_mode: AggMode = "sum"
    qbal_mode: AggMode = "sum"
    qout_mode: AggMode = "sum"
    vol_mode: AggMode = "sum"


def default_cantareira_total_spec(
    new_basin: str = "CT_TOTAL",
    basins: Sequence[str] = ("CT_JAGJAC", "CT_CACH", "CT_ATIB", "CT_PCASTR"),
) -> MultiBasinAggregationSpec:
    """Spec padrão para CT_TOTAL = JAGJAC + CACH + ATIB + PCASTR.

    Usa áreas QGIS (``CANTAREIRA_AREA_KM2``) e os modos físicos canônicos:
    soma para Q/Vol; média ponderada por área para Pr/ETo/TWS.
    """
    basins_tuple = tuple(basins)
    return MultiBasinAggregationSpec(
        basins=basins_tuple,
        new_basin=new_basin,
        areas_km2={b: CANTAREIRA_AREA_KM2[b] for b in basins_tuple},
    )


def _agg_many(
    df: pd.DataFrame,
    basins: Sequence[str],
    mode: AggMode,
    areas_km2: dict[str, float] | None = None,
) -> pd.Series:
    """Agrega N colunas (uma por bacia) em uma única série."""
    cols = [df[b] for b in basins]
    if mode == "sum":
        return sum(cols)  # type: ignore
    if mode == "mean":
        return sum(cols) / len(cols)  # type: ignore
    if mode == "area_weighted_mean":
        if areas_km2 is None:
            raise ValueError(
                "areas_km2 deve ser fornecido para area_weighted_mean.")
        missing = [b for b in basins if b not in areas_km2]
        if missing:
            raise ValueError(
                f"Áreas ausentes para bacias: {missing}")
        weights = [areas_km2[b] for b in basins]
        total_area = sum(weights)
        weighted_sum = sum(df[b] * w for b, w in zip(basins, weights))
        return weighted_sum / total_area  # type: ignore
    raise ValueError(f"Modo de agregação não suportado: {mode}")


def aggregate_many_basins_wide(
    df: pd.DataFrame,
    basins: Sequence[str],
    new_basin: str,
    *,
    mode: AggMode,
    areas_km2: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Agrega N colunas de bacia em uma nova coluna num DataFrame wide.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna ``date`` e uma coluna por bacia.
    basins : sequence of str
        Bacias a serem agregadas (devem existir como colunas em ``df``).
    new_basin : str
        Nome da coluna resultante.
    mode : AggMode
        Modo de agregação ("sum", "mean", "area_weighted_mean").
    areas_km2 : dict[str, float], opcional
        Áreas usadas no modo ``area_weighted_mean``.
    """
    d = _standardize_monthly(df)
    missing = [b for b in basins if b not in d.columns]
    if missing:
        raise KeyError(f"Colunas de bacia ausentes no DataFrame: {missing}")

    out = d.copy()
    out[new_basin] = _agg_many(out, basins, mode=mode, areas_km2=areas_km2)
    return out


def build_total_cantareira_inputs(
    *,
    pr: pd.DataFrame,
    eto: pd.DataFrame,
    qnat: pd.DataFrame,
    vol: pd.DataFrame,
    qbal: pd.DataFrame | None = None,
    qout: pd.DataFrame | None = None,
    tws: pd.DataFrame | None = None,
    spec: MultiBasinAggregationSpec | None = None,
    drop_original: bool = True,
) -> dict[str, pd.DataFrame]:
    """Cria datasets com uma única bacia agregada (Cantareira inteiro).

    Uso típico: agregar JAGJAC + CACH + ATIB + PCASTR → CT_TOTAL.

    Parameters
    ----------
    pr, eto, qnat, vol : pd.DataFrame
        DataFrames wide (date + colunas por bacia) das variáveis básicas.
    qbal, qout, tws : pd.DataFrame, opcional
        DataFrames wide adicionais a serem agregados.
    spec : MultiBasinAggregationSpec, opcional
        Configuração da agregação. Default: agregação dos 4 reservatórios.
    drop_original : bool
        Se True, remove as colunas originais das bacias, deixando apenas
        ``date`` e ``new_basin``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapa com chaves pr, eto, qnat, vol, qbal, qout, tws
        (quando fornecidos).
    """
    if spec is None:
        spec = default_cantareira_total_spec()

    out: dict[str, pd.DataFrame] = {}

    out["pr"] = aggregate_many_basins_wide(
        pr, spec.basins, spec.new_basin,
        mode=spec.pr_mode, areas_km2=spec.areas_km2)
    out["eto"] = aggregate_many_basins_wide(
        eto, spec.basins, spec.new_basin,
        mode=spec.eto_mode, areas_km2=spec.areas_km2)
    out["qnat"] = aggregate_many_basins_wide(
        qnat, spec.basins, spec.new_basin,
        mode=spec.qnat_mode, areas_km2=spec.areas_km2)
    out["vol"] = aggregate_many_basins_wide(
        vol, spec.basins, spec.new_basin,
        mode=spec.vol_mode, areas_km2=spec.areas_km2)

    if qbal is not None:
        out["qbal"] = aggregate_many_basins_wide(
            qbal, spec.basins, spec.new_basin,
            mode=spec.qbal_mode, areas_km2=spec.areas_km2)
    if qout is not None:
        out["qout"] = aggregate_many_basins_wide(
            qout, spec.basins, spec.new_basin,
            mode=spec.qout_mode, areas_km2=spec.areas_km2)
    if tws is not None:
        out["tws"] = aggregate_many_basins_wide(
            tws, spec.basins, spec.new_basin,
            mode=spec.tws_mode, areas_km2=spec.areas_km2)

    if drop_original:
        keep_cols = ["date", spec.new_basin]
        for k, df in list(out.items()):
            cols = [c for c in keep_cols if c in df.columns]
            out[k] = df[cols].copy()

    return out


def area_share(
    basins: Iterable[str],
    areas_km2: dict[str, float] | None = None,
) -> dict[str, float]:
    """Retorna a fração de área (0..1) de cada bacia em relação ao total.

    Útil para anotar plots e explicar pesos da agregação ponderada por área.
    """
    if areas_km2 is None:
        areas_km2 = CANTAREIRA_AREA_KM2
    basins = list(basins)
    total = sum(areas_km2[b] for b in basins)
    return {b: areas_km2[b] / total for b in basins}
