from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .feature_engineering import FeatureEngineering


@dataclass(frozen=True)
class OpsRouting:
    """
    Define quais séries operacionais entram em cada reservatório.

    Convenção:
    - qt7: túnel 7 (JAGJAC -> CACH)
    - qt6: túnel 6 (CACH -> ATIB)
    - qt5: túnel 5 (ATIB -> PCASTR)
    - qpbs_net: (PBS -> ATIB) - (ATIB -> PBS)
    - qelev: elevatória Santa Inês (saída de PCASTR)
    """
    use_qt7: bool = False
    use_qt6: bool = False
    use_qt5: bool = False
    use_qpbs_net: bool = False
    use_qelev: bool = False


class FeatureConstructor:
    """
    Constrói features por bacia a partir de DataFrames wide e devolve:

    - features: dicionário hierárquico com blocos intermediários
    - datasets_by_basin: "tabelão" flat (uma linha por mês) por bacia

    Variáveis hidroclimáticas (por bacia)
    ------------------------------------
    - pr   : precipitação (mensal)
    - eto  : evapotranspiração de referência (mensal)
    - q    : vazão natural (Qnat) (mensal)

    Variáveis operacionais (por bacia)
    ---------------------------------
    - qout : vazão a jusante (despacho)
    - qbal : balanço operacional (m³/s), conforme equações do Cantareira
    - tws  : Total Water Storage (GRACE) (opcional; pode ter datas faltantes)

    Variáveis operacionais globais (roteadas por bacia)
    ---------------------------------------------------
    - qt7      : túnel 7 (aplica em CT_JAGJAC e CT_CACH)
    - qt6      : túnel 6 (aplica em CT_CACH e CT_ATIB)
    - qt5      : túnel 5 (aplica em CT_ATIB e CT_PCASTR)
    - qpbs_net : PBS líquido (aplica em CT_ATIB, CT_CACH_ATIB, CT_TOTAL)
    - qelev    : elevatória Santa Inês (aplica em CT_PCASTR, CT_TOTAL)

    Bacias equivalentes / agregadas
    -------------------------------
    - CT_CACH_ATIB: agrega CACH+ATIB. qt6 vira interno; restam qt7 (in),
      qt5 (out) e qpbs_net (in/out).
    - CT_TOTAL: agrega JAGJAC+CACH+ATIB+PCASTR. qt5/qt6/qt7 são internos;
      restam qpbs_net (in) e qelev (out).

    Observação importante
    ---------------------
    Algumas séries (ex.: TWS) podem ter meses ausentes. Para evitar que
    lags/rolling sejam calculados em uma grade temporal irregular, este
    construtor alinha todas as séries à grade mensal ("MS") do pr.
    """

    def __init__(
        self,
        basins: list[str],
        lags: list[int],
        roll_windows: list[int],
        index_scales: list[int],
        stats_ops: list[str] | None = None,
    ) -> None:
        self.basins = basins
        self.lags = lags
        self.roll_windows = roll_windows
        self.index_scales = index_scales
        self.stats_ops = stats_ops if stats_ops is not None else ["Mean"]

        allowed = {"Mean", "Sum", "Std"}
        invalid = set(self.stats_ops) - allowed
        if invalid:
            raise ValueError(
                f"stats_ops inválido: {sorted(invalid)}. "
                f"Use apenas {sorted(allowed)}."
            )

        self.fe = FeatureEngineering()

        # Regras de roteamento operacional do Cantareira.
        # Para cada bacia, sinalizam quais séries operacionais EXTERNAS
        # entram como features. Túneis "internos" a uma agregação não
        # aparecem (ex: para CT_CACH_ATIB, qt6 é interno; para CT_TOTAL,
        # qt5/qt6/qt7 são todos internos).
        self.ops_routing: dict[str, OpsRouting] = {
            "CT_JAGJAC": OpsRouting(use_qt7=True),
            "CT_CACH": OpsRouting(use_qt7=True, use_qt6=True),
            "CT_ATIB": OpsRouting(
                use_qt6=True, use_qt5=True, use_qpbs_net=True
            ),
            "CT_CACH_ATIB": OpsRouting(
                use_qt7=True,
                use_qt5=True,
                use_qpbs_net=True
            ),
            "CT_CACH+CT_ATIB": OpsRouting(
                use_qt7=True,
                use_qt5=True,
                use_qpbs_net=True
            ),
            "CT_PCASTR": OpsRouting(use_qt5=True, use_qelev=True),
            # CT_TOTAL = JAGJAC + CACH + ATIB + PCASTR. Túneis internos
            # qt5/qt6/qt7 deixam de ser fronteira; restam apenas a
            # transposição PBS (entrada) e o bombeamento Santa Inês (saída).
            "CT_TOTAL": OpsRouting(use_qpbs_net=True, use_qelev=True),
        }

    # ------------------------------------------------------------------ #
    # Helpers básicos
    # ------------------------------------------------------------------ #
    def to_long(
        self,
        df: pd.DataFrame,
        basin: str,
        var_name: str,
    ) -> pd.DataFrame:
        """
        Recorta uma bacia do DataFrame wide e devolve um DataFrame longo.

        Retorna colunas: ['date', var_name].
        """
        out = df[["date", basin]].copy()
        out = out.rename(columns={basin: var_name})
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date")
        return out

    def _to_long_single(
        self,
        df: pd.DataFrame,
        value_col: str,
        out_name: str,
    ) -> pd.DataFrame:
        """
        Converte um DataFrame com coluna única (além de date) para longo.

        Ex.: Qt5_monthly.csv tem colunas ['date', 'Q_T5'].
        """
        out = df[["date", value_col]].copy()
        out = out.rename(columns={value_col: out_name})
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date")
        return out

    def _monthly_grid(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        """Cria uma grade mensal MS entre start e end (inclusive)."""
        start = pd.to_datetime(start).to_period("M").to_timestamp(how="start")
        end = pd.to_datetime(end).to_period("M").to_timestamp(how="start")
        return pd.date_range(start=start, end=end, freq="MS")

    def _align_to_grid(
        self,
        df_long: pd.DataFrame,
        grid: pd.DatetimeIndex,
        value_col: str,
    ) -> pd.DataFrame:
        """
        Alinha uma série longa a uma grade mensal.

        Isso garante que shift/rolling representem lags/janelas em meses,
        mesmo quando o dado original tem datas faltantes.
        """
        base = pd.DataFrame({"date": grid})
        out = base.merge(df_long[["date", value_col]], on="date", how="left")
        out = out.sort_values("date")
        return out

    def _rolling_block(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        out = df[["date", value_col]].copy()
        for op in self.stats_ops:
            tmp = self.fe.rolling_features(
                df.copy(), value_col, op, self.roll_windows
            )
            tmp = tmp.drop(columns=[value_col])
            out = out.merge(tmp, on="date", how="left")
        return out

    def _expanding_block(
        self, df: pd.DataFrame, value_col: str
    ) -> pd.DataFrame:
        out = df[["date", value_col]].copy()
        for op in self.stats_ops:
            tmp = self.fe.expanding_features(df.copy(), value_col, op)
            tmp = tmp.drop(columns=[value_col])
            out = out.merge(tmp, on="date", how="left")
        return out

    def _blocks_generic(
        self,
        df: pd.DataFrame,
        var: str,
        include_expanding: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Blocos genéricos: lag/rolling/expanding para variáveis contínuas.

        Não cria índice padronizado (SPI/SPEI/SRI) — apenas em
        pr/eto/q.
        """
        blocks: dict[str, pd.DataFrame] = {}
        blocks["lag"] = self.fe.lag_features(df.copy(), var, self.lags)
        blocks["rolling"] = self._rolling_block(df, var)
        if include_expanding:
            blocks["expanding"] = self._expanding_block(df, var)
        return blocks

    # ------------------------------------------------------------------ #
    # Blocos hidroclimáticos (com índices)
    # ------------------------------------------------------------------ #
    def add_pr_features(self, df_pr: pd.DataFrame) -> dict[str, pd.DataFrame]:
        blocks: dict[str, pd.DataFrame] = {}
        blocks["lag"] = self.fe.lag_features(df_pr.copy(), "pr", self.lags)
        blocks["rolling"] = self._rolling_block(df_pr, "pr")
        blocks["expanding"] = self._expanding_block(df_pr, "pr")

        spi_df = df_pr[["date", "pr"]].copy()
        for scale in self.index_scales:
            spi_df = self.fe.spi(
                spi_df,
                date_col="date",
                pr_col="pr",
                scale=scale,
                out_col=f"SPI{scale}",
            )
        blocks["spi"] = spi_df
        return blocks

    def add_eto_features(
        self,
        df_eto: pd.DataFrame,
        df_pr: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        blocks: dict[str, pd.DataFrame] = {}
        blocks["lag"] = self.fe.lag_features(df_eto.copy(), "eto", self.lags)
        blocks["rolling"] = self._rolling_block(df_eto, "eto")
        blocks["expanding"] = self._expanding_block(df_eto, "eto")

        spei_df = df_eto[["date", "eto"]].copy()
        for scale in self.index_scales:
            tmp = self.fe.spei(
                df_pr[["date", "pr"]].copy(),
                df_eto[["date", "eto"]].copy(),
                date_col="date",
                pr_col="pr",
                eto_col="eto",
                scale=scale,
                out_col=f"SPEI{scale}",
            )
            spei_df = spei_df.merge(
                tmp[["date", f"SPEI{scale}"]],
                on="date",
                how="left",
            )

        blocks["spei"] = spei_df
        return blocks

    def add_q_features(self, df_q: pd.DataFrame) -> dict[str, pd.DataFrame]:
        blocks: dict[str, pd.DataFrame] = {}
        blocks["lag"] = self.fe.lag_features(df_q.copy(), "q", self.lags)
        blocks["rolling"] = self._rolling_block(df_q, "q")
        blocks["expanding"] = self._expanding_block(df_q, "q")

        sri_df = df_q[["date", "q"]].copy()
        for scale in self.index_scales:
            sri_df = self.fe.sri(
                sri_df,
                date_col="date",
                q_col="q",
                scale=scale,
                out_col=f"SRI{scale}",
            )
        blocks["sri"] = sri_df
        return blocks

    # ------------------------------------------------------------------ #
    # Build principal
    # ------------------------------------------------------------------ #
    def build(
        self,
        pr: pd.DataFrame,
        eto: pd.DataFrame,
        qnat: pd.DataFrame,
        qbal: Optional[pd.DataFrame] = None,
        qout: Optional[pd.DataFrame] = None,
        tws: Optional[pd.DataFrame] = None,
        qt5: Optional[pd.DataFrame] = None,
        qt6: Optional[pd.DataFrame] = None,
        qt7: Optional[pd.DataFrame] = None,
        qpbs_net: Optional[pd.DataFrame] = None,
        qelev: Optional[pd.DataFrame] = None,
        *,
        qt5_col: str = "Q_T5",
        qt6_col: str = "Q_T6",
        qt7_col: str = "Q_T7",
        qpbs_net_col: str = "PBS_NET",
        qelev_col: str = "ST_INES",
    ) -> tuple[
        dict[str, dict[str, dict[str, pd.DataFrame]]],
        dict[str, pd.DataFrame],
    ]:
        """
        Constrói (features, datasets_by_basin) para todas as bacias.

        Parâmetros
        ----------
        pr, eto, qnat : pd.DataFrame
            Wide com colunas ['date', <bacias...>].
        qbal, qout, tws : pd.DataFrame | None
            Wide com colunas ['date', <bacias...>]. `tws` pode ter meses
            ausentes (o alinhamento mensal é feito internamente).
        qt5, qt6, qt7 : pd.DataFrame | None
            Uma coluna + date: ['date', Q_T5] etc.
        qpbs_net, qelev : pd.DataFrame | None
            Uma coluna + date: ['date', PBS_NET], ['date', ST_INES].

        Retorna
        -------
        features : dict
            features[basin][var][block] -> pd.DataFrame
        datasets_by_basin : dict
            datasets_by_basin[basin] -> pd.DataFrame flat
        """
        features: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        datasets_by_basin: dict[str, pd.DataFrame] = {}

        # Pré-processa séries globais (single-column) uma vez
        df_qt5_raw = (
            self._to_long_single(qt5, qt5_col, "qt5")
            if qt5 is not None
            else None
        )
        df_qt6_raw = (
            self._to_long_single(qt6, qt6_col, "qt6")
            if qt6 is not None
            else None
        )
        df_qt7_raw = (
            self._to_long_single(qt7, qt7_col, "qt7")
            if qt7 is not None
            else None
        )
        df_qpbs_raw = (
            self._to_long_single(qpbs_net, qpbs_net_col, "qpbs_net")
            if qpbs_net is not None
            else None
        )
        df_qelev_raw = (
            self._to_long_single(qelev, qelev_col, "qelev")
            if qelev is not None
            else None
        )

        for basin in self.basins:
            routing = self.ops_routing.get(basin, OpsRouting())

            # Hidroclima (por bacia)
            df_pr_raw = self.to_long(pr, basin, "pr")
            grid = self._monthly_grid(
                df_pr_raw["date"].min(), df_pr_raw["date"].max()
            )

            df_pr = self._align_to_grid(df_pr_raw, grid, "pr")
            df_eto = self._align_to_grid(
                self.to_long(eto, basin, "eto"), grid, "eto"
            )
            df_q = self._align_to_grid(
                self.to_long(qnat, basin, "q"), grid, "q"
            )

            pr_blocks = self.add_pr_features(df_pr)
            eto_blocks = self.add_eto_features(df_eto, df_pr)
            q_blocks = self.add_q_features(df_q)

            basin_features: dict[str, dict[str, pd.DataFrame]] = {
                "pr": pr_blocks,
                "eto": eto_blocks,
                "q": q_blocks,
            }

            # Operação por bacia
            df_qout = None
            if qout is not None:
                df_qout = self._align_to_grid(
                    self.to_long(qout, basin, "qout"), grid, "qout"
                )
                basin_features["qout"] = self._blocks_generic(df_qout, "qout")

            df_qbal = None
            if qbal is not None:
                df_qbal = self._align_to_grid(
                    self.to_long(qbal, basin, "qbal"), grid, "qbal"
                )
                basin_features["qbal"] = self._blocks_generic(df_qbal, "qbal")

            df_tws = None
            if tws is not None:
                df_tws = self._align_to_grid(
                    self.to_long(tws, basin, "tws"), grid, "tws"
                )
                basin_features["tws"] = self._blocks_generic(df_tws, "tws")

            # Operação global roteada (alinha à grade por bacia)
            df_qt7 = None
            if routing.use_qt7 and df_qt7_raw is not None:
                df_qt7 = self._align_to_grid(df_qt7_raw, grid, "qt7")
                basin_features["qt7"] = self._blocks_generic(df_qt7, "qt7")

            df_qt6 = None
            if routing.use_qt6 and df_qt6_raw is not None:
                df_qt6 = self._align_to_grid(df_qt6_raw, grid, "qt6")
                basin_features["qt6"] = self._blocks_generic(df_qt6, "qt6")

            df_qt5 = None
            if routing.use_qt5 and df_qt5_raw is not None:
                df_qt5 = self._align_to_grid(df_qt5_raw, grid, "qt5")
                basin_features["qt5"] = self._blocks_generic(df_qt5, "qt5")

            df_qpbs = None
            if routing.use_qpbs_net and df_qpbs_raw is not None:
                df_qpbs = self._align_to_grid(
                    df_qpbs_raw, grid, "qpbs_net"
                )
                basin_features["qpbs_net"] = self._blocks_generic(
                    df_qpbs, "qpbs_net"
                )

            df_qelev = None
            if routing.use_qelev and df_qelev_raw is not None:
                df_qelev = self._align_to_grid(df_qelev_raw, grid, "qelev")
                basin_features["qelev"] = self._blocks_generic(
                    df_qelev, "qelev"
                )

            features[basin] = basin_features

            # -------------------- Flat dataset -------------------- #
            base = df_pr[["date"]].copy()

            pr_all = (
                df_pr[["date", "pr"]]
                .merge(
                    pr_blocks["lag"].drop(columns=["pr"]),
                    on="date",
                    how="left",
                )
                .merge(
                    pr_blocks["rolling"].drop(columns=["pr"]),
                    on="date",
                    how="left"
                )
                .merge(
                    pr_blocks["expanding"].drop(columns=["pr"]),
                    on="date",
                    how="left"
                )
                .merge(
                    pr_blocks["spi"].drop(columns=["pr"]),
                    on="date",
                    how="left"
                )
            )

            eto_all = (
                df_eto[["date", "eto"]]
                .merge(
                    eto_blocks["lag"].drop(columns=["eto"]),
                    on="date",
                    how="left"
                )
                .merge(
                    eto_blocks["rolling"].drop(columns=["eto"]),
                    on="date",
                    how="left"
                )
                .merge(
                    eto_blocks["expanding"].drop(columns=["eto"]),
                    on="date",
                    how="left"
                )
                .merge(
                    eto_blocks["spei"].drop(columns=["eto"]),
                    on="date",
                    how="left"
                )
            )

            q_all = (
                df_q[["date", "q"]]
                .merge(
                    q_blocks["lag"].drop(columns=["q"]),
                    on="date",
                    how="left"
                )
                .merge(
                    q_blocks["rolling"].drop(columns=["q"]),
                    on="date",
                    how="left"
                )
                .merge(
                    q_blocks["expanding"].drop(columns=["q"]),
                    on="date",
                    how="left"
                )
                .merge(
                    q_blocks["sri"].drop(columns=["q"]),
                    on="date",
                    how="left"
                )
            )

            dataset = (
                base.merge(pr_all, on="date", how="left")
                .merge(eto_all, on="date", how="left")
                .merge(q_all, on="date", how="left")
            )

            # qout / qbal / tws por bacia
            def _merge_local(
                var: str,
                df_local: Optional[pd.DataFrame],
            ) -> None:
                nonlocal dataset
                if df_local is None:
                    return
                bl = basin_features[var]
                all_ = (
                    df_local[["date", var]]
                    .merge(
                        bl["lag"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                    .merge(
                        bl["rolling"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                    .merge(
                        bl["expanding"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                )
                dataset = dataset.merge(all_, on="date", how="left")

            _merge_local("qout", df_qout)
            _merge_local("qbal", df_qbal)
            _merge_local("tws", df_tws)

            # globais roteadas
            def _merge_global(
                var: str,
                df_global: Optional[pd.DataFrame],
            ) -> None:
                nonlocal dataset
                if df_global is None:
                    return
                if var not in basin_features:
                    return
                bl = basin_features[var]
                all_ = (
                    df_global[["date", var]]
                    .merge(
                        bl["lag"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                    .merge(
                        bl["rolling"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                    .merge(
                        bl["expanding"].drop(columns=[var]),
                        on="date",
                        how="left"
                    )
                )
                dataset = dataset.merge(all_, on="date", how="left")

            _merge_global("qt7", df_qt7)
            _merge_global("qt6", df_qt6)
            _merge_global("qt5", df_qt5)
            _merge_global("qpbs_net", df_qpbs)
            _merge_global("qelev", df_qelev)

            dataset = dataset.sort_values("date")
            datasets_by_basin[basin] = dataset

        return features, datasets_by_basin
