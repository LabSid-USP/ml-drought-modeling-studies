"""Engenharia de features tabulares: lags, rolling, expanding e índices
padronizados.

NaNs introduzidos por shifts/janelas são esperados nas bordas iniciais das
séries e devem ser filtrados a jusante (em ``XGBoostModel.build_supervised``,
via ``core_required``), e não dentro destas funções.
"""
import numpy as np
import pandas as pd
from scipy import stats


class FeatureEngineering:
    # Lag time e window features
    def lag_features(
            self, df: pd.DataFrame, target_col: str, lags: list[int]
    ) -> pd.DataFrame:
        """
        Cria features de lag para a coluna alvo.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        target_col : str
            Nome da coluna alvo.
        lags : list[int]
            Lista de inteiros representando os períodos de lag.

        Retorna
        -------
        pd.DataFrame
            DataFrame com as features de lag adicionadas.
        """
        for lag in lags:
            df[f'{target_col}_Lag{lag}'] = df[target_col].shift(lag)
        return df

    def rolling_features(
            self, df: pd.DataFrame, target_col: str,
            operation: str, windows: list[int]
    ) -> pd.DataFrame:
        """
        Cria features móveis para a coluna alvo.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        target_col : str
            Nome da coluna alvo.
        operation : str
            Operação a ser aplicada (ex: 'Mean', 'Sum', 'Std').
        windows : list[int]
            Lista de inteiros representando os tamanhos das janelas.

        Retorna
        -------
        pd.DataFrame
            DataFrame com as features móveis adicionadas.
        """
        for window in windows:
            col_name = f'{target_col}_Rolling{operation}{window}'
            if operation == 'Mean':
                df[col_name] = df[target_col].rolling(window=window).mean()
            elif operation == 'Sum':
                df[col_name] = df[target_col].rolling(window=window).sum()
            elif operation == 'Std':
                df[col_name] = df[target_col].rolling(window=window).std()
        return df

    def expanding_features(
            self, df: pd.DataFrame, target_col: str, operation: str
    ) -> pd.DataFrame:
        """
        Cria uma feature de estatística acumulada para a coluna alvo.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        target_col : str
            Nome da coluna alvo.
        operation : str
            Operação a ser aplicada (ex: 'Mean', 'Sum', 'Std').

        Retorna
        -------
        pd.DataFrame
            DataFrame com a feature de estatística acumulada adicionada.
        """
        if operation == 'Mean':
            col_name = f'{target_col}_ExpandingMean'
            df[col_name] = df[target_col].expanding().mean()
        elif operation == 'Sum':
            col_name = f'{target_col}_ExpandingSum'
            df[col_name] = df[target_col].expanding().sum()
        elif operation == 'Std':
            col_name = f'{target_col}_ExpandingStd'
            df[col_name] = df[target_col].expanding().std()
        return df

    def normalization(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Normaliza a coluna alvo usando Min-Max Scaling.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        target_col : str
            Nome da coluna alvo.

        Retorna
        -------
        pd.DataFrame
            DataFrame com a coluna alvo normalizada.
        """
        min_val = df[target_col].min()
        max_val = df[target_col].max()
        if max_val == min_val:
            # Série constante: evita divisão por zero atribuindo 0.0
            df[f'{target_col}_Normalized'] = 0.0
        else:
            df[f'{target_col}_Normalized'] = (
                df[target_col] - min_val) / (max_val - min_val)
        return df

    # Índices padronizados
    def _get_scipy_dist(self, dist: str):
        """
        Retorna o objeto de distribuição do SciPy a partir de um nome.

        Parâmetros
        ----------
        dist : str
            Nome da distribuição. Opções:
            - 'gamma': típica para SPI/SRI (com variáveis positivas).
            - 'pearson3': alternativa comum em Hidrologia.
            - 'fisk': log-logística (comum para SPEI; aceita negativos).

        Retorna
        -------
        scipy.stats.rv_continuous
            Objeto de distribuição do SciPy.
        """
        if dist == "gamma":
            return stats.gamma
        if dist == "pearson3":
            return stats.pearson3
        if dist == "fisk":
            return stats.fisk
        raise ValueError(f"Distribuição não suportada: {dist}")

    def standardized_index(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        scale: int,
        dist: str,
        out_col: str,
        freq: str = "MS",
        handle_zeros: bool = True,
        min_samples_per_month: int = 20,
        clip_eps: float = 1e-10
    ) -> pd.DataFrame:
        """
        Calcula um índice padronizado do tipo SPI/SPEI/SRI.

        O procedimento envolve:
        (1) Agregar a variável na escala `scale` (soma móvel)
        (2) Ajustar uma distribuição por mês do calendário (sazonalidade)
        (3) Obter probabilidade via CDF e transformar para Normal padrão
            usando `norm.ppf`, gerando uma série ~N(0,1)

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo colunas de data e valor.
        date_col : str
            Nome da coluna de datas.
        value_col : str
            Nome da coluna com a variável (pr, q, ou d=pr-eto).
        scale : int
            Escala de acumulação em meses (1, 3, 6, 12, ...).
        dist : str
            Distribuição a ser ajustada ('gamma', 'pearson3', 'fisk').
        out_col : str
            Nome da coluna de saída para o índice calculado.
        freq : str, optional
            Frequência mensal usada para alinhar a série ('MS' padrão).
        handle_zeros : bool, optional
            Se True, trata zeros com mistura (massa em zero) quando `dist`
            é 'gamma'. Útil para precipitação/vazão com meses zerados.
        min_samples_per_month : int, optional
            Mínimo de amostras por mês para ajustar a distribuição.
        clip_eps : float, optional
            Evita 0 e 1 na CDF, prevenindo +/-inf no `norm.ppf`.

        Retorna
        -------
        pd.DataFrame
            DataFrame original com a coluna `out_col` adicionada.
        """
        if scale < 1:
            raise ValueError("scale deve ser >= 1")

        x = df[[date_col, value_col]].copy()
        x[date_col] = pd.to_datetime(x[date_col])
        x = x.sort_values(date_col).set_index(date_col)

        # Alinha para início do mês ("MS"). Isso evita desalinhamentos
        s = x[value_col].asfreq(freq)

        # Soma móvel: para scale=12, os 11 primeiros meses ficam NaN
        acc = s.rolling(window=scale, min_periods=scale).sum()

        dist_obj = self._get_scipy_dist(dist)
        idx = pd.Series(index=acc.index, dtype="float64")

        for month in range(1, 13):
            # Trabalha por mês do calendário para remover sazonalidade
            month_mask = acc.index.month == month  # type: ignore
            month_data = acc.loc[month_mask].dropna()

            if len(month_data) < min_samples_per_month:
                idx.loc[month_mask] = np.nan
                continue

            # Se houver zeros (e for gamma), usa CDF mista:
            # F(x) = q0 + (1-q0)*G(x) para x>0; F(0)=q0
            if handle_zeros and dist == "gamma":
                q0 = (month_data == 0).mean()
                pos = month_data[month_data > 0]
                if len(pos) < min_samples_per_month:
                    idx.loc[month_mask] = np.nan
                    continue

                # Fit da Gamma com loc=0 é prática comum para SPI/SRI
                params = dist_obj.fit(pos.values, floc=0.0)

                vals = acc.loc[month_mask].values.astype(float)
                F = np.full_like(vals, np.nan, dtype=float)

                valid = ~np.isnan(vals)
                vv = vals[valid]

                Fv = np.empty_like(vv, dtype=float)
                Fv[vv <= 0] = q0
                if np.any(vv > 0):
                    G = dist_obj.cdf(vv[vv > 0], *params)
                    Fv[vv > 0] = q0 + (1.0 - q0) * G

                F[valid] = Fv
            else:
                # Para SPEI (fisk) e Pearson3, não há "massa em zero"
                params = dist_obj.fit(month_data.values)

                vals = acc.loc[month_mask].values.astype(float)
                F = np.full_like(vals, np.nan, dtype=float)

                valid = ~np.isnan(vals)
                vv = vals[valid]
                F[valid] = dist_obj.cdf(vv, *params)

            # Evita valores exatamente 0 ou 1 na CDF (ppf -> inf)
            F = np.clip(F, clip_eps, 1.0 - clip_eps)

            # Converte probabilidade para escore Normal padrão (Z-score)
            idx.loc[month_mask] = stats.norm.ppf(F)

        # Reconecta ao DataFrame original preservando alinhamento por data
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col)

        idx_df = idx.rename(out_col).reset_index().rename(
            columns={"index": date_col}
        )
        out = out.merge(idx_df, on=date_col, how="left")
        return out

    def spi(
        self,
        df_pr: pd.DataFrame,
        date_col: str = "date",
        pr_col: str = "pr",
        scale: int = 12,
        out_col: str | None = None
    ) -> pd.DataFrame:
        """
        Calcula SPI a partir de precipitação mensal.

        Parâmetros
        ----------
        df_pr : pd.DataFrame
            DataFrame com colunas de data e precipitação.
        date_col : str, optional
            Nome da coluna de datas.
        pr_col : str, optional
            Nome da coluna de precipitação.
        scale : int, optional
            Escala (meses): 1, 3, 6, 12, ...
        out_col : str | None, optional
            Nome da coluna de saída. Se None, usa 'SPI{scale}'.

        Retorna
        -------
        pd.DataFrame
            DataFrame com a coluna SPI adicionada.
        """
        if out_col is None:
            out_col = f"SPI{scale}"
        return self.standardized_index(
            df=df_pr,
            date_col=date_col,
            value_col=pr_col,
            scale=scale,
            dist="gamma",
            out_col=out_col,
            handle_zeros=True
        )

    def spei(
        self,
        df_pr: pd.DataFrame,
        df_eto: pd.DataFrame,
        date_col: str = "date",
        pr_col: str = "pr",
        eto_col: str = "eto",
        scale: int = 12,
        out_col: str | None = None
    ) -> pd.DataFrame:
        """
        Calcula SPEI a partir de balanço climático D = pr - eto.

        Parâmetros
        ----------
        df_pr : pd.DataFrame
            DataFrame com colunas de data e precipitação.
        df_eto : pd.DataFrame
            DataFrame com colunas de data e evapotranspiração (ET0).
        date_col : str, optional
            Nome da coluna de datas (comum aos dois DataFrames).
        pr_col : str, optional
            Nome da coluna de precipitação.
        eto_col : str, optional
            Nome da coluna de evapotranspiração (ET0).
        scale : int, optional
            Escala (meses): 1, 3, 6, 12, ...
        out_col : str | None, optional
            Nome da coluna de saída. Se None, usa 'SPEI{scale}'.

        Retorna
        -------
        pd.DataFrame
            DataFrame com a coluna SPEI adicionada (baseado em D).
        """
        if out_col is None:
            out_col = f"SPEI{scale}"

        pr = df_pr[[date_col, pr_col]].copy()
        eto = df_eto[[date_col, eto_col]].copy()

        pr[date_col] = pd.to_datetime(pr[date_col])
        eto[date_col] = pd.to_datetime(eto[date_col])

        pr = pr.sort_values(date_col).set_index(date_col)
        eto = eto.sort_values(date_col).set_index(date_col)

        # D_t = pr_t - eto_t alinhado mensalmente
        d = (pr[pr_col].asfreq("MS") - eto[eto_col].asfreq("MS"))
        d_df = d.rename("d").reset_index()

        return self.standardized_index(
            df=d_df,
            date_col=date_col,
            value_col="d",
            scale=scale,
            dist="fisk",
            out_col=out_col,
            handle_zeros=False
        )

    def sri(
        self,
        df_q: pd.DataFrame,
        date_col: str = "date",
        q_col: str = "q",
        scale: int = 12,
        out_col: str | None = None
    ) -> pd.DataFrame:
        """
        Calcula SRI a partir de vazão mensal.

        Parâmetros
        ----------
        df_q : pd.DataFrame
            DataFrame com colunas de data e vazão.
        date_col : str, optional
            Nome da coluna de datas.
        q_col : str, optional
            Nome da coluna de vazão.
        scale : int, optional
            Escala (meses): 1, 3, 6, 12, ...
        out_col : str | None, optional
            Nome da coluna de saída. Se None, usa 'SRI{scale}'.

        Retorna
        -------
        pd.DataFrame
            DataFrame com a coluna SRI adicionada.
        """
        if out_col is None:
            out_col = f"SRI{scale}"
        return self.standardized_index(
            df=df_q,
            date_col=date_col,
            value_col=q_col,
            scale=scale,
            dist="gamma",
            out_col=out_col,
            handle_zeros=True
        )
