from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor


@dataclass(frozen=True)
class SplitMasks:
    """Máscaras booleanas para separar treino/validação/teste."""
    train: np.ndarray
    val: np.ndarray | None
    test: np.ndarray


class XGBoostModel:
    """
    Modelo de regressão baseado em XGBoost para previsão de volumes mensais.

    Formulação supervisionada com horizonte:

        X(t)  ->  y(t+H)

    onde:
    - X(t) contém apenas informações disponíveis até o mês t (observadas),
      incluindo Vol(t) e, opcionalmente, lags de Vol.
    - y(t+H) é o volume observado H meses à frente.

    Observação
    ----------
    O modo padrão aqui é "one-shot": no teste, cada linha usa as features
    observadas do mês t (incluindo Vol(t)) para prever Vol(t+H).
    """

    def __init__(
        self,
        target_col: str = "Vol",
        date_col: str = "date",
        add_seasonality: bool = True,
        target_mode: str = "level",
        add_target_changes: bool = False,
        xgb_params: Dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        if target_mode not in ("level", "delta"):
            raise ValueError("target_mode deve ser 'level' ou 'delta'.")

        self.target_col = target_col
        self.date_col = date_col
        self.add_seasonality = add_seasonality
        self.target_mode = target_mode
        self.add_target_changes = add_target_changes
        self.random_state = random_state

        default_params: Dict[str, Any] = {
            "n_estimators": 5000,
            "learning_rate": 0.02,
            "max_depth": 3,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": random_state,
            "n_jobs": -1,
        }

        self.xgb_params = default_params if xgb_params is None else {
            **default_params, **xgb_params
        }

    # ------------------------------------------------------------------ #
    # Calendar features
    # ------------------------------------------------------------------ #
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona sazonalidade intra-anual com sin/cos do mês.

        Evita a descontinuidade artificial entre Dezembro (12) e Janeiro (1).
        """
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])
        month = out[self.date_col].dt.month

        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        # Dry season flag (Apr–Sep in SE Brazil; adjust if needed)
        out["is_dry"] = ((month >= 4) & (month <= 9)).astype(int)
        return out

    # ------------------------------------------------------------------ #
    # Supervised dataset
    # ------------------------------------------------------------------ #
    def build_supervised(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        basin: str,
        horizon: int,
        min_date: str | None = None,
        max_date: str | None = None,
        core_required: List[str] | None = None,
        target_lags: List[int] | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Faz merge X(t) com Vol(t), cria target y(t+H) e filtra linhas válidas.

        Parâmetros
        ----------
        X_df : pd.DataFrame
            Features (por mês), com coluna de datas.
        y_df : pd.DataFrame
            Volumes wide: colunas [date, CT_ATIB, CT_CACH, ...].
        basin : str
            Coluna de y_df do reservatório.
        horizon : int
            Horizonte H em meses.
        min_date, max_date : str | None
            Recorte temporal.
        core_required : list[str] | None
            Colunas que não podem ser NaN (sanity filter).
        target_lags : list[int] | None
            Lags de volume a criar (ex.: [1] cria Vol_Lag1).

        Retorna
        -------
        X_sup : pd.DataFrame
            Features + Vol(t) (+ Vol_LagK se solicitado).
        y_sup : pd.Series
            Target Vol(t+H).
        """
        if horizon < 1:
            raise ValueError("horizon deve ser >= 1")

        X = X_df.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])

        y = y_df[[self.date_col, basin]].copy()
        y[self.date_col] = pd.to_datetime(y[self.date_col])
        y = y.rename(
            columns={basin: self.target_col}).sort_values(self.date_col)

        df = X.merge(
            y, on=self.date_col, how="left").sort_values(self.date_col)

        # Lags do volume (ex.: Vol_Lag1) para enriquecer "memória" do estado
        if target_lags:
            for lag in target_lags:
                lag_col = f"{self.target_col}_Lag{lag}"
                df[lag_col] = df[self.target_col].shift(lag)

            if self.add_target_changes:
                for lag in target_lags:
                    lag_col = f"{self.target_col}_Lag{lag}"
                    change_col = f"{self.target_col}_Change{lag}"
                    df[change_col] = df[self.target_col] - df[lag_col]

        # Target futuro em nível: Vol(t+H)
        future_name = f"{self.target_col}_t_plus_{horizon}"
        df[future_name] = df[self.target_col].shift(-horizon)

        # Modo do alvo
        if self.target_mode == "level":
            target_name = future_name

        elif self.target_mode == "delta":
            target_name = f"{self.target_col}_delta_plus_{horizon}"
            df[target_name] = df[future_name] - df[self.target_col]

        else:
            raise ValueError("target_mode deve ser 'level' ou 'delta'.")

        if min_date is not None:
            df = df[df[self.date_col] >= pd.to_datetime(min_date)]
        if max_date is not None:
            df = df[df[self.date_col] <= pd.to_datetime(max_date)]

        if self.add_seasonality:
            df = self.add_calendar_features(df)

        y_sup = df[target_name]

        drop_cols = [target_name]
        if future_name != target_name:
            drop_cols.append(future_name)

        X_sup = df.drop(columns=drop_cols)

        mask = y_sup.notna()
        if core_required:
            for col in core_required:
                if col not in X_sup.columns:
                    raise KeyError(f"Coluna obrigatória ausente: {col}")
                mask &= X_sup[col].notna()

        X_sup = X_sup.loc[mask].reset_index(drop=True)
        y_sup = y_sup.loc[mask].reset_index(drop=True)
        return X_sup, y_sup

    # ------------------------------------------------------------------ #
    # Splits
    # ------------------------------------------------------------------ #
    def temporal_split_2way(
            self, df: pd.DataFrame, train_end: str) -> SplitMasks:
        d = df.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col])
        train = (d[self.date_col] <= pd.to_datetime(train_end)).to_numpy()
        test = ~train
        return SplitMasks(train=train, val=None, test=test)

    def temporal_split_3way(
        self, df: pd.DataFrame, train_end: str, val_end: str
    ) -> SplitMasks:
        d = df.copy()
        d[self.date_col] = pd.to_datetime(d[self.date_col])
        train = (d[self.date_col] <= pd.to_datetime(train_end)).to_numpy()
        val = (
            (d[self.date_col] > pd.to_datetime(train_end))
            & (d[self.date_col] <= pd.to_datetime(val_end))
        ).to_numpy()
        test = (d[self.date_col] > pd.to_datetime(val_end)).to_numpy()
        return SplitMasks(train=train, val=val, test=test)

    def _target_dates(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Calcula target_date = issue_date + horizon meses.

        A coluna self.date_col representa a data das features X(t).
        """
        issue_date = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        return issue_date + pd.DateOffset(months=horizon)

    def temporal_split_2way_target_date(
        self,
        df: pd.DataFrame,
        horizon: int,
        train_end: str,
    ) -> SplitMasks:
        """
        Split 2way usando target_date, não issue_date.
        """
        target_date = self._target_dates(df, horizon)

        train = (target_date <= pd.to_datetime(train_end)).to_numpy()
        test = ~train

        return SplitMasks(train=train, val=None, test=test)

    def temporal_split_3way_target_date(
        self,
        df: pd.DataFrame,
        horizon: int,
        train_end: str,
        val_end: str,
    ) -> SplitMasks:
        """
        Split 3way usando target_date, não issue_date.
        """
        target_date = self._target_dates(df, horizon)

        train = (target_date <= pd.to_datetime(train_end)).to_numpy()

        val = (
            (target_date > pd.to_datetime(train_end))
            & (target_date <= pd.to_datetime(val_end))
        ).to_numpy()

        test = (target_date > pd.to_datetime(val_end)).to_numpy()

        return SplitMasks(train=train, val=val, test=test)

    # ------------------------------------------------------------------ #
    # Public helpers for experiments
    # ------------------------------------------------------------------ #
    def prepare_dataset(
        self,
        datasets_by_basin: Dict[str, pd.DataFrame],
        volumes: pd.DataFrame,
        basin: str,
        horizon: int,
        min_date: str,
        max_date: str,
        split_mode: str,
        train_end: str,
        val_end: str | None,
        core_required: List[str],
        target_lags: List[int] | None = None,
        split_on: str = "issue_date",
    ) -> Tuple[pd.DataFrame, pd.Series, SplitMasks]:
        """
        Prepara (X_sup, y_sup, split) sem treinar modelo.

        Útil para baterias de testes: constrói dataset e split uma vez e
        depois testa múltiplos feature sets sem retrabalho.
        """
        X_raw = datasets_by_basin[basin].copy()

        X_sup, y_sup = self.build_supervised(
            X_df=X_raw,
            y_df=volumes,
            basin=basin,
            horizon=horizon,
            min_date=min_date,
            max_date=max_date,
            core_required=core_required,
            target_lags=target_lags,
        )

        if split_on not in ("issue_date", "target_date"):
            raise ValueError(
                "split_on deve ser 'issue_date' ou 'target_date'."
            )

        if split_mode == "2way":
            if split_on == "issue_date":
                split = self.temporal_split_2way(
                    X_sup,
                    train_end=train_end,
                )
            else:
                split = self.temporal_split_2way_target_date(
                    X_sup,
                    horizon=horizon,
                    train_end=train_end,
                )

        elif split_mode == "3way":
            if val_end is None:
                raise ValueError("val_end é obrigatório no split 3way.")

            if split_on == "issue_date":
                split = self.temporal_split_3way(
                    X_sup,
                    train_end=train_end,
                    val_end=val_end,
                )
            else:
                split = self.temporal_split_3way_target_date(
                    X_sup,
                    horizon=horizon,
                    train_end=train_end,
                    val_end=val_end,
                )

        else:
            raise ValueError("split_mode deve ser '2way' ou '3way'.")

        return X_sup, y_sup, split

    def select_features(
            self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Seleciona subconjunto de colunas preservando a coluna de data.

        Levanta KeyError se alguma coluna não existir.
        """
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Features ausentes: {missing}")

        cols = [self.date_col] + feature_cols
        return X[cols].copy()

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    def baseline_persistence(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_col not in X.columns:
            raise KeyError(f"Coluna {self.target_col} não está em X.")
        return X[self.target_col].to_numpy()

    # ------------------------------------------------------------------ #
    # Train / predict / eval
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split: SplitMasks,
        early_stopping_rounds: int = 200,
    ) -> XGBRegressor:
        X_model = self._model_matrix(X)

        X_train = X_model.loc[split.train]
        y_train = y.loc[split.train]

        if split.val is None:
            model = XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train, verbose=False)
            return model

        X_val = X_model.loc[split.val]
        y_val = y.loc[split.val]

        params_with_es = {
            **self.xgb_params,
            "early_stopping_rounds": early_stopping_rounds,
        }
        model = XGBRegressor(**params_with_es)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    def predict(self, model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
        X_model = self._model_matrix(X)
        return model.predict(X_model)

    def evaluate_predictions(
            self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"RMSE": float(rmse), "MAE": float(mae)}

    def predict_frame(
        self,
        model: XGBRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        split: SplitMasks,
        horizon: int,
    ) -> pd.DataFrame:
        """
        DataFrame com observado vs previsto.

        Convenção temporal
        ------------------
        - issue_date: data do vetor de features X(t)
        - target_date: data do alvo y(t+H) = issue_date + H meses

        Retorna colunas:
        - issue_date
        - target_date
        - y_true: Vol(t+H)
        - y_pred: previsão do modelo
        - y_base: baseline persistência = Vol(t) (preferindo Vol_raw)
        - split: train/val/test
        - horizon
        """
        issue_date = pd.to_datetime(X[self.date_col]).reset_index(drop=True)
        target_date = issue_date + pd.DateOffset(months=horizon)

        y_pred_model = self.predict(model, X)
        y_model = y.reset_index(drop=True)

        # Volume atual em escala física: Vol(t)
        if "Vol_raw" in X.columns:
            vol_now = X["Vol_raw"].to_numpy()
        else:
            vol_now = self.baseline_persistence(X)

        # Reconstrução conforme modo do alvo
        if self.target_mode == "delta":
            y_true = vol_now + y_model.to_numpy()
            y_pred = vol_now + y_pred_model

            extra_cols = {
                "y_delta_true": y_model.to_numpy(),
                "y_delta_pred": y_pred_model,
            }

        else:
            y_true = y_model.to_numpy()
            y_pred = y_pred_model
            extra_cols = {}

        out = pd.DataFrame(
            {
                "issue_date": issue_date,
                "target_date": target_date,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_base": vol_now,
                **extra_cols,
            }
        )

        out["split"] = "test"
        out.loc[split.train, "split"] = "train"
        if split.val is not None:
            out.loc[split.val, "split"] = "val"

        out["horizon"] = horizon
        return out

    def _model_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas temporais e auxiliares que não devem entrar no modelo.
        """
        drop_cols = [
            self.date_col,
            "issue_date",
            "target_date",
        ]

        raw_cols = [c for c in X.columns if c.endswith("_raw")]

        return X.drop(columns=drop_cols + raw_cols, errors="ignore")
