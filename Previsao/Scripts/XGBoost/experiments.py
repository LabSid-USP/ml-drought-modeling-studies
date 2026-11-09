from __future__ import annotations

from dataclasses import dataclass
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .xgboost_model import SplitMasks, XGBoostModel


@dataclass(frozen=True)
class ExperimentResult:
    """Resultado agregado de um experimento."""
    experiment: str
    basin: str
    horizon: int
    test_rmse: float
    test_mae: float
    test_corr: float
    val_rmse: float | None = None
    val_mae: float | None = None
    val_corr: float | None = None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.all(np.isclose(a, a[0])) or np.all(np.isclose(b, b[0])):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _infer_scale_columns(
    feature_cols: List[str],
    raw_vars: List[str],
    include_lags: bool = True,
) -> List[str]:
    """
    Decide quais colunas serão normalizadas.

    Política (default)
    ------------------
    - Normaliza apenas variáveis "brutas" listadas em raw_vars.
    - Se include_lags=True, normaliza também colunas *_LagK dessas variáveis.

    Não normaliza índices (SPI/SPEI/SRI) nem calendário (month_sin/cos).
    """
    scale_cols: List[str] = []
    for col in feature_cols:
        if col in raw_vars:
            scale_cols.append(col)
            continue

        if include_lags:
            for v in raw_vars:
                if col.startswith(f"{v}_Lag"):
                    scale_cols.append(col)
                    break
    return scale_cols


def _scale_train_only(
    X: pd.DataFrame,
    split: SplitMasks,
    scale_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, Any]:
    """
    Aplica normalização fitada somente no treino.

    Parâmetros
    ----------
    X : pd.DataFrame
        Features com coluna de data.
    split : SplitMasks
        Máscaras train/val/test.
    scale_cols : list[str]
        Colunas a normalizar.
    mode : str
        'minmax' ou 'zscore'.

    Retorna
    -------
    X_scaled : pd.DataFrame
        Cópia de X com colunas normalizadas.
    scaler : object
        Objeto scaler ajustado no treino.
    """
    if not scale_cols:
        return X.copy(), None

    X_scaled = X.copy()

    if mode == "minmax":
        scaler = MinMaxScaler()
    elif mode == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError("mode deve ser 'minmax' ou 'zscore'.")

    train_data = X_scaled.loc[split.train, scale_cols]
    scaler.fit(train_data)

    X_scaled[scale_cols] = scaler.transform(X_scaled[scale_cols])
    return X_scaled, scaler


def run_experiments(
    xgb: XGBoostModel,
    datasets_by_basin: Dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    basin: str,
    horizon: int,
    min_date: str,
    max_date: str,
    split_mode: str,
    train_end: str,
    val_end: str | None,
    feature_sets: Dict[str, List[str]],
    core_required: List[str],
    target_lags: List[int] | None = None,
    split_on: str = "issue_date",
    scale_mode: str | None = None,
    scale_raw_vars: List[str] | None = None,
    scale_include_lags: bool = True,
    early_stopping_rounds: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roda uma bateria de experimentos para (bacia, horizonte).

    Retorna dois DataFrames:
    - results_df: métricas agregadas por experimento
    - preds_df: previsões (y_true/y_pred/y_base) por data + experimento

    Normalização (opcional)
    -----------------------
    Se scale_mode for 'minmax' ou 'zscore', aplica scaling apenas às colunas
    brutas listadas em scale_raw_vars e (opcionalmente) seus lags.

    Parâmetros
    ----------
    xgb : XGBoostModel
        Objeto com lógica de dataset/split/fit/predict.
    datasets_by_basin : dict[str, pd.DataFrame]
        Tabelão de features por bacia.
    volumes : pd.DataFrame
        DataFrame wide com volumes por bacia.
    basin : str
        Ex.: 'CT_ATIB'.
    horizon : int
        Horizonte (meses).
    min_date, max_date : str
        Recorte temporal.
    split_mode : str
        '2way' ou '3way'.
    train_end : str
        Fim do treino (inclusive).
    val_end : str | None
        Fim da validação (inclusive) quando split_mode='3way'.
    feature_sets : dict[str, list[str]]
        Conjuntos de features para testar.
    core_required : list[str]
        Colunas mínimas (sem NaN) para filtrar linhas válidas.
    target_lags : list[int] | None, optional
        Lags do volume para criar (ex.: [1]).
    scale_mode : str | None, optional
        None, 'minmax' ou 'zscore'.
    scale_raw_vars : list[str] | None, optional
        Variáveis brutas a normalizar (default: ['pr','eto','q','Vol']).
    scale_include_lags : bool, optional
        Se True, normaliza também colunas *_LagK das variáveis brutas.
    early_stopping_rounds : int, optional
        Early stopping quando houver validação.

    Retorna
    -------
    results_df : pd.DataFrame
        Métricas por experimento.
    preds_df : pd.DataFrame
        Previsões por data com colunas:
        date, y_true, y_pred, y_base, split, horizon, experiment, basin
    """
    # 1) Prepara dataset uma única vez
    X_sup, y_sup, split = xgb.prepare_dataset(
        datasets_by_basin=datasets_by_basin,
        volumes=volumes,
        basin=basin,
        horizon=horizon,
        min_date=min_date,
        max_date=max_date,
        split_mode=split_mode,
        train_end=train_end,
        val_end=val_end,
        core_required=core_required,
        target_lags=target_lags,
        split_on=split_on,
    )

    results: List[ExperimentResult] = []
    preds_all: List[pd.DataFrame] = []

    raw_vars = scale_raw_vars if scale_raw_vars is not None else [
        "pr", "eto", "q", "Vol"
    ]

    for exp_name, feature_cols in feature_sets.items():
        # 2) Seleciona features (mantém date)
        X_exp = xgb.select_features(X_sup, feature_cols)

        # Preserva colunas de referência em escala física
        # para baseline/diagnóstico
        if "Vol" in X_exp.columns and "Vol_raw" not in X_exp.columns:
            X_exp["Vol_raw"] = X_exp["Vol"]

        # Se você houver lags de volume como features, preserva também
        for col in X_exp.columns:
            if col.startswith("Vol_Lag") and f"{col}_raw" not in X_exp.columns:
                X_exp[f"{col}_raw"] = X_exp[col]

        # 3) Normaliza (opcional) - fit no treino, aplica em todos
        if scale_mode is not None:
            scale_cols = _infer_scale_columns(
                feature_cols=feature_cols,
                raw_vars=raw_vars,
                include_lags=scale_include_lags,
            )
            X_exp, _ = _scale_train_only(
                X=X_exp,
                split=split,
                scale_cols=scale_cols,
                mode=scale_mode,
            )

        # 4) Treina
        model = xgb.fit(
            X_exp,
            y_sup,
            split,
            early_stopping_rounds=early_stopping_rounds,
        )

        # 5) Predições por data
        pred_df = xgb.predict_frame(model, X_exp, y_sup, split, horizon)
        pred_df["experiment"] = exp_name
        pred_df["basin"] = basin
        preds_all.append(pred_df)

        # 6) Métricas (teste; e validação se existir)
        test = pred_df[pred_df["split"] == "test"]
        test_rmse = root_mean_squared_error(test["y_true"], test["y_pred"])
        test_mae = mean_absolute_error(test["y_true"], test["y_pred"])
        test_corr = _corr(test["y_true"].to_numpy(), test["y_pred"].to_numpy())

        val_rmse = None
        val_mae = None
        val_corr = None
        if split.val is not None:
            val = pred_df[pred_df["split"] == "val"]
            val_rmse = root_mean_squared_error(val["y_true"], val["y_pred"])
            val_mae = mean_absolute_error(val["y_true"], val["y_pred"])
            val_corr = _corr(
                val["y_true"].to_numpy(), val["y_pred"].to_numpy()
            )

        results.append(
            ExperimentResult(
                experiment=exp_name,
                basin=basin,
                horizon=horizon,
                test_rmse=float(test_rmse),
                test_mae=float(test_mae),
                test_corr=float(test_corr),
                val_rmse=None if val_rmse is None else float(val_rmse),
                val_mae=None if val_mae is None else float(val_mae),
                val_corr=None if val_corr is None else float(val_corr),
            )
        )

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df = results_df.sort_values(["test_rmse", "experiment"])

    preds_df = pd.concat(preds_all, ignore_index=True)
    preds_df["target_date"] = pd.to_datetime(preds_df["target_date"])
    preds_df = preds_df.sort_values(["basin", "experiment", "target_date"])

    return results_df, preds_df


def _month_add(d: pd.Timestamp, months: int) -> pd.Timestamp:
    return pd.Timestamp(d) + pd.DateOffset(months=months)


def _make_wf_splits(
    target_dates: pd.Series,
    train_end: pd.Timestamp,
    test_window_months: int,
    val_window_months: int | None,
) -> SplitMasks:
    """
    Cria máscaras train/val/test para uma rodada walk-forward.

    IMPORTANTE: os masks são definidos no tempo do alvo (target_date).

    - train: target_date <= train_end
    - test:  (train_end, train_end + test_window_months]
    - val:   últimos val_window_months do treino (opcional)

    Se val_window_months é fornecido, removemos esse bloco do treino efetivo,
    para evitar leakage no early stopping.
    """
    d = pd.to_datetime(target_dates)

    test_end = train_end + pd.DateOffset(months=test_window_months)
    train_mask = (d <= train_end).to_numpy()
    test_mask = ((d > train_end) & (d <= test_end)).to_numpy()

    if val_window_months is None:
        return SplitMasks(train=train_mask, val=None, test=test_mask)

    val_start = train_end + pd.DateOffset(months=-val_window_months + 1)
    val_mask = ((d >= val_start) & (d <= train_end)).to_numpy()

    # treino efetivo: antes do bloco de validação
    train_mask = (d < val_start).to_numpy()
    return SplitMasks(train=train_mask, val=val_mask, test=test_mask)


def run_walk_forward_expanding(
    xgb: XGBoostModel,
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    basin: str,
    horizon: int,
    min_date: str,
    max_date: str,
    initial_train_end: str,
    feature_cols: list[str],
    core_required: list[str],
    target_lags: list[int] | None = None,
    scale_mode: str | None = None,
    scale_raw_vars: list[str] | None = None,
    scale_include_lags: bool = True,
    val_window_months: int | None = 36,
    early_stopping_rounds: int = 200,
    step_months: int = 1,
    test_window_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward com janela expansiva (análise de sensibilidade).

    Não faz parte do benchmark canônico do trabalho (split 3way fixo). É
    fornecida como utilitário de robustez para inspecionar a estabilidade
    das métricas em horizontes deslizantes.

    Parameters
    ----------
    horizon : int
        Horizonte de previsão (define o alvo y(t+H)).
    test_window_months : int
        Tamanho do bloco de avaliação por iteração (default: 1 mês).

    Returns
    -------
    metrics_df : pd.DataFrame
        Métricas por rodada (indexadas por train_end).
    preds_df : pd.DataFrame
        Previsões concatenadas dos testes (um mês por iteração).
    """
    raw_vars = scale_raw_vars if scale_raw_vars is not None else [
        "pr", "eto", "q", "Vol"
    ]

    # Prepara dataset completo uma vez (split dummy)
    X_sup, y_sup, _ = xgb.prepare_dataset(
        datasets_by_basin=datasets_by_basin,
        volumes=volumes,
        basin=basin,
        horizon=horizon,
        min_date=min_date,
        max_date=max_date,
        split_mode="2way",
        train_end=max_date,
        val_end=None,
        core_required=core_required,
        target_lags=target_lags,
    )

    X_all = xgb.select_features(X_sup, feature_cols)

    # Preserva raws para diagnóstico/plots (antes de qualquer scaling)
    for col in X_all.columns:
        if col in ("date", "issue_date", "target_date"):
            continue
        # só para variáveis contínuas (heurística simples)
        if col in (scale_raw_vars or []):  # se você passou scale_raw_vars
            raw_name = f"{col}_raw"
            if raw_name not in X_all.columns:
                X_all[raw_name] = X_all[col]

    # Preserva Vol_raw SEMPRE para baseline/plots
    if "Vol" in X_all.columns and "Vol_raw" not in X_all.columns:
        X_all["Vol_raw"] = X_all["Vol"]

    # Define target_date para split (tempo do alvo)
    issue_date = pd.to_datetime(X_all[xgb.date_col]).reset_index(drop=True)
    target_date = issue_date + pd.DateOffset(months=horizon)

    t0 = pd.to_datetime(initial_train_end)
    tmax = pd.to_datetime(max_date)

    metrics_rows = []
    preds_all = []

    train_end = t0
    while True:
        test_end = train_end + pd.DateOffset(months=test_window_months)
        if test_end > tmax:
            break

        split = _make_wf_splits(
            target_dates=target_date,
            train_end=train_end,
            test_window_months=test_window_months,
            val_window_months=val_window_months,
        )

        # Copia para permitir scaling fitado no treino da rodada
        X_round = X_all.copy()

        # Scaling opcional (fit no treino efetivo, aplica em todos)
        if scale_mode is not None:
            scale_cols = _infer_scale_columns(
                feature_cols=feature_cols,
                raw_vars=raw_vars,
                include_lags=scale_include_lags,
            )
            X_round, _ = _scale_train_only(
                X=X_round,
                split=split,
                scale_cols=scale_cols,
                mode=scale_mode,
            )

        model = xgb.fit(
            X_round,
            y_sup,
            split,
            early_stopping_rounds=early_stopping_rounds,
        )

        pred_df = xgb.predict_frame(model, X_round, y_sup, split, horizon)

        # Mantém apenas o teste (1 mês por iteração)
        pred_test = pred_df[pred_df["split"] == "test"].copy()
        pred_test["basin"] = basin
        pred_test["train_end"] = train_end
        preds_all.append(pred_test)

        # Como o teste tem 1 mês, RMSE/MAE por rodada = erro desse ponto
        rmse = root_mean_squared_error(
            pred_test["y_true"], pred_test["y_pred"])
        mae = mean_absolute_error(pred_test["y_true"], pred_test["y_pred"])
        # corr = float("nan")  # 1 ponto não permite correlação

        metrics_rows.append(
            {
                "basin": basin,
                "horizon": horizon,
                "train_end": train_end,
                "test_start": train_end + pd.DateOffset(months=1),
                "test_end": test_end,
                "test_rmse": float(rmse),
                "test_mae": float(mae),
                "n_test": int(len(pred_test)),
            }
        )

        train_end = train_end + pd.DateOffset(months=step_months)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = (
        pd.concat(
            preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    )

    if not preds_df.empty:
        preds_df["target_date"] = pd.to_datetime(preds_df["target_date"])
        preds_df = preds_df.sort_values(["basin", "target_date"])

    return metrics_df, preds_df


# --------------------------------------------------------------------------- #
# Baselines não-paramétricos para diagnóstico
# --------------------------------------------------------------------------- #
def seasonal_climatology_baseline(
    preds_df: pd.DataFrame,
    *,
    target_date_col: str = "target_date",
    truth_col: str = "y_true",
    split_col: str = "split",
    train_split_name: str = "train",
    out_col: str = "y_clim",
) -> pd.DataFrame:
    """Adiciona coluna com previsão por climatologia sazonal.

    A "climatologia sazonal" prevê ``y(t+H)`` como a média histórica de
    ``y_true`` para o mês-alvo ``(t+H).month``, calculada **apenas no
    conjunto de treino**. Não há vazamento de informação: o conjunto de
    validação/teste recebe a média estimada exclusivamente sobre as
    observações do treino.

    Esse baseline é mais severo que a persistência (``y_base = Vol(t)``)
    em sistemas com forte ciclo sazonal — por exemplo, na previsão de
    ``ΔVol_3`` em CT_TOTAL. Se o XGBoost não supera a climatologia,
    é sinal de que a alta importância SHAP de ``month_cos`` reflete
    redundância (o modelo apenas replica calendário) e não skill
    hidroclimático.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame de previsões no formato emitido por
        :func:`run_experiments` (colunas ``target_date``, ``y_true``,
        ``split``; opcionalmente ``basin``, ``experiment``, ``target_mode``).
    target_date_col, truth_col, split_col : str
        Nomes das colunas relevantes em ``preds_df``.
    train_split_name : str
        Valor que identifica o split de treino em ``split_col``.
    out_col : str
        Nome da coluna de saída com a previsão climatológica.

    Returns
    -------
    pd.DataFrame
        Cópia de ``preds_df`` com a coluna ``out_col`` adicionada.
        A coluna ``_target_month`` é descartada antes do retorno.
    """
    df = preds_df.copy()
    df[target_date_col] = pd.to_datetime(df[target_date_col])
    df["_target_month"] = df[target_date_col].dt.month

    group_cols = [
        c for c in ("basin", "experiment", "target_mode") if c in df.columns
    ]
    clim = (
        df[df[split_col] == train_split_name]
        .groupby(group_cols + ["_target_month"], dropna=False)[truth_col]
        .mean()
        .rename(out_col)
        .reset_index()
    )
    df = df.merge(clim, on=group_cols + ["_target_month"], how="left")
    return df.drop(columns="_target_month")
