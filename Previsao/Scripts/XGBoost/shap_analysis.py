from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .experiments import _infer_scale_columns, _scale_train_only
from .xgboost_model import SplitMasks, XGBoostModel


@dataclass
class FittedExperiment:
    """Container for a fitted single experiment and its supervised dataset."""

    model: Any
    xgb: XGBoostModel
    X_exp: pd.DataFrame
    y_sup: pd.Series
    split: SplitMasks
    pred_df: pd.DataFrame
    feature_cols: list[str]
    experiment: str
    basin: str
    horizon: int
    scaler: Any | None = None


def get_model_matrix(xgb: XGBoostModel, X: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the matrix used by the XGBoost model."""
    if hasattr(xgb, "_model_matrix"):
        return xgb._model_matrix(X)  # type: ignore[attr-defined]

    drop_cols = [xgb.date_col, "issue_date", "target_date"]
    raw_cols = [c for c in X.columns if c.endswith("_raw")]
    return X.drop(columns=drop_cols + raw_cols, errors="ignore")


def fit_single_experiment_for_shap(
    *,
    xgb: XGBoostModel,
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    basin: str,
    horizon: int,
    min_date: str,
    max_date: str,
    split_mode: str,
    train_end: str,
    val_end: str | None,
    experiment: str,
    feature_cols: list[str],
    core_required: list[str],
    target_lags: list[int] | None = None,
    split_on: str = "target_date",
    scale_mode: str | None = None,
    scale_raw_vars: list[str] | None = None,
    scale_include_lags: bool = True,
    early_stopping_rounds: int = 200,
) -> FittedExperiment:
    """
    Fit one experiment and return objects needed for SHAP.

    This mirrors ``run_experiments`` but keeps the fitted model and feature
    matrix. SHAP should be computed on the same scaled matrix
    used by the model.
    """
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

    X_exp = xgb.select_features(X_sup, feature_cols)

    # Preserve physical-scale references for persistence and plots.
    if "Vol" in X_exp.columns and "Vol_raw" not in X_exp.columns:
        X_exp["Vol_raw"] = X_exp["Vol"]

    for col in X_exp.columns:
        if col.startswith("Vol_Lag") and f"{col}_raw" not in X_exp.columns:
            X_exp[f"{col}_raw"] = X_exp[col]

    scaler = None
    if scale_mode is not None:
        raw_vars = scale_raw_vars if scale_raw_vars is not None else [
            "pr", "eto", "q", "Vol"
        ]
        scale_cols = _infer_scale_columns(
            feature_cols=feature_cols,
            raw_vars=raw_vars,
            include_lags=scale_include_lags,
        )
        X_exp, scaler = _scale_train_only(
            X=X_exp,
            split=split,
            scale_cols=scale_cols,
            mode=scale_mode,
        )

    model = xgb.fit(
        X_exp,
        y_sup,
        split,
        early_stopping_rounds=early_stopping_rounds,
    )

    pred_df = xgb.predict_frame(model, X_exp, y_sup, split, horizon)
    pred_df["experiment"] = experiment
    pred_df["basin"] = basin

    return FittedExperiment(
        model=model,
        xgb=xgb,
        X_exp=X_exp,
        y_sup=y_sup,
        split=split,
        pred_df=pred_df,
        feature_cols=feature_cols,
        experiment=experiment,
        basin=basin,
        horizon=horizon,
        scaler=scaler,
    )


def _split_mask(split: SplitMasks, split_name: str) -> np.ndarray:
    if split_name == "train":
        return split.train
    if split_name == "val":
        if split.val is None:
            raise ValueError("split.val is None; no validation split is available.")
        return split.val
    if split_name == "test":
        return split.test
    if split_name == "all":
        return np.ones_like(split.train, dtype=bool)
    raise ValueError("split_name must be one of: 'train', 'val', 'test', 'all'.")


def compute_tree_shap(
    fitted: FittedExperiment,
    *,
    split_name: str = "test",
    max_samples: int | None = None,
    random_state: int = 42,
):
    """
    Compute SHAP values for a fitted XGBoost experiment.

    For ``target_mode='delta'``, the SHAP values explain the model output in
    delta space (e.g. ΔVol_3), before reconstruction to volume level.
    """
    import shap

    X_model = get_model_matrix(fitted.xgb, fitted.X_exp)
    mask = _split_mask(fitted.split, split_name)
    X_eval = X_model.loc[mask].copy()

    if max_samples is not None and len(X_eval) > max_samples:
        X_eval = X_eval.sample(max_samples, random_state=random_state)

    explainer = shap.TreeExplainer(fitted.model)
    shap_values = explainer(X_eval)
    return shap_values, X_eval, explainer


def shap_importance_table(shap_values, X_eval: pd.DataFrame) -> pd.DataFrame:
    """Return feature-level mean absolute SHAP importance."""
    vals = np.asarray(shap_values.values)
    if vals.ndim != 2:
        raise ValueError("Expected 2D SHAP values for a single-output model.")

    out = pd.DataFrame(
        {
            "feature": X_eval.columns,
            "mean_abs_shap": np.abs(vals).mean(axis=0),
            "mean_shap": vals.mean(axis=0),
            "std_abs_shap": np.abs(vals).std(axis=0),
        }
    )
    total = out["mean_abs_shap"].sum()
    out["relative_importance"] = (
        out["mean_abs_shap"] / total if total > 0 else np.nan
    )
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def assign_feature_group(feature: str) -> str:
    """Map model feature names to hydrologically interpretable groups."""
    if feature in {"Vol_JAGJAC", "Vol_total"}:
        return "system_storage"
    if feature.startswith("Vol"):
        return "local_storage"
    if feature == "tws" or feature.startswith("tws_") or feature.startswith("tws"):
        return "terrestrial_water_storage"
    if feature.startswith(("pr", "SPI")):
        return "precipitation"
    if feature.startswith(("eto", "SPEI")):
        return "evapotranspiration"
    if feature.startswith(("SRI", "q_Rolling", "q_Expanding", "q_Lag")) or feature == "q":
        return "natural_flow"
    if feature.startswith(("qout", "qbal", "qt", "qpbs", "qelev")):
        return "operation"
    if feature.startswith("month") or feature == "is_dry":
        return "calendar"
    return "other"


def shap_group_importance_table(
    shap_values,
    X_eval: pd.DataFrame,
    *,
    group_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return mean absolute SHAP importance aggregated by feature group."""
    vals = np.asarray(shap_values.values)
    abs_df = pd.DataFrame(np.abs(vals), columns=X_eval.columns)

    rows = []
    for feature in abs_df.columns:
        rows.append(
            {
                "feature": feature,
                "group": assign_feature_group(feature),
                "mean_abs_shap": abs_df[feature].mean(),
            }
        )
    long = pd.DataFrame(rows)
    out = (
        long.groupby("group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)  # type: ignore
        .reset_index(drop=True)
    )
    total = out["mean_abs_shap"].sum()
    out["relative_importance"] = (
        out["mean_abs_shap"] / total if total > 0 else np.nan
    )

    if group_order is not None:
        order_map = {g: i for i, g in enumerate(group_order)}
        out["_order"] = out["group"].map(order_map).fillna(len(order_map))
        out = out.sort_values(["_order", "mean_abs_shap"], ascending=[True, False])
        out = out.drop(columns="_order").reset_index(drop=True)

    return out


def save_shap_beeswarm(
    shap_values,
    *,
    output_path: str | Path | None = None,
    max_display: int = 15,
    title: str | None = None,
    dpi: int = 500,
    figsize: tuple[float, float] = (6.4, 5.8),
):
    """Create and optionally save a SHAP beeswarm plot with
    paper-style defaults."""
    import matplotlib.pyplot as plt
    import shap

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": dpi,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Cria figura fresca explicitamente. Caso contrário, em chamadas
    # consecutivas dentro do mesmo cell, ``shap.plots.beeswarm`` pinta
    # sobre a figura ativa e cada chamada adiciona sua própria colorbar
    # — resultado: a última figura mostra os 3 beeswarms sobrepostos.
    fig = plt.figure(figsize=figsize)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)

    if title:
        plt.title(title, loc="left", fontweight="bold", fontsize=10)

    ax = plt.gca()
    ax.grid(True, axis="x", alpha=0.30, linestyle="--")
    ax.axvline(0.0, color="0.35", linewidth=1.0, alpha=0.85)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig
