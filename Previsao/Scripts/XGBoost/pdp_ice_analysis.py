"""Análise de dependência parcial (PDP) e ICE para modelos XGBoost ajustados.

Este módulo opera sobre o mesmo container :class:`FittedExperiment` usado pela
análise SHAP em :mod:`Previsao.Scripts.shap_analysis`, permitindo:

* ``compute_pdp(...)``: calcula curvas PDP para uma lista de features.
* ``compute_ice(...)``: calcula curvas ICE individuais + média.
* ``plot_pdp_static(...)`` e ``plot_ice_static(...)``: figuras matplotlib
  (estilo paper) prontas para exportação.

As curvas são computadas no **conjunto de teste** por padrão (evitando ilusão
de overfit do treino). Internamente reaproveitam a matriz exata usada pelo
XGBoost (via :func:`get_model_matrix`).

Referência: https://scikit-learn.org/stable/modules/partial_dependence.html
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .shap_analysis import FittedExperiment, _split_mask, get_model_matrix


# --------------------------------------------------------------------------- #
# Núcleo: PDP e ICE
# --------------------------------------------------------------------------- #
def _prepare_eval_matrix(
    fitted: FittedExperiment,
    *,
    split_name: str,
    max_samples: int | None,
    random_state: int,
) -> pd.DataFrame:
    X_model = get_model_matrix(fitted.xgb, fitted.X_exp)
    mask = _split_mask(fitted.split, split_name)
    X_eval = X_model.loc[mask].copy()
    if max_samples is not None and len(X_eval) > max_samples:
        X_eval = X_eval.sample(max_samples, random_state=random_state)
    return X_eval


def _grid_for_feature(
    series: pd.Series,
    *,
    grid_resolution: int,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
) -> np.ndarray:
    """Constrói uma grade entre os percentis 5% e 95% da feature.

    Limitar aos percentis evita pontos extremos com pouco suporte (cauda)
    que poderiam dominar visualmente a curva PDP/ICE.
    """
    s = series.dropna().astype(float)
    if len(s) == 0:
        return np.array([])
    lo = np.quantile(s, percentile_low)
    hi = np.quantile(s, percentile_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # Fallback: usar valores únicos da série
        uniq = np.unique(s.values)  # type: ignore
        return uniq[: grid_resolution]
    return np.linspace(lo, hi, num=grid_resolution)


def compute_pdp(
    fitted: FittedExperiment,
    features: Iterable[str],
    *,
    split_name: str = "test",
    grid_resolution: int = 50,
    max_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computa Partial Dependence (PDP) para uma lista de features.

    Implementação manual (sem ``sklearn.inspection.partial_dependence``) para
    operar diretamente sobre o ``xgb.Booster`` ajustado e a matriz já
    pré-processada. O resultado é equivalente à definição clássica:

    .. math:: \\hat{f}_S(x_S) = E_{X_C}[ \\hat{f}(x_S, X_C) ]

    Parameters
    ----------
    fitted : FittedExperiment
        Container retornado por ``fit_single_experiment_for_shap``.
    features : iterable of str
        Nomes das features a serem avaliadas. Devem existir como colunas
        na matriz de modelo.
    split_name : str
        "train" / "val" / "test" / "all". Default: "test".
    grid_resolution : int
        Número de pontos na grade da feature (entre p5% e p95%).
    max_samples : int | None
        Limite opcional do número de amostras usadas como background.
    random_state : int
        Semente para amostragem reprodutível.

    Returns
    -------
    pd.DataFrame
        Colunas: ``feature``, ``feature_value``, ``pdp``.
    """
    X_eval = _prepare_eval_matrix(
        fitted,
        split_name=split_name,
        max_samples=max_samples,
        random_state=random_state,
    )
    rows: list[dict] = []
    for feat in features:
        if feat not in X_eval.columns:
            raise KeyError(
                f"Feature '{feat}' não presente na matriz de modelo. "
                f"Colunas disponíveis: {list(X_eval.columns)[:10]}...")
        grid = _grid_for_feature(
            X_eval[feat], grid_resolution=grid_resolution)
        for g in grid:
            X_tmp = X_eval.copy()
            X_tmp[feat] = g
            y_hat = fitted.model.predict(X_tmp)
            rows.append({
                "feature": feat,
                "feature_value": float(g),
                "pdp": float(np.mean(y_hat)),
            })
    return pd.DataFrame(rows)


def compute_ice(
    fitted: FittedExperiment,
    features: Iterable[str],
    *,
    split_name: str = "test",
    grid_resolution: int = 50,
    n_samples: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computa curvas ICE para uma lista de features.

    Cada amostra do conjunto de avaliação gera uma curva — para visualizar a
    *heterogeneidade* do efeito daquela feature em torno do efeito médio
    (PDP). Útil para identificar interações.

    Returns
    -------
    pd.DataFrame
        Colunas: ``feature``, ``sample_id``, ``feature_value``, ``ice``.
    """
    X_eval = _prepare_eval_matrix(
        fitted,
        split_name=split_name,
        max_samples=None,
        random_state=random_state,
    )
    if n_samples is not None and len(X_eval) > n_samples:
        X_eval = X_eval.sample(n_samples, random_state=random_state)

    rows: list[dict] = []
    for feat in features:
        if feat not in X_eval.columns:
            raise KeyError(
                f"Feature '{feat}' não presente na matriz de modelo.")
        grid = _grid_for_feature(
            X_eval[feat], grid_resolution=grid_resolution)
        for g in grid:
            X_tmp = X_eval.copy()
            X_tmp[feat] = g
            y_hat = fitted.model.predict(X_tmp)
            for sid, yh in zip(X_eval.index, y_hat):
                rows.append({
                    "feature": feat,
                    "sample_id": int(sid),
                    "feature_value": float(g),
                    "ice": float(yh),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots estáticos (matplotlib, estilo paper)
# --------------------------------------------------------------------------- #
def _apply_paper_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 500,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_pdp_static(
    pdp_df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    ncols: int = 3,
    output_path: str | Path | None = None,
    title: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 2.4),
):
    """Plota PDP em um grid de subplots (uma feature por painel)."""
    import matplotlib.pyplot as plt

    _apply_paper_style()
    feats = features if features is not None else sorted(
        pdp_df["feature"].unique())
    n = len(feats)
    if n == 0:
        raise ValueError("Nenhuma feature para plotar.")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        squeeze=False,
    )
    for i, feat in enumerate(feats):
        ax = axes[i // ncols][i % ncols]
        sub = pdp_df[pdp_df["feature"] == feat].sort_values("feature_value")
        ax.plot(sub["feature_value"], sub["pdp"], color="#1f3a5f", lw=1.8)
        ax.set_title(feat, loc="left", fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("PDP")
        ax.grid(True, alpha=0.25, linestyle="--")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontweight="bold", x=0.02, y=1.02, ha="left")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
    return fig


def plot_ice_static(
    ice_df: pd.DataFrame,
    pdp_df: pd.DataFrame | None = None,
    *,
    features: list[str] | None = None,
    ncols: int = 3,
    output_path: str | Path | None = None,
    title: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 2.4),
    ice_alpha: float = 0.12,
    center: bool = False,
):
    """Plota ICE (linhas finas) + PDP (linha grossa sobreposta) por feature.

    Parameters
    ----------
    ice_df : pd.DataFrame
        Output de :func:`compute_ice`.
    pdp_df : pd.DataFrame | None
        Opcional. Se fornecido, sobrepõe a curva PDP em destaque.
    center : bool
        Se True, centraliza cada curva ICE no valor inicial (c-ICE),
        evidenciando o **efeito** da feature em vez do nível absoluto.
    """
    import matplotlib.pyplot as plt

    _apply_paper_style()
    feats = features if features is not None else sorted(
        ice_df["feature"].unique())
    n = len(feats)
    if n == 0:
        raise ValueError("Nenhuma feature para plotar.")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        squeeze=False,
    )

    for i, feat in enumerate(feats):
        ax = axes[i // ncols][i % ncols]
        sub = ice_df[ice_df["feature"] == feat]
        if center:
            # c-ICE: subtrai o ICE do primeiro valor da grade por amostra
            ref = (
                sub.sort_values(["sample_id", "feature_value"])
                .groupby("sample_id", as_index=False)
                .first()[["sample_id", "ice"]]
                .rename(columns={"ice": "ice_ref"})
            )
            sub = sub.merge(ref, on="sample_id", how="left")
            sub["ice"] = sub["ice"] - sub["ice_ref"]

        for sid, grp in sub.groupby("sample_id"):
            grp_sorted = grp.sort_values("feature_value")
            ax.plot(
                grp_sorted["feature_value"], grp_sorted["ice"],
                color="#6699cc", lw=0.7, alpha=ice_alpha,
            )

        if pdp_df is not None:
            sub_pdp = pdp_df[pdp_df["feature"] == feat].sort_values(
                "feature_value")
            if center and not sub_pdp.empty:
                sub_pdp = sub_pdp.copy()
                sub_pdp["pdp"] = sub_pdp["pdp"] - sub_pdp["pdp"].iloc[0]
            ax.plot(
                sub_pdp["feature_value"], sub_pdp["pdp"],
                color="#1f3a5f", lw=2.0, label="PDP",
            )
            ax.legend(loc="best", frameon=False)

        ax.set_title(feat, loc="left", fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("ICE (c-ICE)" if center else "ICE")
        ax.grid(True, alpha=0.25, linestyle="--")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title, fontweight="bold", x=0.02, y=1.02, ha="left")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
    return fig


def top_features_by_shap(
    importance_df: pd.DataFrame,
    *,
    top_k: int = 5,
    importance_col: str = "mean_abs_shap",
) -> list[str]:
    """Helper: retorna as ``top_k`` features mais importantes por SHAP.

    Atalho para selecionar quais features rodar PDP/ICE depois de já ter a
    tabela de importâncias SHAP em mãos.
    """
    if importance_col not in importance_df.columns:
        raise KeyError(
            f"Coluna '{importance_col}' ausente. Encontradas: "
            f"{list(importance_df.columns)}")
    sorted_df = importance_df.sort_values(importance_col, ascending=False)
    return sorted_df["feature"].head(top_k).tolist()
