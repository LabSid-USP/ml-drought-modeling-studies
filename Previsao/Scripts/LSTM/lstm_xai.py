"""
Interpretabilidade (xAI) para modelos LSTM sequenciais.

Três técnicas complementares, todas sem dependências além de PyTorch + numpy:

1. ``compute_temporal_saliency`` — |∂ŷ/∂X| por célula da janela
   (gradient-based).
2. ``compute_perturbation_sensitivity`` — |Δpred| ao mascarar
   X[:, t, f] (ablação local).
3. ``compute_seq_pdp`` — E[ŷ | X_t,f = v] sobre grelha de percentis
   (análogo PDP).

Funções de plotagem:
- ``plot_xai_heatmap``    — heatmap matplotlib (L × p).
- ``plot_seq_pdp_grid``   — curvas PDP por (timestep × feature).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ─── helpers internos ────────────────────────────────────────────────────────

def _to_tensor(X: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(X, dtype=torch.float32, device=device)


# ─── 1. Temporal Saliency Maps ───────────────────────────────────────────────

def compute_temporal_saliency(
    model: nn.Module,
    X_batch: np.ndarray,
    *,
    feature_names: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """Calcula |∂ŷ/∂X| médio sobre as amostras.

    Parameters
    ----------
    model : nn.Module
        Modelo treinado (LSTMRegressor ou similar). Deve estar em modo eval.
    X_batch : np.ndarray, shape (n, L, p)
        Janelas escaladas (mesma escala usada no treino).
    feature_names : list[str]
        Nomes das p features — usado apenas para consistência de shape.
    device : str
        "cpu" (padrão).

    Returns
    -------
    np.ndarray, shape (L, p)
        Saliência média absoluta por (timestep, feature).
        Alto valor → região da janela mais sensível para a previsão.

    Notes
    -----
    Uma backward pass por amostra. Para L=12, p≤12, n=65 amostras de teste:
    ~0.1 s em CPU.
    """
    model = model.to(device).eval()
    n, L, p = X_batch.shape
    if len(feature_names) != p:
        raise ValueError(
            f"feature_names: {len(feature_names)} nomes, X tem p={p}."
        )

    saliency_acc = np.zeros((L, p), dtype=np.float64)

    for i in range(n):
        x = _to_tensor(X_batch[i:i + 1], device).requires_grad_(True)
        pred = model(x)           # (1, 1)
        model.zero_grad()
        pred.sum().backward()
        saliency_acc += x.grad.detach().cpu().numpy()[0]  # (L, p)

    return np.abs(saliency_acc) / n


# ─── 2. Perturbation Sensitivity ────────────────────────────────────────────

def compute_perturbation_sensitivity(
    model: nn.Module,
    X_batch: np.ndarray,
    *,
    feature_names: list[str],
    mask_value: str = "mean",
    device: str = "cpu",
) -> np.ndarray:
    """Mede |Δpred| médio ao mascarar cada célula X[:, t, f].

    Parameters
    ----------
    model : nn.Module
        Modelo treinado em modo eval.
    X_batch : np.ndarray, shape (n, L, p)
        Janelas escaladas.
    feature_names : list[str]
        Nomes das p features.
    mask_value : {"mean", "zero", "noise"}
        Estratégia de mascaramento:
        - ``"mean"``  → substitui por E[X[:, t, f]] (mais realista).
        - ``"zero"``  → substitui por 0 (baseline nula após scaling).
        - ``"noise"`` → adiciona ruído Gaussiano N(0, 0.1).
    device : str
        "cpu" (padrão).

    Returns
    -------
    np.ndarray, shape (L, p)
        Sensibilidade média: quanto a previsão muda ao perturbar (t, f).
    """
    model = model.to(device).eval()
    n, L, p = X_batch.shape
    assert len(feature_names) == p

    X_t = _to_tensor(X_batch, device)
    with torch.no_grad():
        baseline = model(X_t).cpu().numpy().squeeze()   # (n,)

    sensitivity = np.zeros((L, p), dtype=np.float64)

    for t in range(L):
        for f in range(p):
            X_pert = X_t.clone()
            if mask_value == "mean":
                X_pert[:, t, f] = X_t[:, t, f].mean()
            elif mask_value == "zero":
                X_pert[:, t, f] = 0.0
            elif mask_value == "noise":
                X_pert[:, t, f] = X_pert[:, t, f] + torch.randn(n) * 0.1
            else:
                raise ValueError(f"mask_value inválido: {mask_value!r}")

            with torch.no_grad():
                pert_pred = model(X_pert).cpu().numpy().squeeze()   # (n,)

            sensitivity[t, f] = np.abs(baseline - pert_pred).mean()

    return sensitivity


# ─── 3. Seq-PDP (Partial Dependence para sequências) ─────────────────────────

def compute_seq_pdp(
    model: nn.Module,
    X_batch: np.ndarray,
    *,
    feature_names: list[str],
    grid_resolution: int = 40,
    percentile_range: tuple[float, float] = (5.0, 95.0),
    device: str = "cpu",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Dependência marginal E[ŷ | X_t,f = v] para cada (t, f).

    Análogo direto ao PDP do XGBoost (pdp_ice_analysis.py), adaptado para
    janelas 3D: varia X[:, t, f] numa grelha de percentis mantendo tudo
    o mais fixo.

    Parameters
    ----------
    model : nn.Module
        Modelo treinado em modo eval.
    X_batch : np.ndarray, shape (n, L, p)
        Janelas escaladas (normalmente X_test).
    feature_names : list[str]
        Nomes das p features.
    grid_resolution : int
        Número de pontos na grelha de percentis por (t, f).
    percentile_range : tuple[float, float]
        Intervalo de percentis, e.g. (5, 95) evita extrapolação.
    device : str
        "cpu" (padrão).

    Returns
    -------
    pdp_values : np.ndarray, shape (L, p, grid_resolution)
        E[ŷ] médio para cada ponto da grelha.
    feature_grids : list of np.ndarray
        ``feature_grids[t][f]`` → grelha de valores de X[:, t, f].
        Shape de cada elemento: (grid_resolution,).

    Notes
    -----
    Custo: L × p × grid_resolution forward passes, cada sobre n amostras.
    Para L=12, p=10, grid=40, n=65: 4 800 passes × 65 amostras ≈ 10 s em CPU.
    """
    model = model.to(device).eval()
    n, L, p = X_batch.shape
    assert len(feature_names) == p

    pdp_values = np.zeros((L, p, grid_resolution), dtype=np.float64)
    # feature_grids[t][f] é um array 1-D de comprimento grid_resolution
    feature_grids: list[list[np.ndarray]] = [[None] * p for _ in range(L)]

    pcts = np.linspace(
        percentile_range[0], percentile_range[1], grid_resolution
    )

    X_t = _to_tensor(X_batch, device)

    for t in range(L):
        for f in range(p):
            col_vals = X_batch[:, t, f]
            grid = np.percentile(col_vals, pcts)
            feature_grids[t][f] = grid

            for g_idx, g_val in enumerate(grid):
                X_tmp = X_t.clone()
                X_tmp[:, t, f] = float(g_val)
                with torch.no_grad():
                    preds = model(X_tmp).cpu().numpy().squeeze()   # (n,)
                pdp_values[t, f, g_idx] = preds.mean()

    return pdp_values, feature_grids


# ─── Plotagem ────────────────────────────────────────────────────────────────

def plot_xai_heatmap(
    attribution_matrix: np.ndarray,
    *,
    feature_names: list[str],
    title: str = "Attribution Heatmap",
    xlabel: str = "Feature",
    ylabel: str = "Timestep na janela (t=0 = mais antigo)",
    cmap: str = "YlOrRd",
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap (L × p) das atribuições por (timestep, feature).

    Parameters
    ----------
    attribution_matrix : np.ndarray, shape (L, p)
        Saída de ``compute_temporal_saliency`` ou
        ``compute_perturbation_sensitivity``.
    feature_names : list[str]
        Rótulos das colunas (p features).
    title : str
        Título do plot.
    output_path : Path | None
        Se fornecido, salva a figura em PNG 200 dpi.

    Returns
    -------
    plt.Figure
    """
    L, p = attribution_matrix.shape
    assert len(feature_names) == p

    if figsize is None:
        figsize = (max(6, p * 0.9), max(4, L * 0.55))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attribution_matrix, aspect="auto", cmap=cmap,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(p))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"t−{L - 1 - i}" for i in range(L)], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")

    # Anotações de valor
    vmax = attribution_matrix.max()
    for i in range(L):
        for j in range(p):
            v = attribution_matrix[i, j]
            color = "white" if v > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_seq_pdp_grid(
    pdp_values: np.ndarray,
    feature_grids: list,
    *,
    feature_names: list[str],
    timesteps_show: list[int] | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Grade de curvas PDP por (timestep × feature).

    Parameters
    ----------
    pdp_values : np.ndarray, shape (L, p, grid_resolution)
        Saída de ``compute_seq_pdp``.
    feature_grids : list
        ``feature_grids[t][f]`` → grelha de valores para o eixo x.
    feature_names : list[str]
        Rótulos das p features.
    timesteps_show : list[int] | None
        Índices de timesteps a plotar. Se None, usa
        ``[0, L//4, L//2, 3*L//4, L-1]``.
    output_path : Path | None
        Salva em PNG 200 dpi se fornecido.

    Returns
    -------
    plt.Figure
    """
    L, p, _ = pdp_values.shape
    if timesteps_show is None:
        # Distribui uniformemente — máx 5 timesteps para legibilidade
        timesteps_show = sorted(set([0, L // 4, L // 2, 3 * L // 4, L - 1]))

    n_rows = len(timesteps_show)
    if figsize is None:
        figsize = (p * 2.8, n_rows * 2.4)

    fig, axes = plt.subplots(n_rows, p, figsize=figsize, squeeze=False)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_rows))

    for row_i, t in enumerate(timesteps_show):
        lag_label = f"t−{L - 1 - t}"
        for col_j, fname in enumerate(feature_names):
            ax = axes[row_i][col_j]
            grid = feature_grids[t][col_j]
            curve = pdp_values[t, col_j, :]
            ax.plot(grid, curve, color=colors[row_i], lw=1.8)
            ax.axhline(curve.mean(), color="grey", lw=0.7, ls="--", alpha=0.6)
            if row_i == 0:
                ax.set_title(fname, fontsize=9, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(lag_label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Seq-PDP: E[ŷ | X_{t,f} = v]", fontweight="bold", fontsize=11)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
