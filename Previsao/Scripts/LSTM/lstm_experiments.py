"""
Orquestrador de experimentos LSTM/GRU/MLP.

Este módulo provê a função pública :func:`run_lstm_experiment` que
encapsula o pipeline completo:

1. Constrói as janelas via :func:`build_sequence_supervised`.
2. Escala features/target via :func:`fit_sequence_scaler` /
   :func:`apply_sequence_scaler`.
3. Empacota nos :class:`SequenceDataset` + ``DataLoader``.
4. Instancia o modelo (``LSTMRegressor`` / ``GRURegressor`` / ``MLPWindowRegressor``).
5. Treina via :func:`train_lstm` (com early stopping).
6. Prediz em todos os splits, inverte scaling de ``y``, reconstrói volume
   absoluto quando ``target_mode="delta"``.
7. Calcula métricas + monta ``preds_df`` no schema do XGBoost.

Schema do ``preds_df`` produzido
--------------------------------
Idêntico ao XGBoost para que **todos os plots e baselines existentes
funcionem sem modificação**:

================  ==========================================================
Coluna            Significado
================  ==========================================================
``issue_date``    Data $t$ do último mês da janela de entrada.
``target_date``   Data $t+H$ do alvo.
``y_true``        Volume observado em $t+H$, na escala física (hm³).
``y_pred``        Volume previsto em $t+H$, escala física.
``y_base``        Persistência: $\\text{Vol}(t)$ (escala física).
``split``         ``"train"`` / ``"val"`` / ``"test"``.
``experiment``    Nome do experimento (e.g. ``"LSTM_A_min"``).
``basin``         Código da bacia.
``target_mode``   ``"level"`` ou ``"delta"`` — usado pelo
                  ``seasonal_climatology_baseline``.
``seed``          Semente usada (relevante para multi-seed).
``model_kind``    ``"lstm"`` / ``"gru"`` / ``"mlp"`` para auditoria.
================  ==========================================================

Função multi-seed
-----------------
:func:`run_lstm_multi_seed` simplesmente itera sobre uma lista de seeds
e concatena ``preds_df``. Útil para medir incerteza de inicialização —
critério visual via :func:`plot_multi_seed_band_interactive`.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .lstm_dataset import SequenceDataset
from .lstm_metrics import compute_regression_metrics
from .lstm_model import GRURegressor, LSTMRegressor, MLPWindowRegressor
from .lstm_training import PinballLoss, predict_in_batches, train_lstm
from .sequence_constructor import build_sequence_supervised
from .sequence_scaling import (
    apply_sequence_scaler,
    fit_sequence_scaler,
    inverse_transform_y,
)


# --------------------------------------------------------------------------- #
# Fábrica de modelos
# --------------------------------------------------------------------------- #
def _build_model(
    model_kind: Literal["lstm", "gru", "mlp"],
    *,
    input_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    head_hidden: int,
    head_output_size: int = 1,
) -> torch.nn.Module:
    if model_kind == "lstm":
        return LSTMRegressor(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            head_hidden=head_hidden,
            head_output_size=head_output_size,
        )
    if model_kind == "gru":
        return GRURegressor(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            head_hidden=head_hidden,
            head_output_size=head_output_size,
        )
    if model_kind == "mlp":
        return MLPWindowRegressor(
            input_size=input_size, seq_len=seq_len,
            hidden_sizes=[hidden_size * 4, hidden_size],
            dropout=dropout,
            head_output_size=head_output_size,
        )
    raise ValueError(
        f"model_kind inválido: {model_kind!r}. "
        "Use 'lstm', 'gru' ou 'mlp'."
    )


# --------------------------------------------------------------------------- #
# Núcleo: run_lstm_experiment (single seed)
# --------------------------------------------------------------------------- #
def run_lstm_experiment(
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    *,
    # Dados / split
    basin: str,
    feature_cols: Sequence[str],
    target_mode: Literal["level", "delta"],
    seq_len: int = 12,
    horizon: int = 3,
    min_date: str | pd.Timestamp,
    max_date: str | pd.Timestamp,
    split_on: str = "target_date",
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    core_required: Sequence[str] | None = None,
    # Modelo
    model_kind: Literal["lstm", "gru", "mlp"] = "lstm",
    hidden_size: int = 16,
    num_layers: int = 1,
    dropout: float = 0.0,
    head_hidden: int = 8,
    # Treino
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 30,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    scale_y: bool = True,
    # Metadados
    experiment_name: str = "LSTM_A_min",
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Roda um experimento completo (single seed) e devolve métricas + predições.

    :returns: tupla ``(results_df, preds_df, fitted)``:

        * ``results_df`` (1 linha): métricas agregadas no teste.
        * ``preds_df`` (n linhas): predições por amostra, schema XGBoost.
        * ``fitted`` (dict): artefatos para inspeção
          (``model``, ``history``, ``scaler``, ``meta``).
    """
    # 1) Constrói janelas
    X_seq, y, split = build_sequence_supervised(
        datasets_by_basin=datasets_by_basin, volumes=volumes,
        basin=basin, feature_cols=list(feature_cols),
        horizon=horizon, seq_len=seq_len, target_mode=target_mode,
        min_date=min_date, max_date=max_date,
        split_on=split_on, train_end=train_end, val_end=val_end,
        core_required=core_required,
    )

    train_mask = split.train
    val_mask = split.val
    test_mask = split.test

    X_train = X_seq[train_mask]
    y_train = y[train_mask]
    X_val = X_seq[val_mask] if val_mask is not None else X_seq[:0]
    y_val = y[val_mask] if val_mask is not None else y[:0]
    X_test = X_seq[test_mask]
    y_test = y[test_mask]

    # 2) Scaling (ajustado SOMENTE no treino)
    scaler = fit_sequence_scaler(
        X_train, y_train,
        feature_names=list(feature_cols), scale_y=scale_y,
    )
    X_train_s, y_train_s = apply_sequence_scaler(scaler, X_train, y_train)
    X_val_s, y_val_s = apply_sequence_scaler(scaler, X_val, y_val)
    X_test_s, y_test_s = apply_sequence_scaler(scaler, X_test, y_test)

    # 3) DataLoaders
    train_loader = DataLoader(
        SequenceDataset(X_train_s, y_train_s),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val_s, y_val_s),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test_s, y_test_s),
        batch_size=batch_size, shuffle=False,
    )

    # 4) Modelo
    n_features = len(feature_cols)
    model = _build_model(
        model_kind,
        input_size=n_features, seq_len=seq_len,
        hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, head_hidden=head_hidden,
    )

    # 5) Treino
    model, history = train_lstm(
        model, train_loader, val_loader,
        max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, grad_clip=grad_clip,
        seed=seed, device=device, verbose=verbose,
    )

    # 6) Predições em todos os splits (com batch_size para eficiência)
    pred_train_s = predict_in_batches(model, train_loader, device=device)
    pred_val_s = predict_in_batches(model, val_loader, device=device)
    pred_test_s = predict_in_batches(model, test_loader, device=device)

    # ATENÇÃO: train_loader tem shuffle=True → ordem perdida. Re-prever sem shuffle.
    train_loader_ordered = DataLoader(
        SequenceDataset(X_train_s, y_train_s),
        batch_size=batch_size, shuffle=False,
    )
    pred_train_s = predict_in_batches(
        model, train_loader_ordered, device=device,
    )

    # 7) Inversão de scaling de y (se aplicado)
    if scaler.target_scaler is not None:
        pred_train = inverse_transform_y(scaler, pred_train_s)
        pred_val = (
            inverse_transform_y(scaler, pred_val_s)
            if len(pred_val_s) else pred_val_s
        )
        pred_test = inverse_transform_y(scaler, pred_test_s)
    else:
        pred_train = pred_train_s
        pred_val = pred_val_s
        pred_test = pred_test_s

    # 8) Reconstrução para volume absoluto (modo delta)
    meta_all = split.meta.copy()
    meta_train = meta_all[train_mask].reset_index(drop=True)
    meta_val = (
        meta_all[val_mask].reset_index(drop=True)
        if val_mask is not None
        else meta_all.iloc[:0]
    )
    meta_test = meta_all[test_mask].reset_index(drop=True)

    if target_mode == "delta":
        y_pred_level_train = meta_train["Vol_t"].to_numpy() + pred_train
        y_pred_level_val = (
            meta_val["Vol_t"].to_numpy() + pred_val
            if len(pred_val) else pred_val
        )
        y_pred_level_test = meta_test["Vol_t"].to_numpy() + pred_test
    else:
        y_pred_level_train = pred_train
        y_pred_level_val = pred_val
        y_pred_level_test = pred_test

    # 9) Monta preds_df no schema XGBoost
    rows = []
    for meta_split, y_pred_level, split_name in [
        (meta_train, y_pred_level_train, "train"),
        (meta_val, y_pred_level_val, "val"),
        (meta_test, y_pred_level_test, "test"),
    ]:
        if len(meta_split) == 0:
            continue
        rows.append(pd.DataFrame({
            "issue_date": meta_split["issue_date"].values,
            "target_date": meta_split["target_date"].values,
            "y_true": meta_split["y_true_level"].values,
            "y_pred": y_pred_level,
            "y_base": meta_split["Vol_t"].values,  # persistência
            "split": split_name,
            "experiment": experiment_name,
            "basin": basin,
            "target_mode": target_mode,
            "seed": seed,
            "model_kind": model_kind,
        }))
    preds_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # 10) Métricas de teste (volume reconstruído)
    test_metrics = compute_regression_metrics(
        meta_test["y_true_level"].to_numpy(),
        y_pred_level_test,
        y_base=meta_test["Vol_t"].to_numpy(),
    )
    if len(meta_val) > 0:
        val_metrics = compute_regression_metrics(
            meta_val["y_true_level"].to_numpy(),
            y_pred_level_val,
            y_base=meta_val["Vol_t"].to_numpy(),
        )
    else:
        val_metrics = {}

    results_row = {
        "experiment": experiment_name,
        "basin": basin,
        "target_mode": target_mode,
        "seed": seed,
        "model_kind": model_kind,
        "horizon": horizon,
        "seq_len": seq_len,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()) if val_mask is not None else 0,
        "n_test": int(test_mask.sum()),
        "best_epoch": (
            int(history["val_loss"].idxmin()) if len(history) else -1
        ),
        "best_val_loss": (
            float(history["val_loss"].min()) if len(history) else float("nan")
        ),
        "n_epochs_trained": int(len(history)),
        # Métricas teste (já com sufixo)
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_corr": test_metrics["corr"],
        "test_nse": test_metrics["nse"],
        "test_dss": test_metrics["dss"],
        "test_rmse_persist": test_metrics["rmse_base"],
        "test_corr_persist": test_metrics["corr_base"],
        # Validação
        "val_rmse": val_metrics.get("rmse", float("nan")),
        "val_corr": val_metrics.get("corr", float("nan")),
    }
    results_df = pd.DataFrame([results_row])

    fitted = {
        "model": model,
        "history": history,
        "scaler": scaler,
        "meta": split.meta,
        "split": split,
        "X_test": X_test_s,
        "feature_cols": list(feature_cols),
    }
    return results_df, preds_df, fitted


# --------------------------------------------------------------------------- #
# Multi-seed: itera run_lstm_experiment com diferentes seeds
# --------------------------------------------------------------------------- #
def run_lstm_multi_seed(
    *args,
    seeds: Sequence[int] = (42, 7, 123, 2024, 99),
    verbose: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Roda :func:`run_lstm_experiment` para múltiplas seeds.

    :param seeds: lista/tupla de inteiros — uma execução completa por seed.
    :param args, kwargs: repassados para ``run_lstm_experiment`` (exceto
        ``seed``, que é fornecida pelo loop).
    :returns: tupla ``(results_df, preds_df, fitted_list)``:

        * ``results_df``: uma linha por seed, com a coluna ``seed``.
        * ``preds_df``: concatenação das predições — schema XGBoost +
          coluna ``seed``.
        * ``fitted_list``: lista de artefatos (model, history, scaler) por seed.
    """
    if "seed" in kwargs:
        raise ValueError(
            "Não passe 'seed' em run_lstm_multi_seed — use o parâmetro 'seeds'."
        )
    results_parts: list[pd.DataFrame] = []
    preds_parts: list[pd.DataFrame] = []
    fitted_list: list[dict] = []

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        res, preds, fitted = run_lstm_experiment(
            *args, seed=seed, verbose=verbose, **kwargs,
        )
        results_parts.append(res)
        preds_parts.append(preds)
        fitted_list.append(fitted)

    results_df = pd.concat(results_parts, ignore_index=True)
    preds_df = pd.concat(preds_parts, ignore_index=True)
    return results_df, preds_df, fitted_list


# --------------------------------------------------------------------------- #
# Resumo agregado multi-seed: média ± std das métricas de teste
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Regressão Quantílica: run_lstm_quantile_experiment
# --------------------------------------------------------------------------- #
def run_lstm_quantile_experiment(
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    *,
    # Dados / split
    basin: str,
    feature_cols: Sequence[str],
    target_mode: Literal["level", "delta"],
    seq_len: int = 12,
    horizon: int = 3,
    min_date: str | pd.Timestamp,
    max_date: str | pd.Timestamp,
    split_on: str = "target_date",
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    core_required: Sequence[str] | None = None,
    # Modelo
    model_kind: Literal["lstm", "gru", "mlp"] = "lstm",
    hidden_size: int = 16,
    num_layers: int = 1,
    dropout: float = 0.0,
    head_hidden: int = 8,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    # Treino
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 30,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    scale_y: bool = True,
    # Metadados
    experiment_name: str = "LSTM_QR",
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Regressão Quantílica: prevê P10/P50/P90 simultaneamente.

    Análogo direto a :func:`run_lstm_experiment`, com duas diferenças:

    * O head do modelo tem ``len(quantiles)`` saídas em vez de 1
      (``head_output_size = len(quantiles)``).
    * A perda usada é :class:`PinballLoss` em vez de ``MSELoss``.

    :param quantiles: quantis a otimizar. Padrão ``(0.1, 0.5, 0.9)`` para
        P10/P50/P90. A ordem não importa — as saídas são ordenadas
        (non-crossing) antes de devolver.

    :returns: tupla ``(results_df, preds_df, fitted)``:

        * ``preds_df`` tem colunas adicionais ``y_pred_p10``,
          ``y_pred_p50``, ``y_pred_p90``. ``y_pred = y_pred_p50`` para
          compatibilidade com plots e métricas existentes.
        * ``results_df`` inclui ``coverage_80``, ``mean_width`` e
          ``winkler_80`` calculados no teste.
    """
    from .lstm_metrics import (
        coverage_rate as _coverage,
        mean_interval_width as _mwidth,
        winkler_score as _winkler,
    )

    quantiles_list = sorted(quantiles)
    n_q = len(quantiles_list)
    # Índices de P10 (menor) e P90 (maior) após ordenação
    idx_lo, idx_hi = 0, n_q - 1
    # Índice do quantil mediano (P50 quando quantiles=[0.1,0.5,0.9])
    idx_med = n_q // 2

    # 1) Janelas
    X_seq, y, split = build_sequence_supervised(
        datasets_by_basin=datasets_by_basin, volumes=volumes,
        basin=basin, feature_cols=list(feature_cols),
        horizon=horizon, seq_len=seq_len, target_mode=target_mode,
        min_date=min_date, max_date=max_date,
        split_on=split_on, train_end=train_end, val_end=val_end,
        core_required=core_required,
    )

    train_mask = split.train
    val_mask = split.val
    test_mask = split.test

    X_train = X_seq[train_mask]
    y_train = y[train_mask]
    X_val = X_seq[val_mask] if val_mask is not None else X_seq[:0]
    y_val = y[val_mask] if val_mask is not None else y[:0]
    X_test = X_seq[test_mask]

    # 2) Scaling (ajustado SOMENTE no treino)
    scaler = fit_sequence_scaler(
        X_train, y_train,
        feature_names=list(feature_cols), scale_y=scale_y,
    )
    X_train_s, y_train_s = apply_sequence_scaler(scaler, X_train, y_train)
    X_val_s, y_val_s = apply_sequence_scaler(scaler, X_val, y_val)
    X_test_s, _ = apply_sequence_scaler(scaler, X_test, None)

    # 3) DataLoaders
    train_loader = DataLoader(
        SequenceDataset(X_train_s, y_train_s),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val_s, y_val_s),
        batch_size=batch_size, shuffle=False,
    )

    # 4) Modelo com n_q saídas (uma por quantil)
    model = _build_model(
        model_kind,
        input_size=len(feature_cols), seq_len=seq_len,
        hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, head_hidden=head_hidden,
        head_output_size=n_q,
    )

    # 5) Treino com PinballLoss
    loss_fn = PinballLoss(quantiles_list)
    model, history = train_lstm(
        model, train_loader, val_loader,
        max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, grad_clip=grad_clip,
        loss_fn=loss_fn, seed=seed, device=device, verbose=verbose,
    )

    # 6) Predições em batches — shape (n, n_q) em escala normalizada
    def _predict_multi(loader: DataLoader) -> np.ndarray:
        """Retorna array (n, n_q) com predições para cada quantil."""
        model.eval()
        parts = []
        with torch.no_grad():
            for Xb, _ in loader:
                out = model(Xb.to(device)).cpu().numpy()  # (batch, n_q)
                parts.append(out)
        if not parts:
            return np.zeros((0, n_q), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    # y dummy para montar datasets de val/test (rótulos não usados em predict)
    y_dummy_val = np.zeros(len(X_val_s), dtype=np.float32)
    y_dummy_test = np.zeros(len(X_test_s), dtype=np.float32)
    train_loader_ord = DataLoader(
        SequenceDataset(X_train_s, y_train_s),
        batch_size=batch_size, shuffle=False,
    )
    val_loader_pred = DataLoader(
        SequenceDataset(X_val_s, y_dummy_val),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test_s, y_dummy_test),
        batch_size=batch_size, shuffle=False,
    )

    pred_train_s = _predict_multi(train_loader_ord)
    pred_val_s = _predict_multi(val_loader_pred)
    pred_test_s = _predict_multi(test_loader)

    # 7) Non-crossing: garante P10 ≤ P50 ≤ P90 por amostra
    pred_train_s = np.sort(pred_train_s, axis=1)
    pred_val_s = np.sort(pred_val_s, axis=1)
    pred_test_s = np.sort(pred_test_s, axis=1)

    # 8) Inversão de scaling para cada coluna de quantil
    def _inverse_col(arr_s: np.ndarray) -> np.ndarray:
        """Aplica inverse_transform_y coluna a coluna (n,n_q → n,n_q)."""
        if scaler.target_scaler is None:
            return arr_s
        cols = [inverse_transform_y(scaler, arr_s[:, j]) for j in range(n_q)]
        return np.stack(cols, axis=1)

    pred_train = _inverse_col(pred_train_s)
    pred_val = _inverse_col(pred_val_s) if len(pred_val_s) else pred_val_s
    pred_test = _inverse_col(pred_test_s)

    # 9) Reconstrução delta → volume absoluto
    meta_all = split.meta.copy()
    meta_train = meta_all[train_mask].reset_index(drop=True)
    meta_val = (
        meta_all[val_mask].reset_index(drop=True)
        if val_mask is not None
        else meta_all.iloc[:0]
    )
    meta_test = meta_all[test_mask].reset_index(drop=True)

    def _reconstruct(pred_nq, meta):
        if target_mode == "delta":
            base = meta["Vol_t"].to_numpy()[:, None]  # (n, 1) broadcast
            return pred_nq + base
        return pred_nq

    level_train = _reconstruct(pred_train, meta_train)
    level_val = _reconstruct(pred_val, meta_val) if len(pred_val) else pred_val
    level_test = _reconstruct(pred_test, meta_test)

    # 10) Monta preds_df com colunas P10/P50/P90
    rows = []
    for meta_sp, lvl, sp_name in [
        (meta_train, level_train, "train"),
        (meta_val, level_val, "val"),
        (meta_test, level_test, "test"),
    ]:
        if len(meta_sp) == 0:
            continue
        _row = {
            "issue_date": meta_sp["issue_date"].values,
            "target_date": meta_sp["target_date"].values,
            "y_true": meta_sp["y_true_level"].values,
            "y_pred": lvl[:, idx_med],          # P50 → compatível com plots
            "y_pred_p10": lvl[:, idx_lo],
            "y_pred_p50": lvl[:, idx_med],
            "y_pred_p90": lvl[:, idx_hi],
            "y_base": meta_sp["Vol_t"].values,
            "split": sp_name,
            "experiment": experiment_name,
            "basin": basin,
            "target_mode": target_mode,
            "seed": seed,
            "model_kind": model_kind,
        }
        # Expõe TODOS os quantis (colunas ``y_pred_qNN``) além de p10/p50/p90
        # — habilita curvas de confiabilidade/nitidez e o alpha index sem
        # quebrar código que usa p10/p50/p90 (retrocompatível).
        for _i, _q in enumerate(quantiles_list):
            _row[f"y_pred_q{int(round(_q * 100)):02d}"] = lvl[:, _i]
        rows.append(pd.DataFrame(_row))
    preds_df = (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    )

    # 11) Métricas de teste
    y_true_test = meta_test["y_true_level"].to_numpy()
    y_base_test = meta_test["Vol_t"].to_numpy()
    test_metrics = {
        "rmse_p50": float(np.sqrt(
            np.mean((y_true_test - level_test[:, idx_med]) ** 2)
        )),
        "coverage_80": _coverage(
            y_true_test, level_test[:, idx_lo], level_test[:, idx_hi],
        ),
        "mean_width": _mwidth(level_test[:, idx_lo], level_test[:, idx_hi]),
        "winkler_80": _winkler(
            y_true_test, level_test[:, idx_lo], level_test[:, idx_hi],
        ),
        "rmse_persist": float(np.sqrt(
            np.mean((y_true_test - y_base_test) ** 2)
        )),
    }

    results_row = {
        "experiment": experiment_name,
        "basin": basin,
        "target_mode": target_mode,
        "seed": seed,
        "model_kind": model_kind,
        "horizon": horizon,
        "seq_len": seq_len,
        "quantiles": str(quantiles_list),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()) if val_mask is not None else 0,
        "n_test": int(test_mask.sum()),
        "best_epoch": (
            int(history["val_loss"].idxmin()) if len(history) else -1
        ),
        "best_val_loss": (
            float(history["val_loss"].min()) if len(history) else float("nan")
        ),
        "n_epochs_trained": int(len(history)),
        **test_metrics,
    }
    results_df = pd.DataFrame([results_row])

    fitted = {
        "model": model,
        "history": history,
        "scaler": scaler,
        "meta": split.meta,
        "split": split,
        "X_test": X_test_s,
        "feature_cols": list(feature_cols),
        "quantiles": quantiles_list,
    }
    return results_df, preds_df, fitted


def run_lstm_quantile_multi_seed(
    *args,
    seeds: Sequence[int] = (42, 7, 123, 2024, 99),
    verbose: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Roda :func:`run_lstm_quantile_experiment` para múltiplas seeds.

    :param seeds: lista/tupla de inteiros — uma execução completa por seed.
    :param args, kwargs: repassados para ``run_lstm_quantile_experiment``
        (exceto ``seed``, fornecida pelo loop).
    :returns: tupla ``(results_df, preds_df, fitted_list)``:

        * ``results_df``: uma linha por seed.
        * ``preds_df``: concatenação das predições com colunas
          ``y_pred_p10``, ``y_pred_p50``, ``y_pred_p90``.
        * ``fitted_list``: lista de dicionários de artefatos por seed.
    """
    if "seed" in kwargs:
        raise ValueError(
            "Não passe 'seed' em run_lstm_quantile_multi_seed"
            " — use o parâmetro 'seeds'."
        )
    results_parts: list[pd.DataFrame] = []
    preds_parts: list[pd.DataFrame] = []
    fitted_list: list[dict] = []

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        res, preds, fitted = run_lstm_quantile_experiment(
            *args, seed=seed, verbose=verbose, **kwargs,
        )
        results_parts.append(res)
        preds_parts.append(preds)
        fitted_list.append(fitted)

    results_df = pd.concat(results_parts, ignore_index=True)
    preds_df = pd.concat(preds_parts, ignore_index=True)
    return results_df, preds_df, fitted_list


# --------------------------------------------------------------------------- #
# Resumo agregado multi-seed: média ± std das métricas de teste
# --------------------------------------------------------------------------- #
def summarize_multi_seed(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula média e desvio das métricas de teste agrupando por experimento.

    :param results_df: output de :func:`run_lstm_multi_seed` (uma linha
        por seed).
    :returns: DataFrame com colunas ``mean`` e ``std`` para cada métrica
        relevante, indexado por ``experiment``/``basin``/``model_kind``.
    """
    metric_cols = [
        "test_rmse", "test_mae", "test_corr", "test_nse", "test_dss",
        "best_epoch", "n_epochs_trained",
    ]
    group_cols = ["experiment", "basin", "target_mode", "model_kind"]
    agg = (
        results_df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .round(3)
    )
    # Achata o multi-index de colunas
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()
