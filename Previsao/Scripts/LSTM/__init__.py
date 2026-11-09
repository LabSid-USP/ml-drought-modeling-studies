"""Subpacote LSTM — previsão sequencial de volumes via PyTorch.

Re-exporta a API pública para acesso via
``from Previsao.Scripts.LSTM import LSTMRegressor, run_lstm_experiment, ...``.

A API completa também é re-exportada no ``__init__.py`` do pacote pai
(``Previsao.Scripts``) para consistência com o padrão estabelecido pelo
braço XGBoost.

Módulos
-------
* :mod:`sequence_constructor` — janelas 3-D + splits temporais.
* :mod:`sequence_scaling` — scalers fitados apenas no treino.
* :mod:`lstm_dataset` — wrapper :class:`torch.utils.data.Dataset`.
* :mod:`lstm_model` — arquiteturas (LSTM, GRU, MLP-window).
* :mod:`lstm_training` — loop de treino + early stopping.
* :mod:`lstm_experiments` — orquestrador completo (single seed + multi-seed).
* :mod:`lstm_metrics` — RMSE, MAE, corr, NSE, skill direcional.
* :mod:`lstm_plotting` — curvas de aprendizado, bandas multi-seed, resíduos.
"""
from __future__ import annotations

from .sequence_constructor import (
    SequenceSplit,
    build_sequence_supervised,
)
from .sequence_scaling import (
    SequenceScaler,
    fit_sequence_scaler,
    apply_sequence_scaler,
    inverse_transform_y,
)
from .lstm_dataset import SequenceDataset
from .lstm_model import (
    LSTMRegressor,
    GRURegressor,
    MLPWindowRegressor,
)
from .lstm_training import (
    EarlyStopping,
    PinballLoss,
    train_lstm,
    predict_in_batches,
)
from .lstm_experiments import (
    run_lstm_experiment,
    run_lstm_multi_seed,
    summarize_multi_seed,
    run_lstm_quantile_experiment,
    run_lstm_quantile_multi_seed,
)
from .lstm_metrics import (
    rmse,
    mae,
    pearson_corr,
    nse,
    directional_skill_score,
    compute_regression_metrics,
    metrics_by_regime,
    coverage_rate,
    mean_interval_width,
    winkler_score,
)
from .lstm_plotting import (
    plot_training_curves_interactive,
    plot_multi_seed_band_interactive,
    plot_lstm_residuals,
    plot_quantile_band_interactive,
)
from .lstm_xai import (
    compute_temporal_saliency,
    compute_perturbation_sensitivity,
    compute_seq_pdp,
    plot_xai_heatmap,
    plot_seq_pdp_grid,
)


__all__ = [
    # sequence_constructor
    "SequenceSplit", "build_sequence_supervised",
    # sequence_scaling
    "SequenceScaler", "fit_sequence_scaler", "apply_sequence_scaler",
    "inverse_transform_y",
    # lstm_dataset
    "SequenceDataset",
    # lstm_model
    "LSTMRegressor", "GRURegressor", "MLPWindowRegressor",
    # lstm_training
    "EarlyStopping", "PinballLoss", "train_lstm", "predict_in_batches",
    # lstm_experiments
    "run_lstm_experiment", "run_lstm_multi_seed", "summarize_multi_seed",
    "run_lstm_quantile_experiment", "run_lstm_quantile_multi_seed",
    # lstm_metrics
    "rmse", "mae", "pearson_corr", "nse", "directional_skill_score",
    "compute_regression_metrics", "metrics_by_regime",
    "coverage_rate", "mean_interval_width", "winkler_score",
    # lstm_plotting
    "plot_training_curves_interactive", "plot_multi_seed_band_interactive",
    "plot_lstm_residuals", "plot_quantile_band_interactive",
    # lstm_xai
    "compute_temporal_saliency", "compute_perturbation_sensitivity",
    "compute_seq_pdp", "plot_xai_heatmap", "plot_seq_pdp_grid",
]
