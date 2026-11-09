"""Pacote Scripts — utilitários da pesquisa de previsão do Sistema Cantareira.

Centraliza imports para serem usados em notebooks como::

    from Previsao.Scripts import (
        XGBoostModel, FeatureConstructor, run_experiments,
        build_total_cantareira_inputs, compute_pdp,
        LSTMRegressor, run_lstm_experiment, build_sequence_supervised,
        ...
    )

Subpacotes
----------
* :mod:`Previsao.Scripts.XGBoost` — pipeline tabular consolidado:
  feature engineering, modelo XGBoost, experimentos, SHAP, PDP/ICE e plots.
* :mod:`Previsao.Scripts.LSTM` — braço sequencial PyTorch: construção de
  janelas, scaling 3-D, modelos recorrentes/MLP, loop de treino com
  early stopping, métricas e plots multi-seed.

Módulos XGBoost
---------------
* :mod:`feature_engineering`     — utilitários low-level (lags, rolling,
  expanding, SPI/SPEI/SRI).
* :mod:`feature_constructor`     — orquestração de features por bacia.
* :mod:`feature_sets`            — feature sets A/B/C por bacia.
* :mod:`cantareira_equivalent`   — agregação de bacias.
* :mod:`system_features`         — features de estado sistêmico.
* :mod:`xgboost_model`           — modelo XGBoost X(t)→y(t+H).
* :mod:`experiments`             — orchestração de experimentos +
  baselines (persistência, climatologia sazonal).
* :mod:`shap_analysis`           — interpretabilidade via TreeSHAP.
* :mod:`pdp_ice_analysis`        — PDP e ICE.
* :mod:`plotting` / :mod:`plotting_interactive` — figuras estáticas e
  interativas.

Módulos LSTM
------------
Consulte :mod:`Previsao.Scripts.LSTM` para o catálogo completo.

Avaliação comum
---------------
* :mod:`forecast_evaluation`     — teste de Diebold–Mariano (HLN).
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Subpacote XGBoost — re-exporta sob o namespace plano histórico
# --------------------------------------------------------------------------- #
from .XGBoost.xgboost_model import SplitMasks, XGBoostModel

from .XGBoost.feature_engineering import FeatureEngineering
from .XGBoost.feature_constructor import FeatureConstructor, OpsRouting
from .XGBoost.feature_sets import get_experiments_abc

from .XGBoost.cantareira_equivalent import (
    CANTAREIRA_AREA_KM2,
    CANTAREIRA_AREA_KM2_PDAA,
    BasinAggregationSpec,
    MultiBasinAggregationSpec,
    aggregate_many_basins_wide,
    aggregate_two_basins_wide,
    area_share,
    build_equivalent_cantareira_inputs,
    build_total_cantareira_inputs,
    default_cach_atib_spec_mm,
    default_cantareira_total_spec,
)

from .XGBoost.system_features import add_cantareira_system_storage_features

from .XGBoost.experiments import (
    ExperimentResult,
    run_experiments,
    run_walk_forward_expanding,
    seasonal_climatology_baseline,
)

from .XGBoost.shap_analysis import (
    FittedExperiment,
    assign_feature_group,
    compute_tree_shap,
    fit_single_experiment_for_shap,
    get_model_matrix,
    save_shap_beeswarm,
    shap_group_importance_table,
    shap_importance_table,
)

from .XGBoost.pdp_ice_analysis import (
    compute_ice,
    compute_pdp,
    plot_ice_static,
    plot_pdp_static,
    top_features_by_shap,
)

from .XGBoost.plotting import (
    build_system_totals,
    plot_benchmark_scatter_1x2,
    plot_benchmark_scatter_2x1,
    plot_benchmark_timeseries_1x2,
    plot_benchmark_timeseries_2x1,
    plot_error_boxplot,
    plot_residuals,
    plot_scatter,
    plot_shap_group_bar,
    plot_shap_importance_bar,
    plot_timeseries,
    plot_timeseries_compare_experiments,
    save_figure,
    set_paper_plot_style,
    set_plot_style,
)

from .XGBoost.plotting_interactive import (
    plot_data_availability_heatmap,
    plot_pdp_ice_interactive,
    plot_scatter_interactive,
    plot_shap_beeswarm_interactive,
    plot_timeseries_interactive,
    shap_to_long,
)

# --------------------------------------------------------------------------- #
# Subpacote LSTM — braço sequencial (PyTorch)
# --------------------------------------------------------------------------- #
from .LSTM.sequence_constructor import (
    SequenceSplit,
    build_sequence_supervised,
)
from .LSTM.sequence_scaling import (
    SequenceScaler,
    fit_sequence_scaler,
    apply_sequence_scaler,
    inverse_transform_y,
)
from .LSTM.lstm_dataset import SequenceDataset
from .LSTM.lstm_model import (
    LSTMRegressor,
    GRURegressor,
    MLPWindowRegressor,
)
from .LSTM.lstm_training import (
    EarlyStopping,
    PinballLoss,
    train_lstm,
    predict_in_batches,
)
from .LSTM.lstm_experiments import (
    run_lstm_experiment,
    run_lstm_multi_seed,
    summarize_multi_seed,
    run_lstm_quantile_experiment,
    run_lstm_quantile_multi_seed,
)
from .LSTM.lstm_metrics import (
    rmse as lstm_rmse,
    mae as lstm_mae,
    pearson_corr,
    nse,
    directional_skill_score,
    compute_regression_metrics,
    metrics_by_regime,
    coverage_rate,
    mean_interval_width,
    winkler_score,
)
from .LSTM.lstm_plotting import (
    plot_training_curves_interactive,
    plot_multi_seed_band_interactive,
    plot_lstm_residuals,
    plot_quantile_band_interactive,
)
from .LSTM.lstm_xai import (
    compute_temporal_saliency,
    compute_perturbation_sensitivity,
    compute_seq_pdp,
    plot_xai_heatmap,
    plot_seq_pdp_grid,
)

# --------------------------------------------------------------------------- #
# Avaliação comum aos dois braços
# --------------------------------------------------------------------------- #
from .forecast_evaluation import diebold_mariano_test


__all__ = [
    # XGBoost — modelo & features
    "SplitMasks", "XGBoostModel",
    "FeatureEngineering", "FeatureConstructor", "OpsRouting",
    "get_experiments_abc",
    # XGBoost — agregação
    "CANTAREIRA_AREA_KM2", "CANTAREIRA_AREA_KM2_PDAA",
    "BasinAggregationSpec", "MultiBasinAggregationSpec",
    "aggregate_many_basins_wide", "aggregate_two_basins_wide",
    "area_share",
    "build_equivalent_cantareira_inputs", "build_total_cantareira_inputs",
    "default_cach_atib_spec_mm", "default_cantareira_total_spec",
    "add_cantareira_system_storage_features",
    # XGBoost — experimentos & baselines
    "ExperimentResult", "run_experiments", "run_walk_forward_expanding",
    "seasonal_climatology_baseline",
    # XGBoost — SHAP / PDP / ICE
    "FittedExperiment", "assign_feature_group", "compute_tree_shap",
    "fit_single_experiment_for_shap", "get_model_matrix",
    "save_shap_beeswarm", "shap_group_importance_table",
    "shap_importance_table",
    "compute_ice", "compute_pdp", "plot_ice_static", "plot_pdp_static",
    "top_features_by_shap",
    # XGBoost — plots
    "build_system_totals",
    "plot_benchmark_scatter_1x2", "plot_benchmark_scatter_2x1",
    "plot_benchmark_timeseries_1x2", "plot_benchmark_timeseries_2x1",
    "plot_error_boxplot", "plot_residuals", "plot_scatter",
    "plot_shap_group_bar", "plot_shap_importance_bar",
    "plot_timeseries", "plot_timeseries_compare_experiments",
    "save_figure", "set_paper_plot_style", "set_plot_style",
    "plot_data_availability_heatmap", "plot_pdp_ice_interactive",
    "plot_scatter_interactive", "plot_shap_beeswarm_interactive",
    "plot_timeseries_interactive", "shap_to_long",
    # LSTM — núcleo sequencial
    "SequenceSplit", "build_sequence_supervised",
    "SequenceScaler", "fit_sequence_scaler", "apply_sequence_scaler",
    "inverse_transform_y", "SequenceDataset",
    # LSTM — modelos
    "LSTMRegressor", "GRURegressor", "MLPWindowRegressor",
    # LSTM — treino & experimentos
    "EarlyStopping", "PinballLoss", "train_lstm", "predict_in_batches",
    "run_lstm_experiment", "run_lstm_multi_seed", "summarize_multi_seed",
    "run_lstm_quantile_experiment", "run_lstm_quantile_multi_seed",
    # LSTM — métricas (renomeadas para evitar colisão com módulos sklearn)
    "lstm_rmse", "lstm_mae", "pearson_corr", "nse",
    "directional_skill_score", "compute_regression_metrics",
    "metrics_by_regime",
    "coverage_rate", "mean_interval_width", "winkler_score",
    # LSTM — plots
    "plot_training_curves_interactive", "plot_multi_seed_band_interactive",
    "plot_lstm_residuals", "plot_quantile_band_interactive",
    # LSTM — xAI
    "compute_temporal_saliency", "compute_perturbation_sensitivity",
    "compute_seq_pdp", "plot_xai_heatmap", "plot_seq_pdp_grid",
    # Avaliação comum
    "diebold_mariano_test",
]
