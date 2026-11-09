"""Subpacote XGBoost — pipeline tabular de previsão do Sistema Cantareira.

Re-exporta a API pública para acesso via
``from Previsao.Scripts.XGBoost import XGBoostModel, run_experiments, ...``.

A API completa também é re-exportada no ``__init__.py`` do pacote pai
(``Previsao.Scripts``) para consistência com o padrão histórico dos notebooks.

Módulos
-------
* :mod:`xgboost_model`        — modelo XGBoost X(t)→y(t+H); ``SplitMasks``.
* :mod:`feature_engineering`  — utilitários low-level (lags, rolling,
  expanding, SPI/SPEI/SRI).
* :mod:`feature_constructor`  — orquestração de features por bacia;
  ``FeatureConstructor``, ``OpsRouting``.
* :mod:`feature_sets`         — feature sets A/B/C por bacia;
  ``get_experiments_abc``.
* :mod:`cantareira_equivalent` — agregação de bacias; áreas de drenagem.
* :mod:`system_features`      — features de estado sistêmico do Cantareira.
* :mod:`experiments`          — orquestrador de experimentos + baselines
  (persistência, climatologia sazonal).
* :mod:`shap_analysis`        — interpretabilidade via TreeSHAP.
* :mod:`pdp_ice_analysis`     — PDP e ICE.
* :mod:`plotting`             — figuras estáticas paper-ready (matplotlib).
* :mod:`plotting_interactive` — exploração interativa (plotly + ipywidgets).
"""
from __future__ import annotations

from .xgboost_model import SplitMasks, XGBoostModel

from .feature_engineering import FeatureEngineering

from .feature_constructor import FeatureConstructor, OpsRouting

from .feature_sets import get_experiments_abc

from .cantareira_equivalent import (
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

from .system_features import add_cantareira_system_storage_features

from .experiments import (
    ExperimentResult,
    run_experiments,
    run_walk_forward_expanding,
    seasonal_climatology_baseline,
)

from .shap_analysis import (
    FittedExperiment,
    assign_feature_group,
    compute_tree_shap,
    fit_single_experiment_for_shap,
    get_model_matrix,
    save_shap_beeswarm,
    shap_group_importance_table,
    shap_importance_table,
)

from .pdp_ice_analysis import (
    compute_ice,
    compute_pdp,
    plot_ice_static,
    plot_pdp_static,
    top_features_by_shap,
)

from .plotting import (
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

from .plotting_interactive import (
    plot_data_availability_heatmap,
    plot_pdp_ice_interactive,
    plot_scatter_interactive,
    plot_shap_beeswarm_interactive,
    plot_timeseries_interactive,
    shap_to_long,
)


__all__ = [
    # xgboost_model
    "SplitMasks", "XGBoostModel",
    # feature_engineering
    "FeatureEngineering",
    # feature_constructor
    "FeatureConstructor", "OpsRouting",
    # feature_sets
    "get_experiments_abc",
    # cantareira_equivalent
    "CANTAREIRA_AREA_KM2", "CANTAREIRA_AREA_KM2_PDAA",
    "BasinAggregationSpec", "MultiBasinAggregationSpec",
    "aggregate_many_basins_wide", "aggregate_two_basins_wide",
    "area_share",
    "build_equivalent_cantareira_inputs", "build_total_cantareira_inputs",
    "default_cach_atib_spec_mm", "default_cantareira_total_spec",
    # system_features
    "add_cantareira_system_storage_features",
    # experiments
    "ExperimentResult", "run_experiments", "run_walk_forward_expanding",
    "seasonal_climatology_baseline",
    # shap_analysis
    "FittedExperiment", "assign_feature_group", "compute_tree_shap",
    "fit_single_experiment_for_shap", "get_model_matrix",
    "save_shap_beeswarm", "shap_group_importance_table",
    "shap_importance_table",
    # pdp_ice_analysis
    "compute_ice", "compute_pdp", "plot_ice_static", "plot_pdp_static",
    "top_features_by_shap",
    # plotting
    "build_system_totals",
    "plot_benchmark_scatter_1x2", "plot_benchmark_scatter_2x1",
    "plot_benchmark_timeseries_1x2", "plot_benchmark_timeseries_2x1",
    "plot_error_boxplot", "plot_residuals", "plot_scatter",
    "plot_shap_group_bar", "plot_shap_importance_bar",
    "plot_timeseries", "plot_timeseries_compare_experiments",
    "save_figure", "set_paper_plot_style", "set_plot_style",
    # plotting_interactive
    "plot_data_availability_heatmap", "plot_pdp_ice_interactive",
    "plot_scatter_interactive", "plot_shap_beeswarm_interactive",
    "plot_timeseries_interactive", "shap_to_long",
]
