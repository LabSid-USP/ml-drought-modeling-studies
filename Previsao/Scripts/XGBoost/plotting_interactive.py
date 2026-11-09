"""Visualização interativa via Plotly + ipywidgets.

Este módulo oferece *plots de exploração* (interativos) que complementam os
plots estáticos do estilo simpósio em :mod:`Previsao.Scripts.plotting`.

Cada função recebe estruturas em formato longo (DataFrame com colunas
``basin``, ``experiment``, ``feature``, etc.) e devolve um widget composto
por dropdowns que selecionam o que será exibido.

Uso típico no notebook
----------------------
.. code-block:: python

    from Previsao.Scripts.plotting_interactive import (
        plot_timeseries_interactive,
        plot_shap_beeswarm_interactive,
        plot_pdp_ice_interactive,
        plot_scatter_interactive,
    )

    plot_timeseries_interactive(preds_long)

Notas
-----
* Para uso fora de Jupyter, todas as funções aceitam ``return_fig=True`` e
  devolvem objetos ``plotly.graph_objects.Figure`` para uso headless.
* As figuras seguem uma paleta neutra (azul/cinza) coerente com o estilo
  publicado no simpósio.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Padrão de render usado nesta unidade:
# - NÃO usar Output() + display(fig) — esse caminho passa pelo display
#   protocol e emite a figura tanto como MIME
#   ``application/vnd.plotly.v1+json`` (lido pelo renderer do VSCode)
#   quanto como script HTML embutido (lido pelo Jupyter clássico), o que
#   em alguns frontends resulta em duplicação visual da figura.
# - Embalar a figura em um ipywidget próprio via ``_figure_to_widget``,
#   que tenta ``go.FigureWidget`` (requer anywidget) e, se indisponível,
#   recorre a um ``ipywidgets.HTML`` com a figura serializada via
#   ``plotly.io.to_html``. Ambos os caminhos renderizam apenas uma vez.


def _figure_to_widget(fig):
    """Embala uma ``plotly.graph_objects.Figure`` em um ``go.FigureWidget``.

    ``FigureWidget`` é o caminho ideal para Plotly + ipywidgets: renderiza
    a figura exatamente uma vez como ipywidget nativo, evitando a dupla
    emissão de MIME que o padrão ``Output + display(fig)`` produz em
    alguns frontends (VSCode notebook, em particular).

    Em versões modernas do Plotly (>= 5.20), ``FigureWidget`` requer o
    pacote ``anywidget``. Se não estiver disponível, levanta
    ``ImportError`` com instrução clara para instalar.

    Notas
    -----
    O fallback ``ipywidgets.HTML`` com ``pio.to_html(...)`` **não**
    funciona como alternativa em VSCode: o renderer de notebooks
    sanitiza as ``<script>`` tags que o Plotly precisa para desenhar,
    resultando em um placeholder em branco. Por isso optamos por
    falhar explicitamente em vez de produzir uma figura silenciosa
    vazia.
    """
    import plotly.graph_objects as go
    try:
        return go.FigureWidget(fig)
    except ImportError as e:
        raise ImportError(
            "go.FigureWidget requer o pacote 'anywidget'. "
            "Instale com: pip install anywidget\n"
            "Em conda: conda install -c conda-forge anywidget\n"
            f"(erro original: {e})"
        ) from e


COLORS = {
    "observed": "#1f3a5f",
    "predicted": "#d97742",
    "persistence": "#7f7f7f",
    "shade": "rgba(127, 127, 127, 0.10)",
    "ice": "rgba(102, 153, 204, 0.20)",
    "pdp": "#1f3a5f",
}


# --------------------------------------------------------------------------- #
# Timeseries interativo
# --------------------------------------------------------------------------- #
def plot_timeseries_interactive(
    preds_df: pd.DataFrame,
    *,
    basin_col: str = "basin",
    experiment_col: str = "experiment",
    date_col: str = "target_date",
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    baseline_col: str = "y_base",
    split_col: str | None = "split",
    title_prefix: str = "Previsão de volume",
    pred_label: str = "XGBoost",
    return_fig: bool = False,
):
    """Timeseries y_true / y_pred / persistência com dropdowns
    bacia × experimento.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame em formato longo com colunas indicadas pelos ``*_col``.
    basin_col, experiment_col, date_col, truth_col, pred_col : str
        Nomes das colunas no DataFrame.
    baseline_col : str
        Coluna com baseline de persistência (opcional; se ausente,
        ignorada).
    split_col : str | None
        Se presente, sombreia o split de teste em fundo cinza claro.
    return_fig : bool
        Se True, retorna apenas a figura inicial (sem widget). Útil para
        export estático ou uso headless.
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    basins = sorted(preds_df[basin_col].unique())
    experiments = sorted(preds_df[experiment_col].unique())

    def _make_fig(basin: str, experiment: str) -> "go.Figure":
        sub = preds_df[
            (preds_df[basin_col] == basin)
            & (preds_df[experiment_col] == experiment)
        ].sort_values(date_col)

        # Conversão explícita de datetime64[ns] (pandas) para list de
        # ``datetime`` (Python). FigureWidget + anywidget serializa o
        # datetime64 como int64-nanossegundos, mas o eixo "date" do
        # Plotly espera milissegundos-desde-epoch — diferença de 10⁶
        # que joga os pontos para fora da janela visível (ano ~52000).
        # Datetime puro evita essa armadilha. ``np.array(...)`` antes do
        # ``.dt.to_pydatetime()`` é a forma recomendada pelo pandas para
        # silenciar o FutureWarning sobre mudança de retorno.
        x_dates = list(np.array(sub[date_col].dt.to_pydatetime()))

        fig = go.Figure()
        if split_col and split_col in sub.columns:
            test_dates = sub.loc[sub[split_col] == "test", date_col]
            if len(test_dates) > 0:
                fig.add_vrect(
                    x0=test_dates.min().to_pydatetime(),
                    x1=test_dates.max().to_pydatetime(),
                    fillcolor=COLORS["shade"], opacity=1.0,
                    layer="below", line_width=0,
                    annotation_text="teste",
                    annotation_position="top left",
                )
        fig.add_trace(go.Scatter(
            x=x_dates, y=sub[truth_col],
            name="Observado", mode="lines",
            line=dict(color=COLORS["observed"], width=2.0),
        ))
        fig.add_trace(go.Scatter(
            x=x_dates, y=sub[pred_col],
            name=pred_label, mode="lines",
            line=dict(color=COLORS["predicted"], width=1.8, dash="solid"),
        ))
        if baseline_col in sub.columns and sub[baseline_col].notna().any():
            fig.add_trace(go.Scatter(
                x=x_dates, y=sub[baseline_col],
                name="Persistência", mode="lines",
                line=dict(color=COLORS["persistence"], width=1.2, dash="dot"),
            ))
        fig.update_layout(
            title=f"{title_prefix} — {basin} ({experiment})",
            xaxis_title="Data alvo (target_date)",
            xaxis=dict(type="date"),
            yaxis_title="Volume previsto",
            template="plotly_white",
            legend=dict(orientation="h", y=1.05, x=0.0),
            height=420,
        )
        return fig

    if return_fig:
        return _make_fig(basins[0], experiments[0])

    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    dd_exp = Dropdown(
        options=experiments, value=experiments[0], description="Exp.:")
    fig_box = VBox()

    def _refresh(_change=None):
        new_fig = _make_fig(dd_basin.value, dd_exp.value)
        fig_box.children = (_figure_to_widget(new_fig),)

    dd_basin.observe(_refresh, names="value")
    dd_exp.observe(_refresh, names="value")
    _refresh()
    return VBox([HBox([dd_basin, dd_exp]), fig_box])


# --------------------------------------------------------------------------- #
# Scatter y_true vs y_pred interativo
# --------------------------------------------------------------------------- #
def plot_scatter_interactive(
    preds_df: pd.DataFrame,
    *,
    basin_col: str = "basin",
    experiment_col: str = "experiment",
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    split_col: str | None = "split",
    title_prefix: str = "Dispersão observado × previsto",
):
    """Scatter y_true × y_pred com linha 1:1 e métricas no título.

    Útil para inspecionar viés, dispersão e calibração visualmente.
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    basins = sorted(preds_df[basin_col].unique())
    experiments = sorted(preds_df[experiment_col].unique())

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        m = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if m.sum() < 2:
            return ""
        rmse = float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2)))
        mae = float(np.mean(np.abs(y_pred[m] - y_true[m])))
        corr = float(np.corrcoef(y_true[m], y_pred[m])[0, 1])
        return f"RMSE={rmse:.2f}, MAE={mae:.2f}, corr={corr:.2f}"

    def _make_fig(basin: str, experiment: str) -> "go.Figure":
        sub = preds_df[
            (preds_df[basin_col] == basin)
            & (preds_df[experiment_col] == experiment)
        ]
        fig = go.Figure()
        for split, color in [
            ("train", "#cccccc"),
            ("val", "#7da7d9"),
            ("test", COLORS["observed"]),
        ]:
            if split_col is None or split_col not in sub.columns:
                sub_s = sub
                color = COLORS["observed"]
                label = "todos"
            else:
                sub_s = sub[sub[split_col] == split]
                label = split
            if len(sub_s) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub_s[truth_col], y=sub_s[pred_col],
                mode="markers",
                marker=dict(color=color, size=6, line=dict(width=0)),
                name=label,
                opacity=0.7,
            ))
            if split_col is None:
                break

        # Linha 1:1
        if len(sub) > 0:
            lo = float(min(sub[truth_col].min(), sub[pred_col].min()))
            hi = float(max(sub[truth_col].max(), sub[pred_col].max()))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines", line=dict(color="black", dash="dot", width=1),
                name="1:1", showlegend=False,
            ))

        test_sub = (
            sub[sub[split_col] == "test"]
            if split_col and split_col in sub.columns
            else sub
        )
        metrics_str = _metrics(
            test_sub[truth_col].to_numpy(),
            test_sub[pred_col].to_numpy(),
        )
        fig.update_layout(
            title=(
                f"{title_prefix} — {basin} ({experiment}). "
                f"Teste: {metrics_str}"
            ),
            xaxis_title="Observado",
            yaxis_title="Previsto",
            template="plotly_white",
            height=420,
        )
        return fig

    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    dd_exp = Dropdown(
        options=experiments, value=experiments[0], description="Exp.:")
    fig_box = VBox()

    def _refresh(_change=None):
        new_fig = _make_fig(dd_basin.value, dd_exp.value)
        fig_box.children = (_figure_to_widget(new_fig),)

    dd_basin.observe(_refresh, names="value")
    dd_exp.observe(_refresh, names="value")
    _refresh()
    return VBox([HBox([dd_basin, dd_exp]), fig_box])


# --------------------------------------------------------------------------- #
# SHAP beeswarm interativo
# --------------------------------------------------------------------------- #
def plot_shap_beeswarm_interactive(
    shap_long: pd.DataFrame,
    *,
    basin_col: str = "basin",
    experiment_col: str = "experiment",
    feature_col: str = "feature",
    shap_col: str = "shap_value",
    feature_value_col: str = "feature_value",
    max_features: int = 15,
    title_prefix: str = "SHAP beeswarm",
):
    """Beeswarm SHAP em plotly com dropdowns bacia × experimento.

    Espera ``shap_long`` em formato longo: uma linha por (bacia, experimento,
    feature, amostra), com colunas ``shap_value`` e ``feature_value``.

    O dropdown de experimentos é filtrado em função da bacia selecionada,
    de modo que só apareçam combinações ``(basin, experiment)`` realmente
    presentes em ``shap_long``. Se uma combinação não tiver dados, exibe
    um placeholder textual em vez de um plot vazio sem informação.
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    basins = sorted(shap_long[basin_col].unique())
    # Pares (basin, experiment) realmente presentes em shap_long.
    pairs_by_basin: dict[str, list[str]] = {}
    for b in basins:
        exps = (
            shap_long.loc[shap_long[basin_col] == b, experiment_col]
            .unique()
            .tolist()
        )
        pairs_by_basin[b] = sorted(exps)

    def _make_fig(basin: str, experiment: str) -> "go.Figure":
        sub = shap_long[
            (shap_long[basin_col] == basin)
            & (shap_long[experiment_col] == experiment)
        ]
        if len(sub) == 0:
            empty = go.Figure()
            empty.update_layout(
                title=(f"{title_prefix} — sem dados para "
                       f"({basin}, {experiment})"),
                template="plotly_white", height=300,
            )
            return empty

        order = (
            sub.groupby(feature_col)[shap_col]
            .apply(lambda s: s.abs().mean())
            .sort_values(ascending=True)
        )
        keep = order.tail(max_features).index.tolist()
        sub = sub[sub[feature_col].isin(keep)]

        fig = go.Figure()
        rng = np.random.RandomState(0)
        for i, feat in enumerate(keep):
            sf = sub[sub[feature_col] == feat]
            vals = sf[feature_value_col]
            # Normalização robusta para colormap (percentis 5–95).
            if vals.notna().sum() > 1:
                v_min, v_max = np.nanpercentile(vals, [5, 95])
                if v_max == v_min:
                    color_vals = np.zeros(len(vals))
                else:
                    color_vals = (vals.clip(v_min, v_max) - v_min) / (
                        v_max - v_min)
            else:
                color_vals = np.zeros(len(vals))
            jitter = (rng.rand(len(sf)) - 0.5) * 0.30
            # Eixo Y numérico para permitir jitter; rótulos aplicados via
            # tickvals/ticktext após o loop.
            y_pos = np.full(len(sf), i, dtype=float) + jitter
            fig.add_trace(go.Scatter(
                x=sf[shap_col],
                y=y_pos,
                mode="markers",
                marker=dict(
                    color=color_vals, colorscale="RdBu_r",
                    showscale=False, size=5, opacity=0.85,
                ),
                name=feat, showlegend=False,
                hovertemplate=(
                    f"<b>{feat}</b><br>SHAP=%{{x:.3f}}<br>"
                    "valor=%{customdata:.3f}<extra></extra>"
                ),
                customdata=vals,
            ))
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"{title_prefix} — {basin} ({experiment})",
            xaxis_title="Valor SHAP",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(420, 26 * len(keep)),
        )
        fig.update_yaxes(
            tickvals=list(range(len(keep))),
            ticktext=keep,
        )
        return fig

    initial_exps = pairs_by_basin[basins[0]]
    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    dd_exp = Dropdown(
        options=initial_exps,
        value=initial_exps[0] if initial_exps else None,
        description="Exp.:",
    )
    fig_box = VBox()

    def _refresh(_change=None):
        if dd_exp.value is None:
            fig_box.children = ()
            return
        new_fig = _make_fig(dd_basin.value, dd_exp.value)
        fig_box.children = (_figure_to_widget(new_fig),)

    def _update_experiments(_change=None):
        exps = pairs_by_basin[dd_basin.value]
        dd_exp.options = exps
        if dd_exp.value not in exps and exps:
            dd_exp.value = exps[0]
        _refresh()

    dd_basin.observe(_update_experiments, names="value")
    dd_exp.observe(_refresh, names="value")
    _refresh()
    return VBox([HBox([dd_basin, dd_exp]), fig_box])


# --------------------------------------------------------------------------- #
# PDP + ICE interativo
# --------------------------------------------------------------------------- #
def plot_pdp_ice_interactive(
    pdp_df: pd.DataFrame,
    ice_df: pd.DataFrame | None = None,
    *,
    basin_col: str = "basin",
    feature_col: str = "feature",
    pdp_value_col: str = "pdp",
    ice_value_col: str = "ice",
    grid_col: str = "feature_value",
    title_prefix: str = "PDP & ICE",
):
    """PDP (linha) + ICE (background) com dropdowns bacia × feature."""
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    basins = sorted(pdp_df[basin_col].unique())

    def _features_for(basin: str) -> list[str]:
        return sorted(pdp_df[pdp_df[basin_col] == basin][feature_col].unique())

    def _make_fig(basin: str, feature: str) -> "go.Figure":
        fig = go.Figure()
        if ice_df is not None:
            sub_ice = ice_df[
                (ice_df[basin_col] == basin)
                & (ice_df[feature_col] == feature)
            ]
            for sid, grp in sub_ice.groupby("sample_id"):
                grp_sorted = grp.sort_values(grid_col)
                fig.add_trace(go.Scatter(
                    x=grp_sorted[grid_col], y=grp_sorted[ice_value_col],
                    mode="lines",
                    line=dict(color=COLORS["ice"], width=0.6),
                    showlegend=False, hoverinfo="skip",
                ))
        sub_pdp = pdp_df[
            (pdp_df[basin_col] == basin)
            & (pdp_df[feature_col] == feature)
        ].sort_values(grid_col)
        fig.add_trace(go.Scatter(
            x=sub_pdp[grid_col], y=sub_pdp[pdp_value_col],
            mode="lines",
            line=dict(color=COLORS["pdp"], width=2.2),
            name="PDP",
        ))
        fig.update_layout(
            title=f"{title_prefix} — {basin}, feature: {feature}",
            xaxis_title=feature, yaxis_title="Resposta do modelo",
            template="plotly_white", height=420,
        )
        return fig

    initial_feats = _features_for(basins[0])
    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    dd_feat = Dropdown(
        options=initial_feats, value=initial_feats[0], description="Feature:")
    fig_box = VBox()

    def _refresh(_change=None):
        new_fig = _make_fig(dd_basin.value, dd_feat.value)
        fig_box.children = (_figure_to_widget(new_fig),)

    def _update_features(_change=None):
        feats = _features_for(dd_basin.value)
        dd_feat.options = feats
        if dd_feat.value not in feats and len(feats) > 0:
            dd_feat.value = feats[0]
        _refresh()

    dd_basin.observe(_update_features, names="value")
    dd_feat.observe(_refresh, names="value")
    _refresh()
    return VBox([HBox([dd_basin, dd_feat]), fig_box])


# --------------------------------------------------------------------------- #
# Mapa de disponibilidade temporal (heatmap interativo)
# --------------------------------------------------------------------------- #
def plot_data_availability_heatmap(
    datasets_by_basin: dict[str, pd.DataFrame],
    *,
    variables: Iterable[str] | None = None,
    title: str = "Disponibilidade temporal dos dados",
):
    """Heatmap interativo mostrando lacunas (NaN) por variável × tempo.

    Mostra um seletor de bacia; para cada bacia, exibe um heatmap onde o eixo
    Y são as variáveis disponíveis e o eixo X são os meses. Células claras
    indicam ausência (NaN); células escuras indicam presença.

    Tolerância a variáveis ausentes
    -------------------------------
    Se uma variável requisitada não existe no DataFrame de uma bacia (por
    exemplo, `qelev` em CT_JAGJAC, pois o roteamento operacional não o
    inclui), a função **não levanta KeyError**. Em vez disso, exibe a
    variável como uma linha totalmente "ausente" e acrescenta o sufixo
    ``" (n/a)"`` ao label do eixo Y, sinalizando visualmente que a
    variável não se aplica àquela bacia.
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, VBox

    basins = sorted(datasets_by_basin.keys())

    def _make_fig(basin: str) -> "go.Figure":
        df = datasets_by_basin[basin].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        vars_use = (
            list(variables)
            if variables is not None
            else [c for c in df.columns if c != "date"]
        )

        # Para cada variável solicitada, gera uma linha do heatmap.
        # Se a coluna não existir nesta bacia, a linha fica toda "ausente"
        # e o label recebe o sufixo " (n/a)".
        rows = []
        labels = []
        for v in vars_use:
            if v in df.columns:
                rows.append(df[v].notna().astype(int).values)
                labels.append(v)
            else:
                rows.append(np.zeros(len(df), dtype=int))
                labels.append(f"{v} (n/a)")
        mat = (
            np.vstack(rows) if rows
            else np.zeros((0, len(df)), dtype=int)
        )

        fig = go.Figure(data=go.Heatmap(
            z=mat,
            x=df["date"].dt.strftime("%Y-%m"),
            y=labels,
            colorscale=[[0, "#f5f0e1"], [1, "#1f3a5f"]],
            showscale=False,
        ))
        fig.update_layout(
            title=f"{title} — {basin}",
            xaxis_title="Mês", yaxis_title="Variável",
            template="plotly_white",
            height=max(280, 20 * len(labels)),
        )
        return fig

    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    fig_box = VBox()

    def _refresh(_change=None):
        new_fig = _make_fig(dd_basin.value)
        fig_box.children = (_figure_to_widget(new_fig),)

    dd_basin.observe(_refresh, names="value")
    _refresh()
    return VBox([dd_basin, fig_box])


# --------------------------------------------------------------------------- #
# Helpers para converter SHAP / PDP / ICE para formato longo
# --------------------------------------------------------------------------- #
def shap_to_long(
    shap_values_dict: dict[tuple[str, str], object],
    X_eval_dict: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Converte dict[(basin, exp)] → SHAP values em DataFrame longo.

    Espera valores SHAP no formato retornado por ``compute_tree_shap`` (objeto
    ``shap.Explanation`` ou similar com atributo ``.values``).
    """
    rows: list[pd.DataFrame] = []
    for (basin, experiment), shap_obj in shap_values_dict.items():
        X = X_eval_dict[(basin, experiment)]
        vals = np.asarray(shap_obj.values)  # type: ignore
        if vals.ndim != 2:
            raise ValueError(
                "Esperava SHAP 2D (uma saída). Use modelo single-output.")
        df = pd.DataFrame(vals, columns=X.columns, index=X.index)
        long = df.reset_index(drop=False).melt(
            id_vars="index", var_name="feature", value_name="shap_value")
        long = long.rename(columns={"index": "sample_id"})
        # Anexa o valor da feature (para colormap)
        X_long = X.reset_index(drop=False).melt(
            id_vars="index", var_name="feature", value_name="feature_value")
        X_long = X_long.rename(columns={"index": "sample_id"})
        long = long.merge(
            X_long, on=["sample_id", "feature"], how="left")
        long["basin"] = basin
        long["experiment"] = experiment
        rows.append(long)
    return pd.concat(rows, ignore_index=True)
