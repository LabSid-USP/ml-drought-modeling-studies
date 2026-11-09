"""
Visualizações específicas do braço LSTM.

Filosofia: **reusar** ao máximo os plots do XGBoost (que já são consagrados
e estilizados para o simpósio) e adicionar apenas o que é específico de
redes neurais sequenciais:

* **Curvas de aprendizado** (train_loss / val_loss por época) — diagnóstico
  visual de overfitting, convergência e gatilho de early stopping.
* **Banda multi-seed** (predições com ±σ entre seeds) — comunica
  visualmente a incerteza de inicialização da rede.
* **Resíduos no tempo** + histograma + Q-Q plot — diagnóstico de bias
  sistemático, heterocedasticidade e normalidade.

Para timeseries e scatter "padrão", reaproveite diretamente
:func:`Previsao.Scripts.XGBoost.plotting_interactive.plot_timeseries_interactive`
e :func:`...plot_scatter_interactive` — o ``preds_df`` produzido pelo braço
LSTM tem schema idêntico ao do XGBoost (coluna ``seed`` extra é ignorada
pelos plots padrão).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helper compartilhado com o módulo plotting_interactive do XGBoost
# --------------------------------------------------------------------------- #
def _figure_to_widget(fig):
    """Embala uma figura Plotly em widget (FigureWidget se possível).

    Replica o padrão estabelecido em
    :func:`Previsao.Scripts.XGBoost.plotting_interactive._figure_to_widget`
    para evitar duplo-render por MIME duplicado.
    """
    import plotly.graph_objects as go
    try:
        return go.FigureWidget(fig)
    except ImportError as e:
        raise ImportError(
            "go.FigureWidget requer 'anywidget'. "
            "Instale com: pip install anywidget"
        ) from e


# --------------------------------------------------------------------------- #
# Curvas de aprendizado interativas
# --------------------------------------------------------------------------- #
def plot_training_curves_interactive(
    history_df: pd.DataFrame,
    *,
    basin: str = "",
    experiment: str = "",
    seed: int | None = None,
):
    """Plota curvas de loss por época (treino e validação).

    Marca a melhor época (menor ``val_loss``) com linha vertical pontilhada.
    Útil para diagnosticar:

    * **Underfit**: ``train_loss`` alto e estável → modelo pequeno demais.
    * **Overfit**: ``train_loss`` continua caindo mas ``val_loss`` sobe.
    * **Convergência**: ambos caem e estabilizam.
    * **Instabilidade**: ``val_loss`` oscilando muito → reduzir ``lr`` ou
      aumentar ``batch_size``.

    :param history_df: DataFrame com colunas ``epoch``, ``train_loss``,
        ``val_loss`` (output de :func:`train_lstm`).
    :param basin: nome da bacia para o título.
    :param experiment: nome do experimento para o título.
    :param seed: seed para o título (quando rodando multi-seed).

    :returns: ``ipywidgets`` widget renderizável no notebook.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["train_loss"],
        name="train", mode="lines",
        line=dict(color="#1f3a5f", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["val_loss"],
        name="val", mode="lines",
        line=dict(color="#d97742", width=2),
    ))
    if len(history_df) > 0:
        best_epoch = int(history_df["val_loss"].idxmin())
        best_val = float(history_df.loc[best_epoch, "val_loss"])
        fig.add_vline(
            x=best_epoch, line_dash="dot", line_color="gray",
            annotation_text=f"melhor: época {best_epoch} "
                            f"(val={best_val:.4f})",
            annotation_position="top right",
        )
    title = f"Curvas de aprendizado — {basin} ({experiment})"
    if seed is not None:
        title += f" — seed={seed}"
    fig.update_layout(
        title=title,
        xaxis_title="Época", yaxis_title="MSE loss (escala normalizada)",
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=1.05, x=0.0),
    )
    from ipywidgets import VBox
    fig_box = VBox()
    fig_box.children = (_figure_to_widget(fig),)
    return fig_box


# --------------------------------------------------------------------------- #
# Banda multi-seed: timeseries média ± σ
# --------------------------------------------------------------------------- #
def plot_multi_seed_band_interactive(
    preds_long_df: pd.DataFrame,
    *,
    basin_col: str = "basin",
    experiment_col: str = "experiment",
    date_col: str = "target_date",
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    seed_col: str = "seed",
    title_prefix: str = "Previsão LSTM (multi-seed)",
):
    """Banda de ±1σ entre seeds + linha da média, com dropdowns por bacia/exp.

    Visualiza a **incerteza de inicialização** — variabilidade introduzida
    pelos pesos iniciais aleatórios das redes. Em bases pequenas (~480
    amostras) essa variância é não-trivial: rodar a mesma arquitetura com
    seeds diferentes pode produzir RMSEs que diferem em 5-10 %.

    :param preds_long_df: DataFrame em formato longo, uma linha por
        ``(basin, experiment, seed, target_date)`` com colunas
        ``y_true`` e ``y_pred``.
    :returns: widget interativo (FigureWidget).
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    basins = sorted(preds_long_df[basin_col].unique())
    pairs_by_basin = {
        b: sorted(
            preds_long_df.loc[
                preds_long_df[basin_col] == b, experiment_col
            ].unique().tolist()
        )
        for b in basins
    }

    def _make_fig(basin: str, experiment: str):
        sub = preds_long_df[
            (preds_long_df[basin_col] == basin)
            & (preds_long_df[experiment_col] == experiment)
        ].copy()
        sub[date_col] = pd.to_datetime(sub[date_col])

        # Agrega por data: média e ±σ entre seeds
        agg = (
            sub.groupby(date_col)
            .agg(
                y_true=(truth_col, "first"),
                y_pred_mean=(pred_col, "mean"),
                y_pred_std=(pred_col, "std"),
            )
            .sort_index()
            .reset_index()
        )
        agg["y_pred_std"] = agg["y_pred_std"].fillna(0)
        upper = agg["y_pred_mean"] + agg["y_pred_std"]
        lower = agg["y_pred_mean"] - agg["y_pred_std"]
        # plotly fill: precisa do upper primeiro e do lower invertido
        x_dates = [ts.to_pydatetime() for ts in agg[date_col]]
        x_band = x_dates + x_dates[::-1]
        y_band = list(upper) + list(lower[::-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, fill="toself",
            fillcolor="rgba(217, 119, 66, 0.18)",
            line=dict(width=0), hoverinfo="skip",
            name="±σ entre seeds",
        ))
        fig.add_trace(go.Scatter(
            x=x_dates, y=agg["y_pred_mean"],
            name="média entre seeds", mode="lines",
            line=dict(color="#d97742", width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=x_dates, y=agg["y_true"],
            name="observado", mode="lines",
            line=dict(color="#1f3a5f", width=2),
        ))
        fig.update_layout(
            title=f"{title_prefix} — {basin} ({experiment})",
            xaxis_title="target_date", yaxis_title="Volume previsto",
            xaxis=dict(type="date"),
            template="plotly_white", height=420,
            legend=dict(orientation="h", y=1.05, x=0.0),
        )
        return fig

    initial_exps = pairs_by_basin[basins[0]]
    dd_basin = Dropdown(options=basins, value=basins[0], description="Bacia:")
    dd_exp = Dropdown(
        options=initial_exps, value=initial_exps[0],
        description="Exp.:",
    )
    fig_box = VBox()

    def _refresh(_change=None):
        if dd_exp.value is None:
            fig_box.children = ()
            return
        fig_box.children = (
            _figure_to_widget(_make_fig(dd_basin.value, dd_exp.value)),
        )

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
# Diagnóstico de resíduos (estático, paper-ready)
# --------------------------------------------------------------------------- #
def plot_lstm_residuals(
    preds_df: pd.DataFrame,
    *,
    basin: str | None = None,
    experiment: str | None = None,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    date_col: str = "target_date",
    split_filter: str | None = "test",
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """Painel 1×3 de diagnóstico de resíduos: tempo, histograma, Q-Q.

    Resíduos = ``y_pred - y_true``. Boas previsões devem ter:

    * **Painel 1 (tempo)**: resíduos espalhados aleatoriamente em torno
      de zero, sem tendência sazonal sistemática.
    * **Painel 2 (histograma)**: distribuição aproximadamente simétrica
      em torno de zero (bias ≈ 0).
    * **Painel 3 (Q-Q)**: pontos próximos da diagonal teórica = resíduos
      aproximadamente normais. Caudas pesadas indicam erros extremos
      maiores que o esperado por uma normal.

    :param preds_df: DataFrame no schema padrão.
    :param basin, experiment: filtros opcionais.
    :param split_filter: split a usar (default ``"test"``).
    :param output_path: se fornecido, salva PNG.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    df = preds_df.copy()
    if basin is not None and "basin" in df.columns:
        df = df[df["basin"] == basin]
    if experiment is not None and "experiment" in df.columns:
        df = df[df["experiment"] == experiment]
    if split_filter is not None and "split" in df.columns:
        df = df[df["split"] == split_filter]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    residuals = df[pred_col].to_numpy() - df[truth_col].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # 1. Resíduos no tempo
    axes[0].scatter(df[date_col], residuals, color="#1f3a5f", s=18, alpha=0.7)
    axes[0].axhline(0, color="gray", lw=0.6, linestyle="--")
    axes[0].set_title("(a) Resíduos no tempo", loc="left", fontweight="bold")
    axes[0].set_xlabel("target_date")
    axes[0].set_ylabel("resíduo = ŷ - y")
    axes[0].grid(True, alpha=0.25)

    # 2. Histograma
    axes[1].hist(residuals, bins=24, color="#d97742",
                 alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="gray", lw=0.8, linestyle="--")
    bias = float(np.mean(residuals))
    axes[1].axvline(bias, color="#1f3a5f", lw=1.2,
                    label=f"bias = {bias:+.2f}")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("(b) Distribuição de resíduos",
                      loc="left", fontweight="bold")
    axes[1].set_xlabel("resíduo")
    axes[1].grid(True, alpha=0.25)

    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("")  # remove o "Probability Plot" automático do probplot
    axes[2].set_title("(c) Q-Q plot vs normal",
                      loc="left", fontweight="bold")
    axes[2].grid(True, alpha=0.25)

    if title:
        fig.suptitle(title, fontweight="bold", x=0.02, y=1.02, ha="left")
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Banda quantílica P10/P50/P90 (interativa, Plotly)
# --------------------------------------------------------------------------- #
def plot_quantile_band_interactive(
    preds_qr_df: pd.DataFrame,
    *,
    p10_col: str = "y_pred_p10",
    p50_col: str = "y_pred_p50",
    p90_col: str = "y_pred_p90",
    true_col: str = "y_true",
    base_col: str = "y_base",
    date_col: str = "target_date",
    basin_col: str = "basin",
    experiment_col: str = "experiment",
    split_filter: str | None = "test",
    split_col: str = "split",
    title_prefix: str = "Regressão Quantílica LSTM",
):
    """Banda P10-P90 preenchida + linha P50 + observado (Plotly interativo).

    Visualiza as **bandas de previsão** geradas por
    :func:`~lstm_experiments.run_lstm_quantile_experiment`. A banda
    P10-P90 representa um intervalo de 80 % de confiança nominal:
    espera-se que ~80 % das observações caiam dentro da região azul.

    Traces exibidos:

    * **Banda P10-P90** (azul translúcida): intervalo de incerteza.
    * **P50 / mediana** (linha azul sólida): melhor estimativa central.
    * **Observado** (cinza escuro): volume observado no período de teste.
    * **Persistência** (tracejado, se coluna ``y_base`` disponível).

    :param preds_qr_df: DataFrame com colunas ``y_pred_p10``,
        ``y_pred_p50``, ``y_pred_p90`` — output de
        :func:`~lstm_experiments.run_lstm_quantile_multi_seed`.
    :param p10_col, p50_col, p90_col: nomes das colunas dos quantis.
    :param true_col: nome da coluna com valores observados.
    :param base_col: nome da coluna de baseline (persistência).
    :param date_col: coluna de datas (``target_date``).
    :param split_filter: filtra um split específico (default ``"test"``).
    :param title_prefix: prefixo do título do gráfico.

    :returns: widget VBox interativo (Plotly + ipywidgets).
    """
    import plotly.graph_objects as go
    from ipywidgets import Dropdown, HBox, VBox

    df = preds_qr_df.copy()
    if split_filter is not None and split_col in df.columns:
        df = df[df[split_col] == split_filter]
    df[date_col] = pd.to_datetime(df[date_col])

    basins = sorted(df[basin_col].unique())
    pairs_by_basin = {
        b: sorted(
            df.loc[df[basin_col] == b, experiment_col].unique().tolist()
        )
        for b in basins
    }

    def _make_fig(basin: str, experiment: str):
        sub = (
            df[
                (df[basin_col] == basin)
                & (df[experiment_col] == experiment)
            ]
            .sort_values(date_col)
        )

        # Agrega por data: médias entre seeds (se houver múltiplas)
        agg = (
            sub.groupby(date_col)
            .agg(
                y_true=(true_col, "first"),
                y_base=(base_col, "first") if base_col in sub else None,
                p10=(p10_col, "mean"),
                p50=(p50_col, "mean"),
                p90=(p90_col, "mean"),
            )
            .sort_index()
            .reset_index()
        )

        x_dates = list(agg[date_col])
        # Banda fill: P90 forward + P10 reversed → polígono fechado
        x_band = x_dates + x_dates[::-1]
        y_band = list(agg["p90"]) + list(agg["p10"][::-1])

        fig = go.Figure()

        # 1. Banda P10-P90 (área preenchida)
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            name="Banda P10-P90 (80 % nominal)",
        ))

        # 2. Linha P50 (mediana)
        fig.add_trace(go.Scatter(
            x=x_dates, y=agg["p50"],
            name="P50 (mediana)", mode="lines",
            line=dict(color="#1f77b4", width=2),
        ))

        # 3. Observado
        fig.add_trace(go.Scatter(
            x=x_dates, y=agg["y_true"],
            name="observado", mode="lines",
            line=dict(color="#1f3a5f", width=2),
        ))

        # 4. Persistência (opcional)
        if "y_base" in agg.columns and agg["y_base"].notna().any():
            fig.add_trace(go.Scatter(
                x=x_dates, y=agg["y_base"],
                name="persistência", mode="lines",
                line=dict(color="#aaaaaa", width=1.2, dash="dot"),
            ))

        fig.update_layout(
            title=f"{title_prefix} — {basin} ({experiment})",
            xaxis_title="target_date",
            yaxis_title="Volume (hm³)",
            xaxis=dict(type="date"),
            template="plotly_white",
            height=440,
            legend=dict(orientation="h", y=1.06, x=0.0),
        )
        return fig

    initial_exps = pairs_by_basin[basins[0]]
    dd_basin = Dropdown(
        options=basins, value=basins[0], description="Bacia:",
    )
    dd_exp = Dropdown(
        options=initial_exps, value=initial_exps[0], description="Exp.:",
    )
    fig_box = VBox()

    def _refresh(_change=None):
        if dd_exp.value is None:
            fig_box.children = ()
            return
        fig_box.children = (
            _figure_to_widget(_make_fig(dd_basin.value, dd_exp.value)),
        )

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
