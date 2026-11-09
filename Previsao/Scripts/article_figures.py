"""Figuras do artigo de previsão (volume do CT_TOTAL) em alta definição.

Reproduz as Figs. 2–6 aprovadas pelos autores (versão anterior gerada fora do
repositório) e a Fig. S1 (material suplementar) a partir dos CSVs
``article_*`` versionados em ``Previsao/Outputs/`` — nenhum modelo é
retreinado.

Cores, ordem dos painéis, marcadores e legendas seguem as figuras aprovadas
(paleta Okabe–Ito). Mudanças em relação à versão aprovada, todas exigidas pelas
diretrizes de arte da revista-alvo:

* fonte Arial (8–9 pt no tamanho final), em vez de DejaVu Sans;
* sem títulos internos nos painéis (o texto vai para a legenda do manuscrito);
* desenho já no tamanho final (174 mm de largura, altura ≤ 234 mm);
* exportação em EPS (vetor, fontes embutidas), PDF, TIFF e PNG a 600 dpi.

O PostScript não suporta transparência; por isso as sobreposições
semitransparentes do original (faixa P10–P90 sobre meses de seca, áreas da
Fig. 6b) são compostas aqui com cores opacas equivalentes (:func:`blend`),
calculadas a partir das opacidades medidas nos PNGs aprovados.

Uso::

    python -m Previsao.Scripts.article_figures --out Previsao/Outputs/forecast_paper
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Iterator

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from PIL import Image

# --------------------------------------------------------------------------- #
# Dimensões (revista: 84 ou 174 mm de largura, altura máxima 234 mm)
# --------------------------------------------------------------------------- #

MM = 1.0 / 25.4
WIDTH_2COL = 174.0 * MM
WIDTH_1COL = 84.0 * MM
MAX_HEIGHT = 234.0 * MM
DPI_RASTER = 600

# --------------------------------------------------------------------------- #
# Paleta — amostrada dos PNGs aprovados (Okabe–Ito + cinzas neutros)
# --------------------------------------------------------------------------- #

BLUE = "#0072B2"         # LSTM C / H=3 / período de teste completo (SHAP)
ORANGE = "#D55E00"       # XGBoost D / modelo escolhido / meses de seca (SHAP)
GREEN = "#009E73"        # LSTM D / H=12
OBS = "#111111"          # série observada
SLATE = "#677381"        # barras de importância por grupo (Fig. 2a)
BASELINE = "#9AA4AE"     # referências ingênuas (barras) e intervalos indecisos
PERSIST = "#6E7781"      # linha de persistência (tracejada)
CLIM = "#9AA4AE"         # linha de climatologia (pontilhada)
CONNECTOR = "#9AA4AE"    # conectores dos *dumbbells*
CONNECTOR_SOFT = "#C3CCD5"
GRID = "#E1E6EB"
INK = "#1F2933"
MUTED = "#5B6670"
DROUGHT_SHADE = "#F6E4D7"  # faixa dos meses de seca observados (opaca)

# Opacidades medidas nos PNGs aprovados.
ALPHA_PI = 0.165          # faixa P10–P90
ALPHA_AREA = 0.14         # áreas sob as curvas da Fig. 6b
ALPHA_MARK = 0.90         # marcadores e intervalos da Fig. 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "Previsao" / "Outputs"


def blend(fg: str, alpha: float, bg: str = "#FFFFFF") -> str:
    """Compõe ``fg`` com opacidade ``alpha`` sobre ``bg`` e devolve a cor opaca.

    Parameters
    ----------
    fg, bg : str
        Cores (qualquer formato aceito pelo matplotlib).
    alpha : float
        Opacidade de ``fg`` em [0, 1].

    Returns
    -------
    str
        Cor hexadecimal opaca equivalente (compatível com EPS).
    """
    f = np.array(to_rgb(fg))
    b = np.array(to_rgb(bg))
    return to_hex(alpha * f + (1.0 - alpha) * b)


# --------------------------------------------------------------------------- #
# Estilo e exportação
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def paper_style() -> Iterator[None]:
    """Contexto de estilo das figuras do artigo (não altera o estilo global).

    Arial 8–9 pt, sem bordas superior/direita, legendas sem moldura, fontes
    TrueType embutidas (Type 42) em PDF/EPS e linhas ≥ 0,5 pt (a revista pede
    ≥ 0,3 pt).
    """
    rc = {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI_RASTER,
        "figure.dpi": 100,
        "axes.unicode_minus": True,
    }
    with plt.rc_context(rc):
        yield


def panel_label(fig: plt.Figure, ax: plt.Axes, letter: str,
                dx_mm: float = 0.0, dy_mm: float = 0.0) -> None:
    """Escreve a letra do painel (minúscula, negrito) no canto do eixo.

    A letra fica à esquerda da área ocupada pelos rótulos do eixo y
    (``tightbbox``), alinhada ao topo do eixo, como nas figuras aprovadas.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    pos = ax.get_position()
    w_in, h_in = fig.get_size_inches()
    x = max(tb.x0 + dx_mm * MM / w_in, 0.004)
    y = pos.y1 + (2.0 + dy_mm) * MM / h_in
    fig.text(x, y, letter, fontsize=10, fontweight="bold", color=INK,
             ha="left", va="bottom")


def export(fig: plt.Figure, out_dir: Path, name: str) -> dict[str, Path]:
    """Salva a figura em EPS, PDF, TIFF (600 dpi, LZW) e PNG (600 dpi).

    O tamanho físico é o do ``figsize`` (sem ``bbox_inches='tight'``), para que
    a largura final seja exatamente a da coluna da revista.

    Returns
    -------
    dict[str, Path]
        Caminho de cada formato gravado.
    """
    w, h = fig.get_size_inches()
    assert h <= MAX_HEIGHT + 1e-6, f"{name}: altura {h / MM:.1f} mm > 234 mm"
    paths = {
        "eps": out_dir / "eps" / f"{name}.eps",
        "pdf": out_dir / "pdf" / f"{name}.pdf",
        "tiff": out_dir / "tiff_600dpi" / f"{name}.tiff",
        "png": out_dir / "png_600dpi" / f"{name}.png",
    }
    for fmt, p in paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        kw: dict = {"facecolor": "white"}
        if fmt in ("tiff", "png"):
            kw["dpi"] = DPI_RASTER
        if fmt == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, format=fmt, **kw)
    # TIFF/PNG em RGB de 8 bits (sem canal alfa), como pede a revista.
    for fmt in ("tiff", "png"):
        with Image.open(paths[fmt]) as im:
            rgb = im.convert("RGB") if im.mode != "RGB" else None
        if rgb is not None:
            save_kw: dict = {"dpi": (DPI_RASTER, DPI_RASTER)}
            if fmt == "tiff":
                save_kw["compression"] = "tiff_lzw"
            rgb.save(paths[fmt], **save_kw)
    plt.close(fig)
    return paths


def _grid_x(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _grid_both(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Leitura dos dados (CSVs versionados)
# --------------------------------------------------------------------------- #

def _csv(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / rel, **kw)


# --------------------------------------------------------------------------- #
# Fig. 2 — SHAP do XGBoost D (H = 3)
# --------------------------------------------------------------------------- #

GROUP_LABELS = {
    "calendar": "Calendar",
    "precipitation": "Precipitation",
    "operation": "Operation",
    "local_storage": "Local storage",
    "natural_flow": "Natural flow",
    "evapotranspiration": "Evapotranspiration",
}


def fig2_shap() -> plt.Figure:
    """(a) mean |SHAP| por grupo; (b) SHAP médio com sinal, teste × seca."""
    grp = _csv("XGBoost/shap_importance/article_ct_total_h3_shap_group.csv")
    imp = _csv("XGBoost/shap_importance/article_ct_total_h3_shap_importance.csv")
    fall = _csv("XGBoost/shap_importance/article_ct_total_h3_shap_fall_drivers.csv")

    fig = plt.figure(figsize=(WIDTH_2COL, 74 * MM))
    ax_a = fig.add_axes([0.175, 0.15, 0.245, 0.76])
    ax_b = fig.add_axes([0.585, 0.15, 0.405, 0.76])

    # (a) barras horizontais, maior no topo
    grp = grp.sort_values("mean_abs_shap", ascending=True)
    y = np.arange(len(grp))
    ax_a.barh(y, grp["mean_abs_shap"], height=0.68, color=SLATE, zorder=2)
    for yi, v in zip(y, grp["mean_abs_shap"]):
        ax_a.text(v + 0.8, yi, f"{v:.1f}", va="center", ha="left",
                  fontsize=8, color=INK)
    ax_a.set_yticks(y, [GROUP_LABELS.get(g, g) for g in grp["group"]])
    ax_a.set_xlim(0, 52)
    ax_a.set_ylim(-0.6, len(grp) - 0.4)
    ax_a.set_xlabel("mean |SHAP|  (hm³)")
    ax_a.tick_params(axis="y", length=3)
    _grid_x(ax_a)

    # (b) dumbbell: período completo (azul) × meses de seca (laranja)
    d = (fall.rename(columns={"mean_shap": "drought"})
         .merge(imp[["feature", "mean_shap"]].rename(columns={"mean_shap": "full"}),
                on="feature", how="left")
         .sort_values("drought", ascending=True)
         .reset_index(drop=True))
    yb = np.arange(len(d))
    for yi, f_, dr in zip(yb, d["full"], d["drought"]):
        ax_b.plot([f_, dr], [yi, yi], color=CONNECTOR, lw=1.4,
                  solid_capstyle="butt", zorder=2)
    ax_b.scatter(d["full"], yb, s=26, color=BLUE, edgecolor="white",
                 linewidth=0.6, zorder=3, label="Full test period")
    ax_b.scatter(d["drought"], yb, s=26, color=ORANGE, edgecolor="white",
                 linewidth=0.6, zorder=4,
                 label="Drought months\n(observed ≤ Q20)")
    ax_b.axvline(0, color=INK, lw=0.8, zorder=1)
    ax_b.set_yticks(yb, d["feature"], fontstyle="italic")
    ax_b.set_xlim(-13.8, 13.8)
    ax_b.set_ylim(-1.45, len(d) - 0.4)
    ax_b.set_xticks([-10, -5, 0, 5, 10])
    ax_b.set_xlabel("mean signed SHAP  (hm³)")
    _grid_x(ax_b)
    ax_b.text(-13.3, -1.0, "← deepens drawdown",
              fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")
    ax_b.text(13.3, -1.0, "sustains storage →",
              fontsize=8, fontweight="bold", color=INK, ha="right", va="bottom")
    ax_b.legend(loc="center right", bbox_to_anchor=(1.0, 0.40),
                handletextpad=0.3, borderaxespad=0.2, fontsize=8)

    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


# --------------------------------------------------------------------------- #
# Fig. 3 — desempenho do LSTM (H = 3 e H = 12)
# --------------------------------------------------------------------------- #

BASELINE_LABELS = {
    "persistência-3": "Persistence-3",
    "persistência-12": "Persistence-12",
    "climatologia": "Climatology",
}
HEAT_METRICS = [("r", "r"), ("NSE", "NSE"), ("KGE", "KGE"),
                ("POD_q20", "POD"), ("HSS_q20", "HSS"), ("AUC", "AUC")]
SELECTED = {3: "C", 12: "D"}


def _blues_cmap() -> Colormap:
    return plt.get_cmap("Blues")


def _perf_table(h: int) -> pd.DataFrame:
    perf = _csv(f"LSTM/metrics/article_performance_h{h}.csv")
    models = perf[perf["preditor"].str.fullmatch(r"[A-D]\*?")]
    models = models.sort_values("RMSE")
    base = perf[perf["preditor"].isin(BASELINE_LABELS)].sort_values("RMSE")
    return pd.concat([models, base], ignore_index=True).assign(
        is_base=lambda t: t["preditor"].isin(BASELINE_LABELS))


def fig3_performance() -> plt.Figure:
    """(a,c) RMSE com escolhido destacado + ingênuos; (b,d) métricas 0–1."""
    fig = plt.figure(figsize=(WIDTH_2COL, 140 * MM))
    cmap = _blues_cmap()
    left_a, w_a = 0.165, 0.285
    left_b, w_b = 0.505, 0.37
    rows = [(3, 0.565, 0.385, "a", "b"), (12, 0.075, 0.385, "c", "d")]
    heat_axes = []
    for h, bottom, height, la, lb in rows:
        t = _perf_table(h)
        n = len(t)
        y = np.arange(n)[::-1]  # primeiro da tabela no topo
        ax = fig.add_axes([left_a, bottom, w_a, height])
        colors = [BASELINE if b else (ORANGE if p == SELECTED[h] else BLUE)
                  for p, b in zip(t["preditor"], t["is_base"])]
        ax.barh(y, t["RMSE"], height=0.70, color=colors, zorder=2)
        for yi, v in zip(y, t["RMSE"]):
            ax.text(v + 2.5, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=8, color=INK)
        labels = [BASELINE_LABELS.get(p, p) for p in t["preditor"]]
        ax.set_yticks(y, labels)
        for lab, b in zip(ax.get_yticklabels(), t["is_base"]):
            if b:
                lab.set_fontstyle("italic")
                lab.set_color(MUTED)
        n_base = int(t["is_base"].sum())
        ax.axhline(n_base - 0.5, color=CONNECTOR_SOFT, lw=0.8, ls=(0, (4, 3)),
                   xmin=0, xmax=1.02, clip_on=False, zorder=1)
        ax.set_xlim(0, 202)
        ax.set_xticks(np.arange(0, 201, 25))
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_xlabel("RMSE  (hm³)   ↓ better")
        _grid_x(ax)

        hx = fig.add_axes([left_b, bottom, w_b, height])
        mat = t[[m for m, _ in HEAT_METRICS]].to_numpy(dtype=float)
        # pcolormesh (vetorial) em vez de imshow (raster dentro do EPS)
        xe = np.arange(mat.shape[1] + 1) - 0.5
        ye = np.arange(n + 1) - 0.5
        hx.pcolormesh(xe, ye, mat, cmap=cmap, vmin=0, vmax=1,
                      edgecolors="none", antialiased=False)
        hx.set_xlim(-0.5, mat.shape[1] - 0.5)
        hx.set_ylim(n - 0.5, -0.5)
        for i in range(n):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                r, g, b = cmap(np.clip(v, 0, 1))[:3]
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                hx.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if lum < 0.55 else INK)
        # separador branco entre modelos e referências ingênuas
        hx.axhline(n - n_base - 0.5, color="white", lw=1.6)
        hx.set_xticks(range(len(HEAT_METRICS)), [lab for _, lab in HEAT_METRICS])
        hx.set_yticks([])
        for s in ("left", "bottom"):
            hx.spines[s].set_visible(False)
        hx.tick_params(axis="x", length=3)
        heat_axes.append(hx)
        panel_label(fig, ax, la)
        panel_label(fig, hx, lb, dx_mm=-3.0)

    cax = fig.add_axes([0.893, 0.30, 0.016, 0.45])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("metric value  (→ 1 best)", fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7, width=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Fig. 4 — séries observadas × previstas (2019–2025)
# --------------------------------------------------------------------------- #

def _drought_runs(dates: pd.Series, flag: pd.Series) -> list[tuple]:
    """Intervalos contínuos de meses de seca, de meio mês antes a meio depois."""
    runs, start = [], None
    flag = flag.to_numpy(dtype=bool)
    dates = pd.to_datetime(dates).reset_index(drop=True)
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f or i == len(flag) - 1) and start is not None:
            end = i if f else i - 1
            runs.append((dates[start] - pd.Timedelta(days=15),
                         dates[end] + pd.Timedelta(days=15)))
            start = None
    return runs


def _shade_and_band(ax: plt.Axes, x: pd.Series, lo: np.ndarray, hi: np.ndarray,
                    runs: list[tuple], band_color: str) -> None:
    """Meses de seca (opaco) + faixa P10–P90 com a sobreposição composta."""
    for a, b in runs:
        ax.axvspan(a, b, color=DROUGHT_SHADE, lw=0, zorder=0.5)
    ax.fill_between(x, lo, hi, color=blend(band_color, ALPHA_PI), lw=0,
                    zorder=1)
    # Sobreposição faixa × seca: mesma faixa recortada às janelas de seca,
    # com a cor composta (equivale ao alpha do original, sem transparência).
    over = ax.fill_between(x, lo, hi, color=blend(band_color, ALPHA_PI,
                                                  DROUGHT_SHADE),
                           lw=0, zorder=1.1)
    verts, codes = [], []
    for a, b in runs:
        x0, x1 = mdates.date2num(a), mdates.date2num(b)
        verts += [(x0, -1e4), (x1, -1e4), (x1, 1e4), (x0, 1e4), (x0, -1e4)]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    if verts:
        over.set_clip_path(MplPath(verts, codes), ax.transData)
    else:
        over.remove()


def fig4_series() -> plt.Figure:
    """(a) H=3: XGBoost D + LSTM C; (b) H=12: LSTM D; faixa P10–P90."""
    wf3 = _csv("LSTM/predictions/article_preds_wf_h3.csv", parse_dates=["target_date"])
    wf12 = _csv("LSTM/predictions/article_preds_wf_h12.csv", parse_dates=["target_date"])
    qr3 = _csv("LSTM/predictions/article_preds_qr_best_h3.csv", parse_dates=["target_date"])
    qr12 = _csv("LSTM/predictions/article_preds_qr_best_h12.csv", parse_dates=["target_date"])
    xgb = _csv("XGBoost/predictions/article_ct_total_h3_preds.csv",
               parse_dates=["target_date"])
    xgb = xgb[(xgb["split"] == "test") & (xgb["target_date"] >= wf3["target_date"].min())]

    fig = plt.figure(figsize=(WIDTH_2COL, 125 * MM))
    ax_a = fig.add_axes([0.095, 0.535, 0.895, 0.405])
    ax_b = fig.add_axes([0.095, 0.06, 0.895, 0.405], sharex=ax_a)

    def common(ax, wf, qr, band_color):
        x = wf["target_date"]
        runs = _drought_runs(x, wf["obs_seca20"])
        q = qr.set_index("target_date").reindex(x)
        _shade_and_band(ax, x, q["q10"].to_numpy(), q["q90"].to_numpy(),
                        runs, band_color)
        ax.plot(x, wf["y_base_clim"], color=CLIM, lw=1.0, ls=(0, (1, 1.6)),
                zorder=2)
        ax.plot(x, wf["y_base_persist"], color=PERSIST, lw=1.0,
                ls=(0, (5, 3)), zorder=2)
        ax.plot(x, wf["y_true"], color=OBS, lw=1.8, zorder=5)
        ax.set_ylim(160, 1090)
        ax.set_yticks(np.arange(200, 1001, 100))
        ax.set_ylabel("Cantareira System Storage (hm³)", fontsize=8.5)
        _grid_both(ax)
        return runs

    common(ax_a, wf3, qr3, BLUE)
    ax_a.plot(xgb["target_date"], xgb["y_pred"], color=ORANGE, lw=1.3, zorder=3)
    ax_a.plot(wf3["target_date"], wf3["y_C_pers"], color=BLUE, lw=1.3, zorder=4)
    common(ax_b, wf12, qr12, GREEN)
    ax_b.plot(wf12["target_date"], wf12["y_D_pers"], color=GREEN, lw=1.3, zorder=4)

    obs_h = Line2D([], [], color=OBS, lw=1.8, label="Observed")
    pers3 = Line2D([], [], color=PERSIST, lw=1.0, ls=(0, (5, 3)), label="Persistence-3")
    pers12 = Line2D([], [], color=PERSIST, lw=1.0, ls=(0, (5, 3)), label="Persistence-12")
    clim = Line2D([], [], color=CLIM, lw=1.0, ls=(0, (1, 1.6)), label="Climatology")
    dr = Patch(color=DROUGHT_SHADE, label="Observed drought (≤ Q20)")
    ax_a.legend(handles=[
        obs_h, Line2D([], [], color=ORANGE, lw=1.3, label="XGBoost (D)"),
        Line2D([], [], color=BLUE, lw=1.3, label="LSTM (C)"), pers3, clim,
        Patch(color=blend(BLUE, 0.30), label="LSTM 80% PI (P10–P90)"), dr],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
        columnspacing=1.2, handlelength=2.0, fontsize=8)
    ax_b.legend(handles=[
        obs_h, Line2D([], [], color=GREEN, lw=1.3, label="LSTM (D)"), pers12,
        clim, Patch(color=blend(GREEN, 0.30), label="LSTM 80% PI (P10–P90)"), dr],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
        columnspacing=1.2, handlelength=2.0, fontsize=8)

    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_b.set_xlim(pd.Timestamp("2018-12-15"), pd.Timestamp("2025-12-20"))
    plt.setp(ax_a.get_xticklabels(), visible=False)
    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


# --------------------------------------------------------------------------- #
# Fig. 5 — LSTM D, H = 12: determinístico × probabilístico
# --------------------------------------------------------------------------- #

def fig5_h12() -> plt.Figure:
    """(a) dispersão obs×prev; (b) intervalos P10–P90 ordenados pelo observado."""
    wf = _csv("LSTM/predictions/article_preds_wf_h12.csv")
    qr = _csv("LSTM/predictions/article_preds_qr_best_h12.csv")

    # Legendas abaixo do eixo x (revisão dos autores, 06/11): não cobrem dados.
    fig = plt.figure(figsize=(WIDTH_2COL, 96 * MM))
    ax_a = fig.add_axes([0.10, 0.30, 0.33, 0.635])
    ax_b = fig.add_axes([0.585, 0.30, 0.405, 0.635])
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                 fontsize=8, columnspacing=1.4, borderaxespad=0.0)

    # (a)
    yt, yp = wf["y_true"].to_numpy(), wf["y_D_pers"].to_numpy()
    seca = wf["obs_seca20"].to_numpy(dtype=bool)
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    r = float(np.corrcoef(yt, yp)[0, 1])
    nse = float(1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
    ratio = float(np.std(yp) / np.std(yt))
    ax_a.plot([150, 900], [150, 900], color=INK, lw=0.9, ls=(0, (4, 2.5)),
              zorder=1)
    ax_a.scatter(yt[~seca], yp[~seca], s=16, color=blend(GREEN, ALPHA_MARK),
                 edgecolor="white", linewidth=0.5, zorder=2)
    ax_a.scatter(yt[seca], yp[seca], s=16, color=blend(ORANGE, ALPHA_MARK),
                 edgecolor="white", linewidth=0.5, zorder=3)
    ax_a.set_xlim(150, 900)
    ax_a.set_ylim(150, 900)
    ax_a.set_xticks(np.arange(200, 901, 100))
    ax_a.set_yticks(np.arange(200, 901, 100))
    ax_a.set_xlabel("Observed storage (hm³)")
    ax_a.set_ylabel("Predicted storage (hm³)")
    _grid_both(ax_a)
    ax_a.text(0.04, 0.96,
              f"RMSE = {rmse:.0f} hm³\n$r$ = {r:.2f}\nNSE = {nse:.2f}\n"
              f"$\\sigma_{{pred}}/\\sigma_{{obs}}$ = {ratio:.2f}",
              transform=ax_a.transAxes, ha="left", va="top", fontsize=8,
              color=INK, linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#C9D3DD",
                        lw=0.6))
    ax_a.legend(handles=[
        Line2D([], [], color=INK, lw=0.9, ls=(0, (4, 2.5)), label="1:1"),
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=blend(GREEN, ALPHA_MARK),
               mec="white", mew=0.5, label="Normal months"),
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=blend(ORANGE, ALPHA_MARK),
               mec="white", mew=0.5, label="Observed drought (≤ Q20)")],
        handletextpad=0.4, handlelength=1.6, **below)

    # (b)
    q = qr.sort_values("y_true", kind="mergesort").reset_index(drop=True)
    x = np.arange(len(q))
    confident = (q["q10"] > q["thr20"]).to_numpy()
    for xi, lo, hi, c in zip(x, q["q10"], q["q90"], confident):
        ax_b.plot([xi, xi], [lo, hi], lw=2.0, solid_capstyle="butt",
                  color=blend(GREEN if c else BASELINE, ALPHA_MARK), zorder=2)
    ax_b.plot(x, q["y_true"], color=OBS, lw=1.4, zorder=3)
    ax_b.scatter(x, q["q50"], s=9, color="#20303A", zorder=4, lw=0)
    ax_b.set_xlim(-1, len(q))
    ax_b.set_ylim(150, 1010)
    ax_b.set_yticks(np.arange(200, 1001, 100))
    ax_b.set_xlabel("test months, ranked by observed storage")
    ax_b.set_ylabel("Cantareira System Storage (hm³)")
    _grid_both(ax_b)
    ax_b.legend(handles=[
        Line2D([], [], color=OBS, lw=1.4, marker="o", ms=3, mfc="#20303A",
               mec="#20303A", label="Observed"),
        Line2D([], [], ls="", marker="o", ms=4, mfc="#20303A", mec="#20303A",
               label="Median forecast (P50)"),
        Line2D([], [], color=blend(GREEN, ALPHA_MARK), lw=2.4,
               label="Confident: not drought\n(P10 > Q20)"),
        Line2D([], [], color=blend(BASELINE, ALPHA_MARK), lw=2.4,
               label="Indecisive\n(interval spans Q20)")],
        handletextpad=0.5, handlelength=1.8,
        **{**below, "bbox_to_anchor": (0.44, -0.17)})

    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


def fig5_summary() -> dict[str, float]:
    """Números do rodapé removido da Fig. 5b (vão para a legenda)."""
    qr = _csv("LSTM/predictions/article_preds_qr_best_h12.csv")
    width = float((qr["q90"] - qr["q10"]).mean())
    indecisive = float(((qr["q10"] <= qr["thr20"]) & (qr["q90"] >= qr["thr20"])).mean())
    warn = float((qr["q90"] < qr["thr20"]).mean())
    return {"mean_width_80": width, "share_indecisive": indecisive,
            "share_confident_warn": warn}


# --------------------------------------------------------------------------- #
# Fig. 6 — perturbação do LSTM: features × memória temporal
# --------------------------------------------------------------------------- #

def _perturbation(h: int) -> pd.DataFrame:
    return _csv(f"LSTM/metrics/article_xai_perturbation_h{h}.csv", index_col=0)


def fig6_perturbation() -> plt.Figure:
    """(a) importância relativa por feature; (b) fração por posição da janela."""
    p3, p12 = _perturbation(3), _perturbation(12)
    f3 = p3.sum(axis=0) / p3.to_numpy().sum()
    f12 = p12.sum(axis=0) / p12.to_numpy().sum()
    t3 = p3.sum(axis=1) / p3.to_numpy().sum()
    t12 = p12.sum(axis=1) / p12.to_numpy().sum()

    fig = plt.figure(figsize=(WIDTH_2COL, 68 * MM))
    ax_a = fig.add_axes([0.11, 0.17, 0.335, 0.73])
    ax_b = fig.add_axes([0.555, 0.17, 0.44, 0.73])

    # (a) ordem: importância em H=3 (decrescente), features só de H=12 no fim
    feats = list(f3.sort_values(ascending=False).index)
    feats += [f for f in f12.sort_values(ascending=False).index if f not in feats]
    y = np.arange(len(feats))[::-1]
    for yi, f in zip(y, feats):
        a, b = f3.get(f, np.nan), f12.get(f, np.nan)
        if np.isnan(a) or np.isnan(b):
            continue
        if b - a > 0.02:  # ganho relevante em H=12: seta verde
            ax_a.annotate("", xy=(b - 0.004, yi), xytext=(a, yi),
                          arrowprops=dict(arrowstyle="-|>", color=GREEN,
                                          lw=1.2, mutation_scale=7,
                                          shrinkA=0, shrinkB=0), zorder=2)
        else:
            ax_a.plot([a, b], [yi, yi], color=CONNECTOR_SOFT, lw=1.8,
                      solid_capstyle="butt", zorder=1)
    ax_a.scatter([f3.get(f, np.nan) for f in feats], y, s=30, color=BLUE,
                 edgecolor="white", linewidth=0.6, zorder=3)
    ax_a.scatter([f12.get(f, np.nan) for f in feats], y, s=30, color=GREEN,
                 edgecolor="white", linewidth=0.6, zorder=4)
    ax_a.set_yticks(y, feats, fontstyle="italic")
    ax_a.set_xlim(0, 0.243)
    ax_a.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax_a.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax_a.set_ylim(-0.5, len(feats) - 0.5)
    ax_a.set_xlabel("relative importance  (perturbation |Δŷ|)")
    _grid_x(ax_a)
    ax_a.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=5, mfc=BLUE, mec="white",
               label="H = 3 (exp. C)"),
        Line2D([], [], ls="", marker="o", ms=5, mfc=GREEN, mec="white",
               label="H = 12 (exp. D)")],
        loc="lower right", fontsize=8, handletextpad=0.3)

    # (b) memória temporal: áreas compostas sem transparência
    pos = list(p3.index)  # t-11 … t-0
    x = np.arange(len(pos))
    b_, g_ = t3.reindex(pos).to_numpy(), t12.reindex(pos).to_numpy()
    xf = np.linspace(0, len(pos) - 1, 600)
    bf, gf = np.interp(xf, x, b_), np.interp(xf, x, g_)
    lo = np.minimum(bf, gf)
    blue_area = blend(BLUE, ALPHA_AREA)
    green_area = blend(GREEN, ALPHA_AREA)
    overlap = blend(GREEN, ALPHA_AREA, blue_area)
    ax_b.fill_between(xf, 0, lo, color=overlap, lw=0, zorder=1)
    ax_b.fill_between(xf, lo, bf, where=bf > lo, color=blue_area, lw=0,
                      zorder=1)
    ax_b.fill_between(xf, lo, gf, where=gf > lo, color=green_area, lw=0,
                      zorder=1)
    ax_b.plot(x, b_, color=BLUE, lw=1.6, marker="o", ms=3.5, zorder=3,
              label="H = 3 (exp. C)")
    ax_b.plot(x, g_, color=GREEN, lw=1.6, marker="o", ms=3.5, zorder=4,
              label="H = 12 (exp. D)")
    ax_b.set_xticks(x, [p.replace("-", "−") for p in pos], fontsize=8)
    ax_b.set_xlim(-0.6, len(pos) - 0.4)
    ax_b.set_ylim(0, 0.285)
    ax_b.set_yticks(np.arange(0, 0.26, 0.05))
    ax_b.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax_b.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_xlabel("window position  (t−11 oldest  →  t−0 most recent)")
    ax_b.set_ylabel("share of importance")
    # Quadro de "recency" removido na revisão dos autores (06/11).
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                fontsize=8, handlelength=1.8, columnspacing=1.2)

    panel_label(fig, ax_a, "a")
    panel_label(fig, ax_b, "b")
    return fig


# --------------------------------------------------------------------------- #
# Online Resource 1 (Fig. S1) — correlação das features candidatas
# --------------------------------------------------------------------------- #

def _s1_groups(cols: list[str]) -> list[tuple[str, int, int]]:
    """Recupera os blocos (variável de entrada) na ordem da matriz S1."""
    def owner(c: str) -> str:
        if c == "Vol":
            return "Vol"
        if c.startswith("SPEI"):
            return "pr − eto"
        if c.startswith("SPI"):
            return "pr"
        if c.startswith("SRI"):
            return "q"
        if c in ("month_sin", "month_cos", "is_dry"):
            return "calendar"
        return c.split("_Lag")[0].split("_RollingMean")[0]

    bounds: list[tuple[str, int, int]] = []
    for i, c in enumerate(cols):
        o = owner(c)
        if bounds and bounds[-1][0] == o:
            bounds[-1] = (o, bounds[-1][1], i + 1)
        else:
            bounds.append((o, i, i + 1))
    return bounds


def esm1_correlation() -> plt.Figure:
    """Matriz de correlação de Pearson das features candidatas (77 × 77)."""
    corr = _csv("XGBoost/shap_importance/article_ct_total_h3_feature_correlation_S1.csv",
                index_col=0)
    cols = list(corr.columns)
    n = len(cols)
    fig = plt.figure(figsize=(WIDTH_2COL, 180 * MM))
    ax = fig.add_axes([0.165, 0.20, 0.70, 0.70 * WIDTH_2COL / (180 * MM)])
    e = np.arange(n + 1) - 0.5
    im = ax.pcolormesh(e, e, corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                       edgecolors="none", antialiased=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n), cols, rotation=90, fontsize=4.5)
    ax.set_yticks(range(n), cols, fontsize=4.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for lab in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        if lab.get_text() in ("qbal", "SRI12"):
            lab.set_color("#AD5528")
            lab.set_fontweight("bold")
    groups = _s1_groups(cols)
    for _, start, _end in groups[1:]:
        ax.axhline(start - 0.5, color="black", lw=0.6)
        ax.axvline(start - 0.5, color="black", lw=0.6)
    for name, start, end in groups:
        ax.text((start + end - 1) / 2, -1.2, name, ha="left", va="bottom",
                rotation=45, fontsize=6.5, fontweight="bold",
                rotation_mode="anchor")
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.2, 0.015,
                        pos.height * 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pearson correlation", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.5)
    cb.outline.set_linewidth(0.5)
    return fig


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #

FIGURES = {
    "Fig2": fig2_shap,
    "Fig3": fig3_performance,
    "Fig4": fig4_series,
    "Fig5": fig5_h12,
    "Fig6": fig6_perturbation,
}


def build_all(out_dir: Path) -> dict[str, dict[str, Path]]:
    """Gera Figs. 2–6 (EPS/PDF/TIFF/PNG) e o Online Resource 1 (PDF).

    Parameters
    ----------
    out_dir : Path
        Pasta de saída; subpastas ``eps/``, ``pdf/``, ``tiff_600dpi/`` e
        ``png_600dpi/`` são criadas.

    Returns
    -------
    dict[str, dict[str, Path]]
        Caminhos gravados por figura e formato.
    """
    out_dir = Path(out_dir)
    written: dict[str, dict[str, Path]] = {}
    with paper_style():
        for name, fn in FIGURES.items():
            written[name] = export(fn(), out_dir, name)
        fig = esm1_correlation()
        esm = out_dir / "ESM_1.pdf"
        fig.savefig(esm, format="pdf", facecolor="white")
        plt.close(fig)
        written["ESM_1"] = {"pdf": esm}
    return written


def main() -> None:
    """Ponto de entrada da linha de comando."""
    matplotlib.use("Agg")  # só na CLI: não troca o backend de quem importa o módulo
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUTPUTS / "forecast_paper",
                    help="pasta de saída (padrão: Previsao/Outputs/forecast_paper)")
    args = ap.parse_args()
    written = build_all(args.out)
    for name, paths in written.items():
        print(name, *(str(p) for p in paths.values()), sep="\n  ")
    s = fig5_summary()
    print(f"Fig5b (para a legenda): largura média 80% = {s['mean_width_80']:.0f} hm³; "
          f"indecisos = {s['share_indecisive']:.0%}; "
          f"alerta confiante = {s['share_confident_warn']:.0%}")


if __name__ == "__main__":
    main()
