#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Mapa da area de estudo - Sistema Cantareira (v3), Fig. 1 do artigo de previsao.

Figura DESCRITIVA: le camadas de dados/mapa_dados.gpkg (ja recortadas), reprojeta e aplica estilo.
Nao calcula resultados, nao delineia bacias, nao inventa geometrias.

Rodar:  python -I mapa_cantareira.py
Saidas: saidas/F_study_area_v2.{pdf,png,svg}   (v0 e v1 arquivadas em versoes/)
Edite SOMENTE o bloco de CONFIGURACAO abaixo (e o corpo do script, se preciso). Nao edite saidas/.

v1 (04/11/2026): sem nomes de rios; sem painel (d) RMSP e sem a legenda
da coluna esquerda; sem Aguas Claras; sem "CESP"; sem numeracao (nomes dos reservatorios direto no mapa);
interligacoes so com os tuneis (sem Canal Juqueri e sem conduto de recalque); paleta e estilo do artigo
de previsao do volume do Sistema Cantareira (DejaVu Sans; titulos 10 pt; marcas e legenda 8 pt).
v2 (04/11/2026): "Paraiba do Sul basin" e "Jaguari (PBS)"; legenda so com reservatorios, rios, tuneis e
transposicao; tuneis em cinza para separar dos reservatorios.
v3 (06/11/2026, diretrizes de arte da revista-alvo): fonte Arial; SEM transparencia
(alpha substituido pela cor opaca equivalente sobre o fundo, para o EPS sair 100% vetorial); raster a
600 dpi (combination art); saidas tambem em EPS e TIFF (RGB, LZW). Titulos dos paineis mantidos.
"""
# =====================================================================================
# CONFIGURACAO (editar aqui)
# =====================================================================================
VERSAO = "v3"
IDIOMA = "en"                 # "en" ou "pt"
LARGURA_MM = 174              # revista: 84 mm (1 coluna) ou 174 mm (pagina); altura maxima 234 mm
ALTURA_MM = 130               # altura aproximada (mm); o recorte final usa bbox_inches="tight"
DPI_PNG = 600                 # revista: combination art (cor + texto) >= 600 dpi; vale para PNG e TIFF
FORMATOS = ("pdf", "eps", "png", "tiff", "svg")
CRS_MAPA = "EPSG:31983"       # SIRGAS 2000 / UTM 23S (mesma convencao do f1_mapa_sim.py)
FONTE = "Arial"               # revista: Helvetica ou Arial, 8-12 pt
TAM_TITULO = 10               # pt: titulos dos paineis "(a)", "(b)"
TAM_ROTULO = 9                # pt: rotulos de eixos (nao usados no mapa; reservado)
TAM_MARCA = 8                 # pt: marcas (coordenadas), legenda e nomes no mapa

# camadas ligadas/desligadas
LIGAR = dict(
    rios_ct=True, rios_jaguari_pbs=True, tuneis=True, transposicao=True,
    bacia_jaguari_pbs=True, nomes_reservatorios=True,
    inset_america_sul=True, inset_sp=True,
    graticula=True, escala=True, norte=True, legenda=True,
)

# Paleta do artigo de previsao do volume do Cantareira (hex). Mapeamento para o mapa (proposta v1):
#   conjuntos de features A-D -> 4 bacias incrementais; observado (azul-marinho) -> reservatorios; tuneis em cinza;
#   limiar de seca (vermelho) -> transposicao; persistencia (cinza) -> bacia do Jaguari (Paraiba do Sul);
#   destaque SHAP azul -> rios.
COR_BACIA = {"CT_JAGJAC": "#d97742", "CT_CACH": "#7b5aa6", "CT_ATIB": "#2a9d8f", "CT_PCASTR": "#c9a227"}
ALFA_BACIA = 0.55
COR_RES = "#1f3a5f"
COR_RIO = "#0072B2"
COR_TUNEL = "#666666"     # v2: cinza; mais escuro que a borda da bacia PBS (#999999)
COR_TRANSP = "#c44e52"
COR_PBS_FILL = "#f2f2f2"
COR_PBS_EDGE = "#999999"
COR_SPEST = "#1f3a5f"     # estado de SP nos insets (preenchimento com transparencia)
COR_MSP = "#c44e52"       # municipio de Sao Paulo (capital) no inset (c)
COR_BR = "#d9d9d9"
COR_PAISES = "#f2f2f2"

# espessuras (pt)
LW = dict(bacia=0.6, rio=0.6, rio_pbs=0.5, tunel=1.6, transp=1.6, res_edge=0.3, grat=0.5)

# folga (m) ao redor do conjunto das bacias no painel principal
PAD_M = 6000

# nomes dos reservatorios direto no mapa: ID_Map -> (dx, dy, ha) em m, relativos ao ponto representativo
# (15 = Aguas Claras fica fora, por decisao dos autores)
DESLOC_NOME = {11: (-7500, 4500, "right"), 12: (5500, 3000, "left"), 13: (-7000, -3500, "right"),
               14: (4000, -6000, "left"), 19: (9000, 2500, "left")}
# posicao (fracao do eixo) do rotulo da bacia do Jaguari no Paraiba do Sul e ponto da linha-guia
POS_ROT_PBS = (0.855, 0.15)
PONTA_ROT_PBS = (0.64, 0.19)

# textos (EN segue o manuscrito do artigo de previsao, secao 2.1)
TXT = {
    "en": dict(
        main="Cantareira Water Supply System",
        res={11: "Jaguari/Jacareí", 12: "Cachoeira", 13: "Atibainha", 14: "Paiva Castro", 19: "Jaguari (PBS)"},
        basins="Incremental basins", reservoir="Reservoirs", river="Rivers", tunnel="Tunnels",
        transp="Inter-basin transfer: Jaguari–Atibainha",
        pbs_basin="Paraíba do Sul basin", pbs_basin_leg="Paraíba do Sul basin",
        sa="South America", sp="São Paulo State", msp="São Paulo", km="km", sp_lbl="SP", br_lbl="Brazil"),
    "pt": dict(
        main="Sistema Cantareira",
        res={11: "Jaguari/Jacareí", 12: "Cachoeira", 13: "Atibainha", 14: "Paiva Castro", 19: "Jaguari (PBS)"},
        basins="Bacias incrementais", reservoir="Reservatórios", river="Rios", tunnel="Túneis",
        transp="Transposição Jaguari–Atibainha",
        pbs_basin="Bacia do Paraíba do Sul", pbs_basin_leg="Bacia do Paraíba do Sul",
        sa="América do Sul", sp="Estado de São Paulo", msp="São Paulo", km="km", sp_lbl="SP", br_lbl="Brasil"),
}

# =====================================================================================
# FIM DA CONFIGURACAO
# =====================================================================================
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import box

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
GPKG = AQUI / "dados" / "mapa_dados.gpkg"
SAIDAS = AQUI / "saidas"
CRS_GEO = "EPSG:4674"
T = TXT[IDIOMA]
PE = [pe.withStroke(linewidth=2.0, foreground="white")]
IDS_RES = list(DESLOC_NOME)   # reservatorios desenhados (sem Aguas Claras, ID 15)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONTE], "font.size": TAM_MARCA,
                     "axes.titlesize": TAM_TITULO, "axes.labelsize": TAM_ROTULO,
                     "xtick.labelsize": TAM_MARCA, "ytick.labelsize": TAM_MARCA, "legend.fontsize": TAM_MARCA,
                     "legend.frameon": False, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "hatch.linewidth": 0.4, "figure.facecolor": "white"})


def opaca(cor, alfa, fundo="#ffffff"):
    """Cor opaca equivalente a `cor` com transparencia `alfa` sobre `fundo` (o EPS nao suporta alpha)."""
    c = np.array(to_rgb(cor)); f = np.array(to_rgb(fundo))
    return to_hex(alfa * c + (1 - alfa) * f)


COR_GRADE = opaca("#999999", 0.3)   # grade: antes #999999 com alpha 0,3


def L(nome, crs=CRS_MAPA):
    return gpd.read_file(GPKG, layer=nome).to_crs(crs)


def graticula(ax, passo=0.25):
    tr = Transformer.from_crs(CRS_GEO, CRS_MAPA, always_xy=True)
    inv = Transformer.from_crs(CRS_MAPA, CRS_GEO, always_xy=True)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    lo0, la0 = inv.transform(x0, y0); lo1, la1 = inv.transform(x1, y1)
    lons = np.arange(np.ceil(lo0 / passo) * passo, lo1, passo)
    lats = np.arange(np.ceil(la0 / passo) * passo, la1, passo)
    lr = np.linspace(la0 - .2, la1 + .2, 30); lor = np.linspace(lo0 - .2, lo1 + .2, 30)
    # grade tracejada clara (equivalente opaco da transparencia 0,3), como no artigo de previsao.
    # v3: zorder 3.2 (acima do preenchimento das bacias, abaixo dos contornos), porque o preenchimento
    # opaco esconderia a grade que antes aparecia atraves do alpha.
    for lo in lons:
        x, y = tr.transform(np.full_like(lr, lo), lr); ax.plot(x, y, color=COR_GRADE, lw=LW["grat"], ls="--", zorder=3.2)
    for la in lats:
        x, y = tr.transform(lor, np.full_like(lor, la)); ax.plot(x, y, color=COR_GRADE, lw=LW["grat"], ls="--", zorder=3.2)
    ax.set_xticks([tr.transform(lo, la0)[0] for lo in lons]); ax.set_xticklabels([f"{abs(l):.2f}°W" for l in lons])
    ax.set_yticks([tr.transform(lo0, la)[1] for la in lats])
    ax.set_yticklabels([f"{abs(l):.2f}°S" for l in lats], rotation=90, va="center")
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def escala(ax, km=20):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    Lm = km * 1000; xs = x1 - 0.05 * (x1 - x0) - Lm; ys = y0 + 0.05 * (y1 - y0)
    ax.plot([xs, xs + Lm], [ys, ys], color="k", lw=1.6, solid_capstyle="butt", zorder=21)
    for xx in (xs, xs + Lm):
        ax.plot([xx, xx], [ys - 0.007 * (y1 - y0), ys + 0.007 * (y1 - y0)], color="k", lw=0.6, zorder=21)
    ax.text(xs + Lm / 2, ys + 0.012 * (y1 - y0), f"{km} {T['km']}", ha="center", va="bottom", zorder=21, path_effects=PE)


def norte(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); xa = x1 - 0.06 * (x1 - x0)
    ax.annotate("", xy=(xa, y1 - 0.04 * (y1 - y0)), xytext=(xa, y1 - 0.14 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9, mutation_scale=8), zorder=21)
    ax.text(xa, y1 - 0.155 * (y1 - y0), "N", ha="center", va="top", fontweight="bold", zorder=21)


def linhas(ax, geoms, **kw):
    """Desenha linhas/contornos com ax.plot (aceita tracejados personalizados)."""
    for g in geoms:
        if g.is_empty:
            continue
        partes = g.geoms if hasattr(g, "geoms") else [g]
        for p in partes:
            if p.geom_type == "Polygon":
                ax.plot(*p.exterior.xy, **kw)
                for i in p.interiors:
                    ax.plot(*i.xy, **kw)
            elif p.geom_type == "MultiPolygon":
                linhas(ax, [p], **kw)
            else:
                ax.plot(*p.xy, **kw)


def sem_bordas_sup_dir(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    bac = L("bacias_ct"); res = L("reservatorios"); rios = L("rios_ct")
    il = L("interligacoes_ct"); tr = L("transposicao_jaguari_atibainha")
    b19 = L("bacia_jaguari_pbs"); rp = L("rios_jaguari_pbs")
    res = res[res.ID_Map.isin(IDS_RES)]
    tun = il[il.nome.str.startswith("Tunel")]

    allg = gpd.GeoSeries(list(bac.geometry) + list(b19.geometry), crs=CRS_MAPA)
    xmin, ymin, xmax, ymax = allg.total_bounds
    xmin -= PAD_M; xmax += PAD_M; ymin -= PAD_M; ymax += PAD_M
    ext = box(xmin, ymin, xmax, ymax)

    W = LARGURA_MM / 25.4; H = ALTURA_MM / 25.4
    fig = plt.figure(figsize=(W, H))
    asp = (xmax - xmin) / (ymax - ymin)
    main_w = 0.64; main_h = main_w * W / asp / H
    axm = fig.add_axes([0.345, 0.97 - main_h, main_w, main_h])
    axm.set_xlim(xmin, xmax); axm.set_ylim(ymin, ymax); axm.set_aspect("equal", adjustable="box")

    # ---- painel principal
    if LIGAR["bacia_jaguari_pbs"]:
        b19.plot(ax=axm, fc=COR_PBS_FILL, ec=COR_PBS_EDGE, lw=0, hatch="////", zorder=2)
        linhas(axm, b19.geometry, color=COR_PBS_EDGE, lw=LW["bacia"] + .3, ls=(0, (4, 2)), zorder=2.5)
    for _, r in bac.iterrows():
        g = gpd.GeoSeries([r.geometry], crs=CRS_MAPA)
        g.plot(ax=axm, fc=opaca(COR_BACIA[r.BACIA], ALFA_BACIA), ec="none", zorder=3)
        linhas(axm, g, color=COR_BACIA[r.BACIA], lw=LW["bacia"], zorder=3.5)
    if LIGAR["rios_ct"]:
        rios.plot(ax=axm, color=COR_RIO, lw=LW["rio"], zorder=5)
    if LIGAR["rios_jaguari_pbs"]:
        rp.plot(ax=axm, color=COR_RIO, lw=LW["rio_pbs"], zorder=5)
    res.plot(ax=axm, fc=COR_RES, ec=COR_RES, lw=LW["res_edge"], zorder=6)
    if LIGAR["tuneis"]:
        linhas(axm, tun.geometry, color=COR_TUNEL, lw=LW["tunel"], solid_capstyle="butt", zorder=7)
    if LIGAR["transposicao"]:
        linhas(axm, tr.geometry, color=COR_TRANSP, lw=LW["transp"], ls=(0, (4, 1.6)), zorder=8)

    # nomes dos reservatorios direto no mapa (sem numeracao)
    if LIGAR["nomes_reservatorios"]:
        for _, r in res.iterrows():
            c = r.geometry.representative_point(); dx, dy, ha = DESLOC_NOME[r.ID_Map]
            axm.plot([c.x, c.x + dx * 0.85], [c.y, c.y + dy * 0.85], color="k", lw=0.4, zorder=29)
            axm.text(c.x + dx, c.y + dy, T["res"][r.ID_Map], ha=ha, va="center", fontweight="bold",
                     zorder=30, path_effects=PE)
    if LIGAR["bacia_jaguari_pbs"]:
        axm.annotate(T["pbs_basin"], xy=PONTA_ROT_PBS, xycoords="axes fraction", xytext=POS_ROT_PBS,
                     textcoords="axes fraction", ha="center", va="center", style="italic", zorder=30, path_effects=PE,
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#333"))

    if LIGAR["graticula"]:
        graticula(axm)
    if LIGAR["escala"]:
        escala(axm)
    if LIGAR["norte"]:
        norte(axm)
    sem_bordas_sup_dir(axm)
    axm.set_title(f"(a) {T['main']}", loc="left", pad=3, fontweight="bold")

    # ---- insets (coluna esquerda)
    uf = L("uf_brasil", CRS_GEO); sa = L("america_sul", CRS_GEO); br = L("brasil", CRS_GEO)
    msp = L("municipio_sao_paulo", CRS_GEO)
    eg = gpd.GeoSeries([ext], crs=CRS_MAPA).to_crs(CRS_GEO).total_bounds
    colx, colw = 0.01, 0.31

    def boxax(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#999999")

    y = 0.97
    if LIGAR["inset_america_sul"]:
        hh = 0.36; y -= hh
        axs = fig.add_axes([colx, y, colw, hh])
        sa.plot(ax=axs, fc=COR_PAISES, ec="#999999", lw=0.3)
        br.plot(ax=axs, fc=COR_BR, ec="#555555", lw=0.4)
        uf[uf.SIGLA_UF == "SP"].plot(ax=axs, fc=COR_SPEST, ec=COR_SPEST, lw=0.5)
        axs.set_xlim(-82, -33); axs.set_ylim(-57, 14); axs.set_aspect(1 / np.cos(np.radians(20)))
        axs.text(-58, -10, T["br_lbl"], ha="center", va="center", zorder=11)
        boxax(axs); axs.set_title(f"(b) {T['sa']}", loc="left", pad=3, fontweight="bold")
        y -= 0.07
    if LIGAR["inset_sp"]:
        hh = 0.30; y -= hh
        axp = fig.add_axes([colx, y, colw, hh])
        uf.plot(ax=axp, fc=COR_PAISES, ec="#bbbbbb", lw=0.3)
        uf[uf.SIGLA_UF == "SP"].plot(ax=axp, fc=opaca(COR_SPEST, 0.18, COR_PAISES), ec="none")
        uf[uf.SIGLA_UF == "SP"].plot(ax=axp, fc="none", ec=COR_SPEST, lw=0.7)
        msp.plot(ax=axp, fc=COR_MSP, ec=COR_MSP, lw=0.4, zorder=5)
        axp.add_patch(Rectangle((eg[0], eg[1]), eg[2] - eg[0], eg[3] - eg[1], fc="none", ec="k", lw=0.8, zorder=10))
        axp.text(eg[0] - 0.12, eg[3], "(a)", ha="right", va="top", zorder=11)
        c = msp.union_all().representative_point()
        axp.annotate(T["msp"], xy=(c.x, c.y), xytext=(c.x - 1.6, c.y - 1.1), ha="right", va="center", zorder=11,
                     path_effects=PE, arrowprops=dict(arrowstyle="-", lw=0.4, color="#333"))
        axp.set_xlim(-53.5, -44); axp.set_ylim(-25.6, -19.6); axp.set_aspect(1 / np.cos(np.radians(23)))
        axp.text(-50.5, -22, T["sp_lbl"], fontweight="bold", ha="center", zorder=11)
        boxax(axp); axp.set_title(f"(c) {T['sp']}", loc="left", pad=3, fontweight="bold")

    # ---- legenda (abaixo do painel principal, sem moldura e sem numeracao)
    if LIGAR["legenda"]:
        # v2: sem as entradas das bacias incrementais e da bacia do Paraiba do Sul (decisao dos autores)
        h = [Patch(fc=COR_RES, ec=COR_RES), Line2D([], [], color=COR_RIO, lw=LW["rio"] + .3),
             Line2D([], [], color=COR_TUNEL, lw=LW["tunel"]), Line2D([], [], color=COR_TRANSP, lw=LW["transp"], ls=(0, (4, 1.6)))]
        lb = [T["reservoir"], T["river"], T["tunnel"], T["transp"]]
        pm = axm.get_position()
        fig.legend(h, lb, loc="upper left", bbox_to_anchor=(pm.x0 - 0.01, pm.y0 - 0.045), ncol=2, frameon=False,
                   handlelength=2.6, handleheight=0.9, labelspacing=0.4, columnspacing=1.2, borderaxespad=0,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    SAIDAS.mkdir(exist_ok=True)
    for e in FORMATOS:
        out = SAIDAS / f"F_study_area_{VERSAO}.{e}"
        kw = dict(dpi=DPI_PNG) if e in ("png", "tiff") else {}
        if e == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03, facecolor="white", **kw)
        if e in ("png", "tiff"):   # revista: RGB de 8 bits (o matplotlib grava RGBA)
            from PIL import Image
            with Image.open(out) as im:
                rgb = im.convert("RGB")
            skw = {"dpi": (DPI_PNG, DPI_PNG)}
            if e == "tiff":
                skw["compression"] = "tiff_lzw"
            rgb.save(out, **skw)
        print("ok", out)


if __name__ == "__main__":
    main()
