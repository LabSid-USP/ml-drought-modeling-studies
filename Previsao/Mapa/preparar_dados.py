#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Gera dados/mapa_dados.gpkg a partir das fontes originais (rastreabilidade).

Rodar (so se precisar regenerar dados/; o pacote ja vem com dados/ pronto):
    python -I preparar_dados.py --sig-dir <pasta SIG> --natural-earth <ne_50m_admin_0_countries.shp>

So le, filtra, recorta (clip) e reprojeta para EPSG:4674. Nada e calculado nem delineado.
Nomes de colunas/valores em ASCII. A pasta SIG (camadas FCTH/LabSid, nao versionada) e o
shapefile do Natural Earth sao passados por argumento; ver README.md (Proveniencia).
"""
import argparse
import warnings
from pathlib import Path

import geopandas as gpd

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "dados" / "mapa_dados.gpkg"
CRS = "EPSG:4674"

ap = argparse.ArgumentParser(description="Gera dados/mapa_dados.gpkg a partir das fontes originais.")
ap.add_argument("--sig-dir", type=Path, required=True,
                help="pasta SIG com SHP/ e GEOJSON/ (camadas FCTH/LabSid; nao versionada)")
ap.add_argument("--natural-earth", type=Path, required=True,
                help="ne_50m_admin_0_countries.shp do Natural Earth 5.1.1")
args = ap.parse_args()

SIG = args.sig_dir
F_BACIAS = SIG / "SHP" / "mar2026" / "BACIAS" / "CT.gpkg"
F_RESERV = SIG / "SHP" / "mar2026" / "Reservatorios.gpkg"
F_RIOS = SIG / "SHP" / "mar2026" / "Rios.gpkg"
F_HA = SIG / "GEOJSON" / "HA_WatershedFeature.geojson"
F_INTERL = SIG / "GEOJSON" / "Interligacoes.geojson"
F_TRANSP = SIG / "GEOJSON" / "Transposicoes.geojson"
F_RMSP = SIG / "SHP" / "mar2026" / "RMSP.gpkg"
F_PAIS = SIG / "SHP" / "Texto" / "BR_Pais_2024.shp"
F_UF = SIG / "SHP" / "Texto" / "BR_UF_2024.shp"
F_MUN_SP = SIG / "SHP" / "mar2026" / "OUTROS" / "SP.gpkg"   # 645 municipios de SP (IBGE; ano [CHECAR])
F_NE = args.natural_earth
# Natural Earth 5.1.1, 50m admin_0 countries:
# https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip (baixado em 04/11/2026)

def rd(p, **kw):
    g = gpd.read_file(p, **kw)
    return g.to_crs(CRS) if g.crs is None or g.crs.to_string() != CRS else g

if SAIDA.exists():
    SAIDA.unlink()

def save(g, nome):
    g = g.reset_index(drop=True)
    g.to_file(SAIDA, layer=nome, driver="GPKG")
    print(f"{nome}: {len(g)} feicoes")

# 1) Bacias incrementais (4 feicoes). Atencao: CT_PCASTR (337,6 km2) inclui a bacia de Aguas Claras (HA ID_Map 15).
bac = rd(F_BACIAS)[["BACIA", "geometry"]]
bac["res_num"] = bac.BACIA.map({"CT_JAGJAC": 1, "CT_CACH": 2, "CT_ATIB": 3, "CT_PCASTR": 4})
save(bac, "bacias_ct")
bacU = bac.to_crs(31983).union_all()

# 2) Reservatorios (Cantareira: ID_Map 11-15; Jaguari CESP = 19). Nomes ASCII atribuidos por ID_Map.
nomes = {11: "Jaguari/Jacarei", 12: "Cachoeira", 13: "Atibainha", 14: "Paiva Castro", 15: "Aguas Claras", 19: "Jaguari (CESP)"}
res = rd(F_RESERV)
res = res[res.ID_Map.isin(nomes)][["ID_Map", "geometry"]].copy()
res["nome"] = res.ID_Map.map(nomes)
save(res, "reservatorios")

# 3) Rios formadores: Rios.gpkg com Id_Map 11xx-15xx (level2 11-15), recortados as 4 bacias.
#    Nomes vem de HA_WatershedFeature.geojson (zoom 3, mesmo Id_Map), em ASCII.
ha = rd(F_HA)
rios = rd(F_RIOS)
rios = rios[rios.level2.isin([11, 12, 13, 14, 15])].copy()
nm = ha.set_index("ID_Map").STR_Name
rios["nome_orig"] = rios.Id_Map.map(nm)
grupo = {1101: "Jacarei", 1102: "Jaguari", 1103: "Jaguari", 1104: "Jaguari", 1105: "Camanducaia",
         1201: "Cachoeira", 1202: "Cachoeira", 1203: "Cachoeira", 1204: "Canca",
         1301: "Atibainha", 1302: "Atibainha", 1401: "Juqueri", 1402: "Juqueri", 1403: "Pinheiros",
         1404: "Juqueri-Mirim", 1405: "Juqueri-Mirim", 1406: "Juqueri-Mirim", 1407: "Santa Ines",
         1501: "Aguas Claras", 1502: "Santa Ines"}
rios["grupo"] = rios.Id_Map.map(grupo)
rios = rios[["Id_Map", "grupo", "geometry"]]
r31 = rios.to_crs(31983)
r31["geometry"] = r31.geometry.intersection(bacU)
r31 = r31[~r31.geometry.is_empty]
save(r31.to_crs(CRS), "rios_ct")

# 4) Interligacoes do Cantareira (SystemID 0) e transposicao Jaguari (CESP) -> Atibainha (ID_Map 4).
il = rd(F_INTERL)
il = il[il.SystemID == 0][["ID_Map", "geometry"]].copy()
il["nome"] = il.ID_Map.map({1: "Tunel 7", 2: "Tunel 6", 3: "Tunel 5", 4: "Tunel 4", 0: "Conduto de recalque (ESI - Tunel 4)",
                            5: "Tunel 3", 6: "Tunel 2", 10: "Canal Juqueri (Atibainha - Paiva Castro)"})
save(il, "interligacoes_ct")
tr = rd(F_TRANSP)
tr = tr[(tr.SystemID == 0) & (tr.ID_Map == 4)][["ID_Map", "geometry"]].copy()
tr["nome"] = "Jaguari (CESP) para Atibainha"
save(tr, "transposicao_jaguari_atibainha")

# 5) Bacia do Jaguari (Paraiba do Sul): feicao ja existente em HA_WatershedFeature (zoom 2, ID_Map 19).
b19 = ha[ha.ID_Map == 19][["ID_Map", "area_km2", "geometry"]].copy()
b19["nome"] = "Bacia Jaguari (CESP)"
save(b19, "bacia_jaguari_pbs")
b19U = b19.to_crs(31983).union_all()
rp = rd(F_RIOS)
rp = rp[rp.Id_Map.isin([1901, 1902, 1903, 1904, 1905, 1906, 1907])].copy()
rp["grupo"] = rp.Id_Map.map({1901: "Jaguari", 1902: "Jaguari"}).fillna("Peixe")
rp = rp[["Id_Map", "grupo", "geometry"]].to_crs(31983)
rp["geometry"] = rp.geometry.intersection(b19U)
rp = rp[~rp.geometry.is_empty]
save(rp.to_crs(CRS), "rios_jaguari_pbs")

# 6) Contexto: RMSP, municipio de Sao Paulo (CD_MUN 3550308), UF SP, outras UFs, Brasil, paises da America do Sul.
save(rd(F_RMSP)[["name", "geometry"]], "rmsp")
mun = rd(F_MUN_SP)
msp = mun[mun.CD_MUN.astype(str) == "3550308"][["CD_MUN", "geometry"]].copy()
msp["nome"] = "Sao Paulo"
save(msp, "municipio_sao_paulo")
uf = rd(F_UF)[["SIGLA_UF", "geometry"]].copy()
uf["geometry"] = uf.simplify(0.01)   # simplificacao grafica
save(uf, "uf_brasil")
pais = rd(F_PAIS)[["PAIS", "geometry"]].copy()
pais["geometry"] = pais.simplify(0.01)
save(pais, "brasil")
ne = gpd.read_file(F_NE)
sa = ne[ne.CONTINENT == "South America"][["NAME", "ISO_A3", "geometry"]].to_crs(CRS)
save(sa, "america_sul")
print("ok ->", SAIDA)
