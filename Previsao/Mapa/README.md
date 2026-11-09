# Study-area map (Fig. 1)

`mapa_cantareira.py` draws the study-area map of the Cantareira Water Supply
System. It reads only `dados/mapa_dados.gpkg`, which holds 12 layers in
EPSG:4674, and writes `saidas/F_study_area_v3.{pdf,eps,png,tiff,svg}`. Raster
outputs are 600 dpi RGB. The script computes nothing: it reprojects the layers
and applies the style.

```bash
python scripts/download_data.py --only R3     # gets dados/mapa_dados.gpkg from HydroShare
python Previsao/Mapa/mapa_cantareira.py
```

The language (EN/PT), colours, layers and label positions are set in the
`CONFIGURACAO` block at the top of the script.

`preparar_dados.py` documents how `dados/mapa_dados.gpkg` was built from the
original layers. It only reads, filters, clips and reprojects them; nothing is
delineated. The original layers are not distributed: the FCTH/LabSid GIS and the
Sabesp SSD layers. The script records each step, and the HydroShare resource
Readme gives the source of every layer. To rebuild the GeoPackage:

```bash
python Previsao/Mapa/preparar_dados.py --sig-dir <GIS folder> \
    --natural-earth <ne_50m_admin_0_countries.shp>
```

Natural Earth 5.1.1, 1:50m admin-0 countries:
https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip
(SHA-256 of the zip: `5fed433373581fa648920435f937d95f2d3c0200e067409c6478dcdf1b853139`).
