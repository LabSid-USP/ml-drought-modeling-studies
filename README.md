# ml-drought-modeling-studies

This repository has the code for the machine-learning studies of drought and
water supply in the São Paulo Metropolitan Region (Brazil), developed at LabSid,
Escola Politécnica, Universidade de São Paulo.

**Release v1.0.0** has the code for the **forecasting paper**. The paper
forecasts the total storage of the Cantareira Water Supply System 3 and 12 months
ahead, using XGBoost and LSTM models with hydroclimatic and operational
predictors, and interprets the models with SHAP, saliency and perturbation
analyses. The release reproduces every figure and table of the paper. Later
releases will add the regime-mapping study.

## Repository layout

```
Previsao/
├── Scripts/                 Python package: feature engineering, models, metrics, plots
│   ├── XGBoost/             gradient-boosting pipeline, SHAP, PDP/ICE
│   ├── LSTM/                recurrent network, walk-forward training, explainability
│   └── article_figures.py   final figures of the paper (Figs. 2–6, Fig. S1)
├── Notebooks/
│   ├── XGBoost/xgboost_article.ipynb   XGBoost experiments (H = 3)
│   └── LSTM/lstm_article.ipynb         LSTM experiments (H = 3, 12) and quantile forecasts
├── Mapa/                    study-area map (Fig. 1)
├── Outputs/                 model outputs read by the figures (article_*.csv)
└── Dados/                   input series (downloaded, see Data)
scripts/download_data.py     downloads the data from HydroShare
docs/methodological_rules.md methodological rules cited in the notebooks
```

The notebooks and code comments are in Portuguese; function and variable names
are in English.

## Quick start

```bash
conda env create -f environment.yml
conda activate ml-drought-modeling-studies
python scripts/download_data.py          # input series and spatial layers (HydroShare)
```

| Output | Command |
|---|---|
| Fig. 1 (study area) | `python Previsao/Mapa/mapa_cantareira.py` |
| Figs. 2–6 and Fig. S1 | `python Previsao/Scripts/article_figures.py --out Previsao/Outputs/forecast_paper` |
| XGBoost results (ablation, skill scores, Diebold–Mariano tests, SHAP) | run `Previsao/Notebooks/XGBoost/xgboost_article.ipynb` |
| LSTM results (walk-forward skill, drought detection, quantile forecasts, explainability) | run `Previsao/Notebooks/LSTM/lstm_article.ipynb` |

`article_figures.py` only needs the `article_*.csv` files that come with the
repository, so it runs right after cloning, without downloading the data. The
notebooks write their tables to `Previsao/Outputs/` (`article_*.csv`) and print
them inline.

**Reproducibility.** The XGBoost notebook is deterministic: with the versions
pinned in `environment.yml` it reproduces the stored outputs exactly. The LSTM
notebook reuses the walk-forward forecasts stored in
`Previsao/Outputs/LSTM/predictions/article_preds_wf_h{3,12}.csv` and only
retrains the parts that follow them: the quantile-regression model and the
saliency and perturbation analyses. Those parts are not bit-reproducible:
neural-network training varies with the hardware and the thread scheduling,
and the network weights are initialised from the global random state, not from
the experiment seed. Retraining them gives results of the same order that can
differ from the stored ones. In a test, the median-forecast RMSE at H = 12
changed by about 9%. The stored `article_*.csv` files are the ones reported in
the paper. To retrain the whole LSTM grid from scratch, delete the two files
above first.

The notebooks mention `*_exploratory` notebooks. Those belong to the development
history and are not part of this release.

## Data

The input series and the spatial layers are in HydroShare. `scripts/download_data.py`
restores them to their place in this tree and checks their SHA-256 checksums.

* Monthly series and model outputs: https://www.hydroshare.org/resource/4d705def5c5f4fd6b46c29dde302e846/
* Spatial layers of the study area: https://www.hydroshare.org/resource/4141d4f21beb415b8099e967f743f633/

The sources are:

* precipitation and reference evapotranspiration: BR-DWGD (Xavier et al. 2022);
* terrestrial water storage: JPL GRACE/GRACE-FO mascons;
* reservoir operation (storage, inflows, releases, tunnel and transfer flows):
  monthly records of Sabesp (Companhia de Saneamento Básico do Estado de São
  Paulo), extracted from the Sabesp Decision Support System (SSD).

The monthly Sabesp series used in the study are in the HydroShare resource. The
underlying raw SSD records are owned by Sabesp and are not publicly available.
They can be obtained from the authors upon reasonable request and with
permission of Sabesp. Each resource's Readme gives the provenance, processing
and terms of every file.

## How to cite

Please cite the paper (reference to be added on publication) and this software:
see `CITATION.cff` or the Zenodo record of this repository.

## Licence

Code: MIT (`LICENSE`). Model outputs in `Previsao/Outputs/`: CC BY 4.0
(`LICENSE-DATA.md`). Data in HydroShare: see each resource.

## Authors

* Veronica Lima Gonsalez — ORCID [0000-0002-8159-4255](https://orcid.org/0000-0002-8159-4255)
* Raphael Ferreira Perez — ORCID [0000-0001-5674-6571](https://orcid.org/0000-0001-5674-6571)

Contributors: Gerald Augusto Corzo-Pérez ([0000-0002-2773-7817](https://orcid.org/0000-0002-2773-7817)),
Joaquin Ignacio Bonnecarrère ([0000-0002-3852-7444](https://orcid.org/0000-0002-3852-7444)),
Dario Hachisu Hossoda ([0000-0001-7618-2857](https://orcid.org/0000-0001-7618-2857)).

## Acknowledgements

The authors thank the Fundação Centro Tecnológico de Hidráulica (FCTH),
LabSid (Escola Politécnica, Universidade de São Paulo), and Sabesp for the
operational records. AI-assisted development is described in `AI_USE.md`.

## References

Xavier AC, Scanlon BR, King CW, Alves AI (2022) New improved Brazilian daily
weather gridded data (1961–2020). Int J Climatol 42:8390–8404.
https://doi.org/10.1002/joc.7731
