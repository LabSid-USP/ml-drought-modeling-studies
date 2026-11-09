# Methodological rules

These rules govern every change to the forecasting code. The notebooks cite
them by section number (for example "§3.1"). The numbering is kept from the
authors' internal guidelines, so it starts at 3.

## 3. Rules

**3.1 Temporal split.** Training, validation and test sets are always split on
the **target date** (`split_on="target_date"`), never on the issue date.
Splitting on the issue date would leak future information into training.

**3.2 Minimum required predictors.** The core set is precipitation (`pr`),
reference evapotranspiration (`eto`), natural inflow (`q`), storage (`Vol`) and
its first lag, plus the calendar encodings. Terrestrial water storage (`tws`) is
**not** part of it. The GRACE series starts in 2002, and requiring it would
drop 18 years of record.

**3.3 Raw volume outside the model matrix.** `Vol_raw` (and `Vol_LagN_raw`) are
only used to rebuild the storage level after a change (`delta`) forecast. They
never enter the model matrix.

**3.4 Target mode.** The model forecasts either the storage level (`level`) or
its change over the horizon (`delta`). For the whole system (`CT_TOTAL`) the
mode is chosen empirically by ablation. The paper reports the change relative to
persistence (A–D) or to seasonal climatology (A*–D*).

**3.5 Mandatory baselines.** Every model metric is compared with persistence,
ŷ(t+H) = Vol(t), and with seasonal climatology. When a model does not beat
persistence in RMSE/MAE, this is stated explicitly.

**3.6 Non-causal interpretation.** SHAP values, PDP/ICE curves, saliency and
perturbation results are interpretive, not causal. Results are described with
wording such as "indicates", "acts as a proxy for" or "is associated with".

**3.7 Drainage areas.** Area-weighted aggregation (precipitation,
evapotranspiration, terrestrial water storage) uses the incremental drainage
areas, in km²: Jaguari/Jacareí 1239.25, Cachoeira 392.38, Atibainha 315.07 and
Paiva Castro 337.57. They are defined once in
`Previsao/Scripts/XGBoost/cantareira_equivalent.py` (`CANTAREIRA_AREA_KM2`).
