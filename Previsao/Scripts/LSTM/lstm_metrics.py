"""
Métricas de avaliação para previsão de volumes (regressão hidrológica).

As métricas aqui implementadas espelham as do braço XGBoost para que
LSTM e XGBoost sejam **diretamente comparáveis**. Adicionamos também:

* **NSE** (Nash–Sutcliffe Efficiency): padrão na literatura hidrológica.
* **Skill direcional**: fração de meses em que o modelo acerta o
  *sentido* da variação ($\\Delta\\text{Vol}$) — útil para diagnosticar
  antecipação de recessões e recuperações, métrica mais sensível a
  transições de regime que RMSE bruto.
* **Métricas por regime**: decompõe RMSE/MAE em estação seca
  (Abr–Set) vs úmida (Out–Mar). Importante porque o XGBoost mostrou
  que CT_CACH_ATIB tem padrões de erro distintos entre regimes.

Por que não usar só RMSE?
-------------------------
RMSE penaliza grandes erros mas é cego ao **sinal**: prever +20 quando
o real é -10 e prever -10 quando o real é +20 têm o mesmo RMSE (30).
Para gestão de recursos hídricos, errar a *direção* (entrar em
recuperação vs continuar em recessão) tem implicações operacionais
muito distintas. O skill direcional captura isso.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Métricas elementares (escalares)
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raiz do erro quadrático médio: $\\sqrt{\\overline{(y - \\hat y)^2}}$."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro absoluto médio: $\\overline{|y - \\hat y|}$."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Correlação de Pearson. Retorna NaN se desvio for zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() < 2:
        return float("nan")
    if np.std(y_true[m]) == 0 or np.std(y_pred[m]) == 0:
        return float("nan")
    return float(np.corrcoef(y_true[m], y_pred[m])[0, 1])


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency.

    Definição clássica:

    .. math::

        \\text{NSE} = 1 - \\frac{\\sum_i (y_i - \\hat y_i)^2}
                            {\\sum_i (y_i - \\bar y)^2}

    Interpretação:

    * ``NSE = 1``: previsão perfeita.
    * ``NSE = 0``: modelo equivale a usar a média histórica como previsão.
    * ``NSE < 0``: modelo é **pior** que prever a média — sinal de problema
      sério.

    NSE é mais sensível a *picos* (variações grandes) do que RMSE relativo,
    e é a métrica de comparação canônica na literatura de modelagem
    hidrológica.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    if m.sum() < 2:
        return float("nan")
    num = np.sum((y_true[m] - y_pred[m]) ** 2)
    den = np.sum((y_true[m] - np.mean(y_true[m])) ** 2)
    if den == 0:
        return float("nan")
    return float(1.0 - num / den)


# --------------------------------------------------------------------------- #
# Skill direcional
# --------------------------------------------------------------------------- #
def directional_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ref: np.ndarray | None = None,
) -> float:
    """Fração de previsões que acertam o **sinal** da variação vs ``ref``.

    Conceito: para um problema de previsão de volume, mais importante que
    a magnitude exata pode ser saber se o volume vai SUBIR ou DESCER. O
    skill direcional mede:

    .. math::

        \\text{DSS} = \\frac{1}{n} \\sum_i \\mathbb{1}\\big[
            \\text{sign}(\\hat y_i - r_i) = \\text{sign}(y_i - r_i)
        \\big]

    onde $r_i$ é uma referência (default: ``y_lag1``, ou seja, valor
    observado anterior — define se o sinal previu corretamente subida ou
    descida em relação ao último valor conhecido).

    Para um classificador aleatório, DSS ≈ 0.5. Valores ``> 0.6`` indicam
    skill direcional não-trivial.

    :param y_true: valores observados.
    :param y_pred: previsões do modelo.
    :param ref: referência. Se ``None``, usa ``np.roll(y_true, 1)`` (lag 1).
    :returns: fração entre 0 e 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if ref is None:
        ref = np.roll(y_true, 1)
        # Descarta o primeiro ponto (não tem lag válido)
        y_true = y_true[1:]
        y_pred = y_pred[1:]
        ref = ref[1:]
    else:
        ref = np.asarray(ref, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(ref))
    if m.sum() == 0:
        return float("nan")
    sgn_true = np.sign(y_true[m] - ref[m])
    sgn_pred = np.sign(y_pred[m] - ref[m])
    # Trata sinal zero como "acerto" se ambos forem zero
    hits = (sgn_true == sgn_pred).astype(float)
    return float(np.mean(hits))


# --------------------------------------------------------------------------- #
# Conjunto agregado
# --------------------------------------------------------------------------- #
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_base: np.ndarray | None = None,
) -> dict:
    """Calcula um conjunto canônico de métricas de regressão.

    :param y_true: observado.
    :param y_pred: predição do modelo.
    :param y_base: predição de baseline (default: persistência). Se
        fornecido, também devolve métricas do baseline para comparação
        direta.

    :returns: dicionário com ``rmse``, ``mae``, ``corr``, ``nse``,
        ``dss`` (skill direcional) — e os mesmos com sufixo ``_base``
        quando ``y_base`` for fornecido.
    """
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "corr": pearson_corr(y_true, y_pred),
        "nse": nse(y_true, y_pred),
        "dss": directional_skill_score(y_true, y_pred),
    }
    if y_base is not None:
        out.update({
            "rmse_base": rmse(y_true, y_base),
            "mae_base": mae(y_true, y_base),
            "corr_base": pearson_corr(y_true, y_base),
            "nse_base": nse(y_true, y_base),
            "dss_base": directional_skill_score(y_true, y_base),
        })
    return out


# --------------------------------------------------------------------------- #
# Métricas por regime (Abr-Set vs Out-Mar)
# --------------------------------------------------------------------------- #
def metrics_by_regime(
    preds_df: pd.DataFrame,
    *,
    target_date_col: str = "target_date",
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    baseline_col: str | None = "y_base",
    split_filter: str | None = "test",
    split_col: str = "split",
) -> pd.DataFrame:
    """Decompõe métricas em estação seca vs úmida.

    A separação Abr-Set (seca) vs Out-Mar (úmida) reflete o ciclo
    hidrológico do Cantareira: meses úmidos recebem maior parte da
    precipitação e contribuem para recuperação de volume; meses secos
    veem recessão dominada por demanda e evapotranspiração.

    :param preds_df: DataFrame no schema padrão (``issue_date``,
        ``target_date``, ``y_true``, ``y_pred``, opcional ``y_base``).
    :param target_date_col: nome da coluna de data alvo.
    :param truth_col, pred_col, baseline_col: nomes das colunas.
    :param split_filter: se fornecido, filtra apenas esse split (default
        ``"test"``).
    :param split_col: nome da coluna de split.

    :returns: DataFrame com uma linha por regime (``wet`` / ``dry``) e
        colunas de métricas.
    """
    df = preds_df.copy()
    df[target_date_col] = pd.to_datetime(df[target_date_col])
    if split_filter is not None:
        df = df[df[split_col] == split_filter]
    df["regime"] = np.where(
        df[target_date_col].dt.month.isin([4, 5, 6, 7, 8, 9]),
        "dry", "wet",
    )

    rows = []
    for regime, sub in df.groupby("regime"):
        y_t = sub[truth_col].to_numpy()
        y_p = sub[pred_col].to_numpy()
        y_b = (
            sub[baseline_col].to_numpy()
            if baseline_col and baseline_col in sub.columns
            else None
        )
        m = compute_regression_metrics(y_t, y_p, y_base=y_b)
        m["regime"] = regime
        m["n"] = len(sub)
        rows.append(m)

    cols_order = ["regime", "n", "rmse", "mae", "corr", "nse", "dss"]
    if baseline_col:
        cols_order += ["rmse_base", "mae_base", "corr_base",
                       "nse_base", "dss_base"]
    return pd.DataFrame(rows)[cols_order]


# --------------------------------------------------------------------------- #
# Métricas probabilísticas (Regressão Quantílica)
# --------------------------------------------------------------------------- #
def coverage_rate(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Fração das observações dentro do intervalo [y_lower, y_upper].

    Para um intervalo P10-P90 (80 % nominal), o alvo teórico é ≥ 0.80.
    Uma cobertura abaixo de 0.75 indica que o intervalo é demasiado
    estreito (sub-cobertura). Acima de 0.95 indica intervalo muito
    conservador (super-cobertura, pouca utilidade operacional).

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_lower : np.ndarray
        Limite inferior do intervalo (e.g. P10).
    y_upper : np.ndarray
        Limite superior do intervalo (e.g. P90).

    Returns
    -------
    float
        Fração entre 0 e 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if m.sum() == 0:
        return float("nan")
    inside = (y_true[m] >= y_lower[m]) & (y_true[m] <= y_upper[m])
    return float(np.mean(inside))


def mean_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Largura média do intervalo de previsão (*sharpness*).

    Mede quão informativo o intervalo é: larguras menores indicam
    maior precisão condicional. Deve ser interpretada em conjunto com
    :func:`coverage_rate` — um intervalo estreito mas com sub-cobertura
    é inútil.

    Parameters
    ----------
    y_lower : np.ndarray
        Limite inferior (e.g. P10), em hm³.
    y_upper : np.ndarray
        Limite superior (e.g. P90), em hm³.

    Returns
    -------
    float
        Largura média em hm³.
    """
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    m = ~(np.isnan(y_lower) | np.isnan(y_upper))
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(y_upper[m] - y_lower[m]))


def winkler_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """Winkler score para o intervalo de $(1-\\alpha)\\times 100\\%$ de confiança.

    O Winkler score penaliza simultaneamente:
    (a) intervalos largos (penalidade de largura) e
    (b) observações fora do intervalo (penalidade adicional proporcional
    à distância de violação).

    .. math::

        S_i =
        \\begin{cases}
            (U_i - L_i) + \\frac{2}{\\alpha}(L_i - y_i)
                & \\text{se } y_i < L_i \\\\
            (U_i - L_i)
                & \\text{se } L_i \\leq y_i \\leq U_i \\\\
            (U_i - L_i) + \\frac{2}{\\alpha}(y_i - U_i)
                & \\text{se } y_i > U_i
        \\end{cases}

    Menor score = melhor (intervalo mais preciso e com boa cobertura).

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_lower : np.ndarray
        Limite inferior (e.g. P10).
    y_upper : np.ndarray
        Limite superior (e.g. P90).
    alpha : float
        ``1 − cobertura nominal``. Para intervalo P10-P90: ``alpha=0.20``
        (80 % nominal).

    Returns
    -------
    float
        Winkler score médio (mesma unidade que ``y_true``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if m.sum() == 0:
        return float("nan")
    yt, yl, yu = y_true[m], y_lower[m], y_upper[m]
    width = yu - yl
    penalty = np.where(
        yt < yl,
        (2.0 / alpha) * (yl - yt),
        np.where(yt > yu, (2.0 / alpha) * (yt - yu), 0.0),
    )
    return float(np.mean(width + penalty))
