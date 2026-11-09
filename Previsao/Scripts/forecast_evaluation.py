"""Avaliação comparativa de previsões, comum aos braços XGBoost e LSTM.

Hoje contém o teste de Diebold–Mariano usado no artigo para verificar se a
redução de erro do modelo selecionado em relação à persistência é
estatisticamente significativa.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_reference: np.ndarray,
    horizon: int,
    power: int = 2,
    max_lag: int | None = None,
) -> dict[str, float]:
    """Teste de Diebold–Mariano com a correção de Harvey–Leybourne–Newbold.

    Compara a acurácia de duas previsões da mesma série. O diferencial de
    perda é ``d_t = |e_model|^p - |e_ref|^p``; ``d`` médio negativo indica que
    o modelo erra menos que a referência.

    Parameters
    ----------
    y_true : np.ndarray
        Série observada, na ordem temporal.
    pred_model : np.ndarray
        Previsão do modelo avaliado, alinhada a ``y_true``.
    pred_reference : np.ndarray
        Previsão de referência (ex.: persistência), alinhada a ``y_true``.
    horizon : int
        Horizonte H da previsão. Previsões com H > 1 se sobrepõem, então a
        variância de ``d`` inclui autocovariâncias até a defasagem H − 1.
    power : int, default 2
        Expoente da função de perda: 2 = erro quadrático, 1 = erro absoluto.
    max_lag : int, optional
        Defasagem máxima das autocovariâncias. ``None`` usa ``horizon - 1``;
        valores maiores servem ao teste de robustez (Newey–West).

    Returns
    -------
    dict
        ``n`` (tamanho da amostra), ``mean_loss_diff`` (média de ``d``),
        ``statistic`` (estatística corrigida) e ``p_value`` (bilateral).

    Notes
    -----
    A estatística é comparada com uma t de Student com n − 1 graus de
    liberdade (Harvey et al., 1997). Se a variância estimada de ``d`` não for
    positiva (possível com autocovariâncias negativas), a estatística e o
    p-valor retornam ``NaN``.

    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253–263.
    Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. International Journal of Forecasting,
    13(2), 281–291.
    """
    y = np.asarray(y_true, dtype=float)
    d = (np.abs(y - np.asarray(pred_model, dtype=float)) ** power
         - np.abs(y - np.asarray(pred_reference, dtype=float)) ** power)
    if np.isnan(d).any():
        raise ValueError("Séries com NaN: alinhe e filtre antes do teste.")

    n = len(d)
    h = int(horizon)
    lag = h - 1 if max_lag is None else int(max_lag)
    d_bar = d.mean()
    # Autocovariâncias do diferencial de perda, defasagens 0..lag.
    gamma = [np.mean((d[k:] - d_bar) * (d[:n - k] - d_bar))
             for k in range(lag + 1)]
    var_d_bar = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if var_d_bar <= 0:
        return {"n": n, "mean_loss_diff": float(d_bar),
                "statistic": np.nan, "p_value": np.nan}

    dm = d_bar / np.sqrt(var_d_bar)
    # Correção de amostra pequena (Harvey et al., 1997).
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = float(dm * hln)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return {"n": n, "mean_loss_diff": float(d_bar),
            "statistic": statistic, "p_value": p_value}
