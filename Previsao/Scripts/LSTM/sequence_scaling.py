"""
Normalização (scaling) de janelas 3-D **sem vazamento temporal**.

Por que escalar?
----------------
Redes neurais ajustadas via gradient descent treinam de forma muito mais
estável quando as features estão em escalas comparáveis. Para o Cantareira,
features hidroclimáticas podem variar em ordens de magnitude muito distintas:

* ``Vol`` em hm³ → centenas a milhares.
* ``q`` em m³/s → unidades a dezenas.
* ``pr`` em mm/mês → dezenas a centenas.
* ``month_sin``, ``month_cos`` → [-1, 1].

Sem padronização, gradientes seriam dominados pela feature de maior escala,
e a otimização poderia divergir ou progredir excessivamente devagar.

Por que ajustar SOMENTE no treino?
----------------------------------
Estatísticas (média, desvio) calculadas em todo o histórico — incluindo
val/test — vazam informação futura para o modelo, inflando artificialmente
as métricas de avaliação. A regra inegociável é:

* ``fit_sequence_scaler`` recebe APENAS ``X_train`` e ``y_train``.
* ``apply_sequence_scaler`` reaproveita o mesmo scaler em ``X_val``,
  ``X_test``, ``y_val``, ``y_test``.

Reshape 3-D → 2-D → 3-D
-----------------------
``StandardScaler`` do sklearn opera em matrizes 2-D ``(n, p)``. Nossas
janelas são 3-D ``(n, L, p)``. A solução é colapsar os dois primeiros
eixos para o fit e o transform, e remontar o shape original:

.. code-block:: text

   X.shape = (n_samples, seq_len, n_features)
              │            │         └── eixo "feature" (p)
              │            └── eixo "tempo dentro da janela" (L)
              └── eixo "amostra" (n)

   reshape → (n_samples * seq_len, n_features)   ← fit/transform
   reshape ← (n_samples, seq_len, n_features)

Cada passo temporal da janela é tratado como uma observação independente
da MESMA distribuição da feature — o que é coerente porque a janela é
apenas uma reorganização do histórico contínuo daquela bacia.

Escala do target
----------------
Em ``target_mode="delta"`` o alvo tem média próxima de zero e desvio em
hm³ — escala bem comportada. Em ``target_mode="level"`` o alvo é volume
absoluto (centenas a milhares de hm³). Escalar ``y`` (opcional) ajuda a
manter a perda em magnitudes próximas das ativações ``tanh``, evitando
saturação. A função ``inverse_transform_y`` devolve as predições à
escala física para cálculo de RMSE em unidades reais.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Container para o par (feature_scaler, target_scaler)
# --------------------------------------------------------------------------- #
@dataclass
class SequenceScaler:
    """Container imutável para os scalers ajustados no treino.

    :ivar feature_scaler: scaler para a matriz de features (eixo ``p``).
        Sempre presente.
    :ivar target_scaler: scaler para o alvo ``y``. Pode ser ``None`` se
        ``scale_y=False`` no fit (e.g. quando o alvo já está bem
        comportado em ``delta``).
    :ivar feature_names: lista das colunas que entraram no scaler — usada
        para sanity-check em ``apply_sequence_scaler``.
    """

    feature_scaler: StandardScaler
    target_scaler: StandardScaler | None
    feature_names: list[str]


# --------------------------------------------------------------------------- #
# Ajuste do scaler — usa APENAS o split de treino
# --------------------------------------------------------------------------- #
def fit_sequence_scaler(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    scale_y: bool = True,
) -> SequenceScaler:
    """Ajusta scalers de features e (opcionalmente) target no conjunto de treino.

    :param X_train: tensor de janelas de treino, shape ``(n_train, L, p)``.
    :param y_train: alvo de treino, shape ``(n_train,)``.
    :param feature_names: nomes das ``p`` features na ordem do eixo
        de feature. Opcional — usado apenas para anotar o scaler.
    :param scale_y: se ``True``, ajusta também um scaler para ``y``.
        Recomendado para ``target_mode="level"``. Para ``"delta"``,
        pode ser ``False`` (alvo já tem média ≈ 0 e desvio pequeno).
    :returns: instância de :class:`SequenceScaler` pronta para ser
        aplicada em todos os splits.

    :raises ValueError: se ``X_train`` não for 3-D.
    """
    if X_train.ndim != 3:
        raise ValueError(
            f"X_train deve ter shape (n, L, p); recebido {X_train.shape}."
        )

    n, L, p = X_train.shape

    # 3-D → 2-D para o sklearn
    feat_scaler = StandardScaler()
    feat_scaler.fit(X_train.reshape(n * L, p))

    tgt_scaler: StandardScaler | None = None
    if scale_y:
        tgt_scaler = StandardScaler()
        tgt_scaler.fit(y_train.reshape(-1, 1))

    return SequenceScaler(
        feature_scaler=feat_scaler,
        target_scaler=tgt_scaler,
        feature_names=list(feature_names) if feature_names else [],
    )


# --------------------------------------------------------------------------- #
# Aplicação do scaler a um split arbitrário (train, val ou test)
# --------------------------------------------------------------------------- #
def apply_sequence_scaler(
    scaler: SequenceScaler,
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Aplica os scalers ajustados, preservando o shape 3-D de ``X``.

    :param scaler: container retornado por :func:`fit_sequence_scaler`.
    :param X: tensor de janelas, shape ``(n, L, p)``. ``p`` deve casar
        com o número de features vistas no fit.
    :param y: alvo opcional, shape ``(n,)``. Se ``None`` ou se o scaler
        de target for ``None``, o segundo elemento do retorno é ``None``.

    :returns: tupla ``(X_scaled, y_scaled)``.

    :raises ValueError: se ``X`` não for 3-D ou tiver número de features
        incompatível com o scaler.
    """
    if X.ndim != 3:
        raise ValueError(
            f"X deve ter shape (n, L, p); recebido {X.shape}."
        )

    n, L, p = X.shape
    expected_p = scaler.feature_scaler.n_features_in_
    if p != expected_p:
        raise ValueError(
            f"Número de features incompatível: scaler espera {expected_p}, "
            f"recebido {p}."
        )

    X2 = X.reshape(n * L, p)
    X_scaled = scaler.feature_scaler.transform(X2).reshape(n, L, p)
    X_scaled = X_scaled.astype(np.float32)

    y_scaled: np.ndarray | None = None
    if y is not None and scaler.target_scaler is not None:
        y_scaled = scaler.target_scaler.transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)
    elif y is not None:
        # Scaler de y desabilitado — devolve y inalterado (apenas cast)
        y_scaled = y.astype(np.float32)

    return X_scaled, y_scaled


# --------------------------------------------------------------------------- #
# Inverso do scaler de y — converte previsões de volta à escala física
# --------------------------------------------------------------------------- #
def inverse_transform_y(
    scaler: SequenceScaler,
    y_scaled: np.ndarray,
) -> np.ndarray:
    """Converte ``y_pred`` da escala normalizada para a escala física.

    Operação obrigatória antes de calcular RMSE em unidades reais (hm³) e
    antes de reconstruir o volume a partir de previsões em modo ``delta``
    (``Vol_pred = Vol_t + ΔVol_pred``).

    :param scaler: o mesmo container usado em :func:`apply_sequence_scaler`.
    :param y_scaled: predições do modelo, shape ``(n,)`` ou ``(n, 1)``.
    :returns: predições na escala original, shape ``(n,)``.

    :raises RuntimeError: se ``scaler.target_scaler is None`` — nesse caso,
        ``y`` nunca foi escalado e o "inverso" seria identidade. Levantamos
        o erro para evitar bug silencioso onde o usuário esperava ter
        escalado.
    """
    if scaler.target_scaler is None:
        raise RuntimeError(
            "scaler.target_scaler é None — nenhum scaling foi aplicado em y. "
            "Use diretamente y_scaled (já está na escala original)."
        )
    y_arr = np.asarray(y_scaled).reshape(-1, 1)
    return scaler.target_scaler.inverse_transform(y_arr).ravel()
