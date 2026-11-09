"""
Construção de amostras supervisionadas sequenciais para LSTM/GRU/MLP-window.

Este módulo é o **portal de entrada** do pipeline LSTM: ele transforma o
``datasets_by_basin`` tabular (mesmo formato produzido por
:class:`Previsao.Scripts.XGBoost.feature_constructor.FeatureConstructor`)
em **janelas temporais 3-D** prontas para redes recorrentes.

Diferença de paradigma com o XGBoost
------------------------------------
No braço XGBoost, cada mês $t$ vira uma linha:

.. math::

   X_\\text{tab}(t) = [\\text{pr}(t), \\text{eto}(t), \\text{Vol}(t),
                       \\text{Vol\\_Lag1}(t), \\text{Vol\\_Lag3}(t), \\ldots]

Os lags são features explícitas pré-selecionadas. A LSTM dispensa lags
discretos: a memória vem da própria recorrência. Por isso, cada amostra é
uma **janela de** ``seq_len`` **meses consecutivos**:

.. math::

   X_\\text{seq}(t) =
   \\begin{bmatrix}
   \\text{pr}(t-L+1) & \\text{eto}(t-L+1) & \\ldots \\\\
   \\vdots & \\vdots & \\\\
   \\text{pr}(t)     & \\text{eto}(t)     & \\ldots
   \\end{bmatrix}
   \\in \\mathbb{R}^{L \\times p}

onde $L = $ ``seq_len`` e $p = $ ``len(feature_cols)``.

O alvo permanece definido pelo horizonte $H$:

* ``target_mode="level"``: $y = \\text{Vol}(t+H)$.
* ``target_mode="delta"``: $y = \\text{Vol}(t+H) - \\text{Vol}(t)$.

Protocolo de validação
----------------------
O *split* temporal segue **exatamente** o do XGBoost (sem vazamento):

* ``split_on="target_date"``: a amostra cai em treino/val/teste pela data
  ALVO ($t+H$), não pela data de emissão ($t$).
* Sem embaralhamento entre splits.

Garantias estruturais
---------------------
* ``X_seq`` tem shape ``(n_samples, seq_len, n_features)``.
* ``y`` tem shape ``(n_samples,)``.
* ``meta`` é um ``pd.DataFrame`` com colunas:

  - ``issue_date``: data $t$ do último mês da janela de entrada.
  - ``target_date``: data $t+H$ do alvo.
  - ``Vol_t``: volume observado em $t$ (necessário para reconstruir
    previsões em modo ``delta``).
  - ``y_true_level``: volume observado em $t+H$ na escala física
    (referência para diagnóstico independente do modo).
  - ``split``: ``"train"`` / ``"val"`` / ``"test"``.

Notas
-----
Amostras com janela incompleta (início da série) ou com valores faltantes em
``core_required`` são silenciosamente descartadas. Isso preserva o histórico
máximo dos splits a jusante, ao custo de poucos meses iniciais.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Container de metadados + máscaras de split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequenceSplit:
    """Máscaras booleanas + metadados por amostra sequencial.

    :ivar train: máscara booleana indicando quais amostras pertencem ao
        conjunto de treino. Shape ``(n_samples,)``.
    :ivar val: idem para validação (ou ``None`` se ``split_mode="2way"``).
    :ivar test: máscara booleana de teste.
    :ivar meta: DataFrame com uma linha por amostra contendo
        ``issue_date``, ``target_date``, ``Vol_t``, ``y_true_level`` e
        ``split``. Útil para reconstruir previsões e gerar diagnósticos
        compatíveis com o schema do XGBoost.
    """

    train: np.ndarray
    val: np.ndarray | None
    test: np.ndarray
    meta: pd.DataFrame


# --------------------------------------------------------------------------- #
# Helper interno: mescla Vol(t) e cria target Vol(t+H)
# --------------------------------------------------------------------------- #
def _merge_volume_and_target(
    df_features: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    basin: str,
    horizon: int,
    target_mode: Literal["level", "delta"],
    date_col: str = "date",
) -> pd.DataFrame:
    """Anexa ``Vol(t)`` e calcula ``target_date`` / ``y_true_level`` / ``y``.

    Reproduz o mesmo padrão de :meth:`XGBoostModel.build_supervised`, mas
    sem criar ``Vol_LagN`` (não fazem sentido para LSTM — a janela já
    contém a memória).

    :returns: DataFrame com colunas extras
        ``Vol_t``, ``target_date``, ``y_true_level``, ``y``.
    """
    df = df_features.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    vol = volumes[[date_col, basin]].copy()
    vol[date_col] = pd.to_datetime(vol[date_col])
    vol = vol.rename(columns={basin: "Vol_t"}).sort_values(date_col)

    # Merge do volume no tempo atual t
    df = df.merge(vol, on=date_col, how="left")

    # Volume no tempo alvo t+H — desloca H linhas para cima (assume grade
    # mensal regular, garantida pelo FeatureConstructor)
    df["y_true_level"] = df["Vol_t"].shift(-horizon)
    df["target_date"] = df[date_col].shift(-horizon)

    if target_mode == "level":
        df["y"] = df["y_true_level"]
    elif target_mode == "delta":
        df["y"] = df["y_true_level"] - df["Vol_t"]
    else:
        raise ValueError(
            f"target_mode inválido: {target_mode!r}. "
            "Use 'level' ou 'delta'."
        )
    return df


# --------------------------------------------------------------------------- #
# Função pública principal
# --------------------------------------------------------------------------- #
def build_sequence_supervised(
    datasets_by_basin: dict[str, pd.DataFrame],
    volumes: pd.DataFrame,
    *,
    basin: str,
    feature_cols: Sequence[str],
    horizon: int,
    seq_len: int,
    target_mode: Literal["level", "delta"],
    min_date: str | pd.Timestamp,
    max_date: str | pd.Timestamp,
    split_on: Literal["issue_date", "target_date"] = "target_date",
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp | None = None,
    core_required: Sequence[str] | None = None,
    date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray, SequenceSplit]:
    """Constrói amostras supervisionadas em janela 3-D para LSTM/GRU/MLP.

    :param datasets_by_basin: dicionário produzido por
        :meth:`FeatureConstructor.build` (chave = código da bacia, valor =
        DataFrame longo com ``date`` + colunas de features).
    :param volumes: DataFrame *wide* com ``date`` + uma coluna por bacia
        (o mesmo passado ao XGBoost).
    :param basin: código da bacia-alvo (e.g. ``"CT_CACH_ATIB"``).
    :param feature_cols: lista das colunas que vão compor a feature dimension
        (eixo ``p`` do tensor). NÃO inclua lags discretos: a janela já
        carrega a memória temporal.
    :param horizon: horizonte de previsão em meses, $H$.
    :param seq_len: comprimento $L$ da janela de entrada (em meses).
    :param target_mode: ``"level"`` para prever $\\text{Vol}(t+H)$ direto;
        ``"delta"`` para prever $\\Delta\\text{Vol}_H = \\text{Vol}(t+H) -
        \\text{Vol}(t)$ — útil quando o sistema tem alta inércia (a
        previsão de variação é mais estável).
    :param min_date: data mínima admitida em ``date_col``. Linhas anteriores
        são descartadas antes da construção da janela.
    :param max_date: data máxima admitida em ``date_col`` (na *issue date*).
    :param split_on: se ``"target_date"``, a amostra cai em treino/val/teste
        pela data do alvo (recomendado, evita vazamento). Se
        ``"issue_date"``, pela data de emissão.
    :param train_end: limite superior (inclusivo) do split de treino.
    :param val_end: limite superior do split de validação. Se ``None``, a
        validação é desabilitada (split 2-way).
    :param core_required: lista de colunas que NÃO podem ter NaN dentro da
        janela. Default sensato:
        ``["pr", "eto", "q", "Vol_t"]``.
    :param date_col: nome da coluna de datas em ``datasets_by_basin[basin]``.

    :returns: tupla ``(X_seq, y, split)`` onde:

        * ``X_seq``: ``np.ndarray`` shape ``(n_samples, seq_len, len(feature_cols))``,
          dtype ``float32``.
        * ``y``: ``np.ndarray`` shape ``(n_samples,)``, dtype ``float32``.
        * ``split``: instância de :class:`SequenceSplit`.

    :raises KeyError: se ``basin`` ou ``feature_cols`` não existirem.
    :raises ValueError: se nenhuma amostra sobreviver aos filtros (janela,
        ``core_required``, recorte temporal).
    """
    if basin not in datasets_by_basin:
        raise KeyError(
            f"Bacia {basin!r} ausente em datasets_by_basin. "
            f"Disponíveis: {sorted(datasets_by_basin)}"
        )

    df_features = datasets_by_basin[basin]

    # Colunas INJETADAS pelo construtor — passam a existir somente após
    # _merge_volume_and_target. Não devem ser exigidas no DataFrame
    # original (pois ainda não estão lá).
    _INJECTED = {"Vol_t", "y", "y_true_level", "target_date"}

    # Validação preliminar: somente colunas que o usuário deveria ter
    # fornecido em datasets_by_basin (i.e., não injetadas).
    needed_from_user = [c for c in feature_cols if c not in _INJECTED]
    missing_user = [c for c in needed_from_user
                    if c not in df_features.columns]
    if missing_user:
        raise KeyError(
            f"Features ausentes em datasets_by_basin[{basin!r}]: "
            f"{missing_user}. "
            f"Disponíveis: {sorted(df_features.columns)}"
        )

    if core_required is None:
        core_required = ["pr", "eto", "q", "Vol_t"]
    core_required = list(core_required)

    if seq_len < 1:
        raise ValueError(f"seq_len deve ser >= 1, recebido {seq_len}.")
    if horizon < 1:
        raise ValueError(f"horizon deve ser >= 1, recebido {horizon}.")

    # 1) Mescla Vol(t) e calcula target Vol(t+H) / ΔVol_H
    df = _merge_volume_and_target(
        df_features, volumes,
        basin=basin, horizon=horizon, target_mode=target_mode,
        date_col=date_col,
    )

    # Validação completa pós-merge: agora todas as feature_cols
    # (inclusive Vol_t) devem existir no DataFrame enriquecido.
    missing_post = [c for c in feature_cols if c not in df.columns]
    if missing_post:
        raise KeyError(
            f"Features ausentes após merge: {missing_post}. "
            f"Colunas disponíveis: {sorted(df.columns)}"
        )

    # 2) Recorte temporal (issue_date dentro de [min_date, max_date])
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    df = df[(df[date_col] >= min_date) & (df[date_col] <= max_date)]
    df = df.reset_index(drop=True)

    # Sanity: core_required deve incluir "Vol_t" para descartar bordas
    # finais sem Vol(t+H) calculado (shift(-H) introduz NaN nas últimas H
    # linhas). Adicionamos automaticamente "y" e "y_true_level".
    required_cols = set(core_required) | {"y", "y_true_level", "Vol_t"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Colunas necessárias ausentes após merge: {sorted(missing_cols)}"
        )

    # 3) Construção das janelas
    feature_mat = df[list(feature_cols)].to_numpy(dtype=np.float64)
    n_rows = len(df)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    issue_dates: list[pd.Timestamp] = []
    target_dates: list[pd.Timestamp] = []
    vol_t_list: list[float] = []
    y_level_list: list[float] = []

    for i in range(seq_len - 1, n_rows):
        # janela = linhas [i - seq_len + 1, ..., i]  → seq_len meses
        window = feature_mat[i - seq_len + 1: i + 1, :]

        # Filtros de validade da amostra
        if np.isnan(window).any():
            continue
        # Os campos críticos no instante i devem existir
        if any(pd.isna(df.iloc[i][c]) for c in required_cols):
            continue

        X_list.append(window)
        y_list.append(float(df.iloc[i]["y"]))
        issue_dates.append(df.iloc[i][date_col])
        target_dates.append(df.iloc[i]["target_date"])
        vol_t_list.append(float(df.iloc[i]["Vol_t"]))
        y_level_list.append(float(df.iloc[i]["y_true_level"]))

    if not X_list:
        raise ValueError(
            "Nenhuma amostra sobreviveu aos filtros (seq_len, core_required, "
            "recorte temporal). Reduza seq_len ou afrouxe core_required."
        )

    X_seq = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    meta = pd.DataFrame({
        "issue_date": issue_dates,
        "target_date": target_dates,
        "Vol_t": vol_t_list,
        "y_true_level": y_level_list,
    })

    # 4) Máscaras de split (por target_date ou issue_date)
    if split_on not in ("target_date", "issue_date"):
        raise ValueError(
            f"split_on inválido: {split_on!r}. "
            "Use 'target_date' ou 'issue_date'."
        )
    split_key = pd.to_datetime(meta[split_on])
    train_end = pd.to_datetime(train_end)

    if val_end is not None:
        val_end = pd.to_datetime(val_end)
        train_mask = (split_key <= train_end).to_numpy()
        val_mask = (
            (split_key > train_end) & (split_key <= val_end)
        ).to_numpy()
        test_mask = (split_key > val_end).to_numpy()
    else:
        train_mask = (split_key <= train_end).to_numpy()
        val_mask = None
        test_mask = (split_key > train_end).to_numpy()

    # Anota split em meta para uso a jusante (plots, métricas por regime)
    meta = meta.copy()
    meta["split"] = "test"
    meta.loc[train_mask, "split"] = "train"
    if val_mask is not None:
        meta.loc[val_mask, "split"] = "val"

    split = SequenceSplit(
        train=train_mask, val=val_mask, test=test_mask, meta=meta,
    )

    return X_seq, y, split
