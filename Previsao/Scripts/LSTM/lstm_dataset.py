"""
Wrapper :class:`torch.utils.data.Dataset` para amostras sequenciais.

PyTorch trabalha com :class:`~torch.Tensor` em vez de ``np.ndarray`` no loop
de treinamento. O :class:`Dataset` define **como obter uma amostra dado um
índice**, e o :class:`~torch.utils.data.DataLoader` cuida do *batching*,
embaralhamento (quando seguro) e prefetch.

Por que essa abstração?
-----------------------
* **Lazy materialization**: o ``DataLoader`` chama ``__getitem__`` em paralelo
  conforme o treino consome lotes — útil para datasets grandes que não
  caberiam em memória GPU. No nosso caso (480 meses × poucas features),
  carregamos tudo em memória, mas o padrão é o mesmo.
* **Shape contrato**: cada ``__getitem__`` devolve ``(X_i, y_i)`` com
  ``X_i`` de shape ``(seq_len, n_features)`` e ``y_i`` de shape ``(1,)``.
  O ``DataLoader`` empilha em ``(batch_size, seq_len, n_features)`` —
  exatamente o que a ``nn.LSTM`` (com ``batch_first=True``) espera.
* **Cast para float32**: redes ``torch.nn`` usam ``float32`` por padrão.
  Convertemos uma vez no construtor para evitar conversão em cada batch.

Sobre embaralhamento
--------------------
``DataLoader(shuffle=True)`` é seguro **dentro de um split** — embaralhar
amostras de treino não causa vazamento porque cada janela já está
identificada por sua ``issue_date``. Não embaralhamos entre splits (isso
seria vazamento).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """``torch.utils.data.Dataset`` para janelas LSTM.

    :param X: tensor de janelas, shape ``(n, seq_len, n_features)``,
        tipicamente já escalado por :func:`apply_sequence_scaler`.
    :param y: alvo, shape ``(n,)``.

    Exemplos
    --------
    .. code-block:: python

        from torch.utils.data import DataLoader

        ds_train = SequenceDataset(X_train_scaled, y_train_scaled)
        loader = DataLoader(ds_train, batch_size=16, shuffle=True)
        for Xb, yb in loader:
            # Xb.shape == (16, seq_len, n_features)
            # yb.shape == (16, 1)
            ...
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"X deve ter 3 dimensões (n, L, p); recebido shape={X.shape}."
            )
        if len(X) != len(y):
            raise ValueError(
                f"X e y devem ter o mesmo n: len(X)={len(X)}, len(y)={len(y)}."
            )

        # Conversão única para float32 — economiza tempo no loop de treino
        self.X = torch.as_tensor(X, dtype=torch.float32)
        # y vira coluna (n, 1) para casar com a saída do head Linear(1)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
