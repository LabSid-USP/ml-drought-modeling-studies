"""
Arquiteturas neurais para previsão sequencial de volume.

Três modelos coexistem aqui para permitir comparações sistemáticas:

* :class:`LSTMRegressor` — recorrente com 3 gates (forget, input, output)
  e estado de célula explícito; principal modelo do braço.
* :class:`GRURegressor` — recorrente "primo" da LSTM com apenas 2 gates
  (reset, update). Tem menos parâmetros e treina mais rápido; serve como
  *ablation* para testar se a complexidade extra da LSTM compensa
  estatisticamente para o tamanho da nossa base.
* :class:`MLPWindowRegressor` — **não-recorrente**: achata a janela em um
  vetor e passa por camadas densas. Serve como teste se a recorrência
  agrega valor sobre uma rede densa equivalente. Se MLP ≈ LSTM em RMSE,
  a "memória recorrente" pode não estar capturando dependências
  temporais que uma rede densa não capturaria via pesos diretos.

Decisões de design comuns
-------------------------
* **``batch_first=True``**: tensor de entrada na forma natural
  ``(batch, seq_len, features)``. Convenção mais legível que
  ``(seq_len, batch, features)`` da API original PyTorch.
* **Head pequeno** (``Linear → ReLU → Linear``): mantém o número de
  parâmetros baixo. Para ~480 amostras mensais, modelos com milhares
  de parâmetros overfit rapidamente.
* **Sem bidirectional**: para forecasting *online*, o modelo só pode
  usar o passado. Uma BiLSTM olharia ``[t+1, t+2, ...]`` dentro da
  janela atual, vazando informação que não estaria disponível em
  produção. **Anti-padrão proibido neste módulo**.
* **``dropout`` entre camadas LSTM**: só atua quando ``num_layers > 1``,
  por design do PyTorch.

Sobre o número de parâmetros
----------------------------
Uma LSTM com ``hidden_size=h`` e ``input_size=p`` tem
``4 * (p*h + h*h + h)`` parâmetros na camada recorrente. Para
``p=7, h=16`` isso dá ``4 * (7*16 + 16*16 + 16) = 4 * 384 = 1536``
parâmetros. Com ~340 amostras de treino, a razão dados/parâmetros é
``~0.2`` — já pede regularização (early stopping, weight decay) para
evitar overfit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    """LSTM pequena + head densa para regressão univariada.

    Arquitetura:

    .. code-block:: text

        Input (batch, seq_len, input_size)
              │
              ▼
        ┌──────────────┐
        │   nn.LSTM    │  num_layers x (hidden_size canais)
        │ batch_first  │  + dropout entre camadas se num_layers > 1
        └──────┬───────┘
               │  out (batch, seq_len, hidden_size)
               ▼
        Pega último timestep:  out[:, -1, :]  shape (batch, hidden_size)
               │
               ▼
        ┌──────────────────────┐
        │   Linear(h, hh)      │  hh = head_hidden
        │       ReLU           │
        │ Linear(hh, n_out)    │  n_out = head_output_size
        └──────────────────────┘
               │
               ▼
            ŷ (batch, n_out)   # n_out=1 → regressão pontual
                               # n_out=3 → P10/P50/P90 (quantílica)

    :param input_size: número de features por timestep (eixo ``p``).
    :param hidden_size: tamanho do estado oculto $h_t$ e da célula $C_t$.
        Recomendado 8–32 para a nossa base mensal (~480 amostras).
    :param num_layers: número de camadas LSTM empilhadas. Comece com 1.
    :param dropout: taxa de dropout aplicada ENTRE camadas LSTM
        (não dentro da mesma camada). Só faz efeito se ``num_layers > 1``.
    :param head_hidden: largura da camada densa intermediária do head.
    :param head_output_size: número de saídas do head. 1 (padrão) para
        regressão pontual; 3 para regressão quantílica (P10/P50/P90).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden: int = 8,
        head_output_size: int = 1,
    ) -> None:
        super().__init__()

        if dropout > 0.0 and num_layers == 1:
            # Avisa silenciosamente que dropout vai ser ignorado
            # (não há "entre camadas" com apenas uma).
            dropout_effective = 0.0
        else:
            dropout_effective = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_effective,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: tensor ``(batch, seq_len, input_size)``.
        :returns: predições ``(batch, head_output_size)``.
        """
        # nn.LSTM devolve (output_all_timesteps, (h_n, c_n))
        # output[:, -1, :] = estado oculto do último timestep da janela
        # que é o que carrega o resumo sequencial completo
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# --------------------------------------------------------------------------- #
# GRU — alternativa mais simples
# --------------------------------------------------------------------------- #
class GRURegressor(nn.Module):
    """GRU + head denso. API idêntica à :class:`LSTMRegressor`.

    GRU (Gated Recurrent Unit) usa apenas 2 gates (reset, update) e
    funde estado oculto e estado de célula em um único $h_t$. Tem ~25 %
    menos parâmetros que a LSTM equivalente e treina mais rápido.
    Útil quando se suspeita que a complexidade extra da LSTM não está
    pagando bem para a quantidade de dados disponíveis.

    :param input_size: número de features.
    :param hidden_size: tamanho de $h_t$.
    :param num_layers: empilhamento de camadas GRU.
    :param dropout: dropout entre camadas.
    :param head_hidden: largura do head denso.
    :param head_output_size: número de saídas. 1 para regressão pontual;
        3 para quantílica (P10/P50/P90). Veja :class:`LSTMRegressor`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden: int = 8,
        head_output_size: int = 1,
    ) -> None:
        super().__init__()

        dropout_effective = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_effective,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# --------------------------------------------------------------------------- #
# MLP com janela achatada — baseline não-recorrente
# --------------------------------------------------------------------------- #
class MLPWindowRegressor(nn.Module):
    """MLP densa sobre janela achatada.

    Recebe ``(batch, seq_len, input_size)``, achata para
    ``(batch, seq_len * input_size)`` e passa por uma MLP. **Não modela
    estrutura sequencial** — todos os timesteps entram como features
    independentes, no mesmo nível.

    Função: serve como baseline para responder à pergunta:

        *"A recorrência explícita (LSTM/GRU) agrega valor sobre uma rede
        densa equivalente que enxerga os mesmos dados?"*

    Se MLP ≈ LSTM em RMSE, a evidência empírica é que, para esta base,
    a rede dispensa o viés indutivo recorrente.

    :param input_size: número de features por timestep.
    :param seq_len: comprimento da janela (necessário para definir o
        tamanho da camada de entrada após o achatamento).
    :param hidden_sizes: lista de larguras das camadas ocultas.
    :param dropout: taxa de dropout aplicada após cada ativação ReLU.
    :param head_output_size: número de saídas da camada final. 1 para
        regressão pontual; 3 para quantílica (P10/P50/P90).
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        head_output_size: int = 1,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 16]

        self.input_size = input_size
        self.seq_len = seq_len
        in_features = input_size * seq_len

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, head_output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Achata (batch, L, p) -> (batch, L*p)
        flat = x.reshape(x.size(0), -1)
        return self.net(flat)
