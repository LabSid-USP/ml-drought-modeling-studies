"""
Loop de treino para modelos sequenciais (LSTM/GRU/MLP).

O treinamento de uma rede neural é um processo iterativo:

1. Para cada **época** (passada completa pelo conjunto de treino):

   a. Particionar treino em **lotes** (batches) de tamanho fixo.
   b. Para cada lote:

      i.   *Forward*: rodar o modelo nas entradas para obter ``y_pred``.
      ii.  *Loss*: calcular o MSE entre ``y_pred`` e ``y_true``.
      iii. *Backward*: ``loss.backward()`` propaga gradientes via cadeia.
      iv.  *Step*: ``optimizer.step()`` atualiza os pesos.

2. Avaliar o modelo no conjunto de validação (sem atualizar pesos).
3. Decidir parar (early stopping) ou continuar.

Salvaguardas implementadas
--------------------------
* **Reprodutibilidade**: ``torch.manual_seed`` + ``np.random.seed`` fixados
  no início. Cada chamada com a mesma seed produz exatamente o mesmo
  histórico de pesos.
* **Early stopping**: monitora ``val_loss``; salva os melhores pesos.
  Para se ``val_loss`` não melhora por ``patience`` épocas. Evita
  treinar até *overfit* completo.
* **Gradient clipping**: ``clip_grad_norm_`` limita o norm-2 dos gradientes
  por iteração. Mitiga *exploding gradients* — risco mais alto em RNNs
  com sequências longas ou taxas de aprendizado generosas.
* **Weight decay (L2)**: penalização ``α||W||²`` adicionada implicitamente
  pelo Adam (``weight_decay=...``). Regularização padrão para evitar
  pesos excessivamente grandes.
* **Modo eval no validation**: ``model.eval()`` + ``torch.no_grad()``
  desliga dropout e desativa o cálculo do grafo de gradientes —
  obrigatório para métricas honestas.

Sobre o critério de parada
--------------------------
``patience=30`` significa: se o ``val_loss`` não melhorar em 30 épocas
consecutivas, paramos. ``max_epochs=300`` é um teto de segurança caso
``val_loss`` continue melhorando devagar para sempre. Os melhores pesos
(menor ``val_loss`` visto) são restaurados ANTES de retornar o modelo —
nunca devolvemos os pesos da última época.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Regressão Quantílica: PinballLoss
# --------------------------------------------------------------------------- #
class PinballLoss(nn.Module):
    """Perda Pinball (quantílica) para regressão com múltiplos quantis.

    A função de perda pinball generaliza o MSE para otimizar quantis
    específicos da distribuição condicional de $y | X$. Para o quantil
    $\\tau$ e erro $e_i = y_i - \\hat{y}_i$:

    .. math::

        L_{\\tau}(e) =
        \\begin{cases}
            \\tau \\cdot e        & \\text{se } e \\geq 0 \\\\
            (\\tau - 1) \\cdot e & \\text{se } e < 0
        \\end{cases}

    Equivalentemente (sem ramificação condicional):

    .. math::

        L_{\\tau}(e) = \\max(\\tau \\cdot e,\\; (\\tau - 1) \\cdot e)

    Para **múltiplos quantis simultâneos** (e.g. $[0.1, 0.5, 0.9]$), o
    modelo produz uma saída por quantil e a perda total é a média das
    perdas individuais. O head do modelo deve ter
    ``head_output_size = len(quantiles)``.

    Propriedade chave: a perda é **assimétrica**. Para $\\tau = 0.9$, os
    resíduos negativos (sub-estimativas) são penalizados 9 vezes mais
    que os positivos. Isso força o modelo a aprender o limite superior
    que cobre 90 % das observações.

    Parameters
    ----------
    quantiles : sequence of float
        Quantis alvo em (0, 1). Padrão ``[0.1, 0.5, 0.9]`` para P10/P50/P90.
        O número de quantis deve ser igual a ``head_output_size`` do modelo.

    Notes
    -----
    Uso com ``train_lstm``::

        loss_fn = PinballLoss([0.1, 0.5, 0.9])
        model = LSTMRegressor(..., head_output_size=3)
        model, history = train_lstm(model, ..., loss_fn=loss_fn)

    Para garantir P10 ≤ P50 ≤ P90 nas predições, aplique
    ``np.sort(y_pred_raw, axis=1)`` *após* o treinamento (non-crossing).
    """

    def __init__(
        self, quantiles: list[float] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.register_buffer(
            "q", torch.tensor(list(quantiles), dtype=torch.float32),
        )

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula a perda pinball média.

        Parameters
        ----------
        y_pred : torch.Tensor, shape (batch, n_quantiles)
            Predições do modelo para cada quantil.
        y_true : torch.Tensor, shape (batch, 1)
            Valores observados.

        Returns
        -------
        torch.Tensor (escalar)
            Média da perda sobre amostras e quantis.
        """
        # errors: (batch, n_q) — broadcast automático de (batch,1) - (batch,n_q)
        errors = y_true - y_pred
        # max(τ·e, (τ-1)·e) — positivo quando e > 0 e τ > 0
        loss = torch.max(self.q * errors, (self.q - 1.0) * errors)
        return loss.mean()


# --------------------------------------------------------------------------- #
# Early stopping helper
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    """Mantém o estado do early stopping ao longo das épocas.

    :ivar patience: número de épocas SEM melhora aceitas antes de parar.
    :ivar min_delta: mínima melhora absoluta para ser considerada "melhora"
        (default 0.0 = qualquer queda já conta).
    :ivar counter: épocas decorridas desde a última melhora.
    :ivar best_val: menor ``val_loss`` observado até agora.
    :ivar best_state: ``state_dict()`` do modelo no momento da melhor
        ``val_loss``. Será restaurado ao final do treino.
    :ivar best_epoch: índice (0-based) da época que produziu ``best_val``.
    """

    patience: int = 30
    min_delta: float = 0.0
    counter: int = 0
    best_val: float = float("inf")
    best_state: dict | None = None
    best_epoch: int = -1

    def step(
        self, val_loss: float, model: nn.Module, epoch: int,
    ) -> bool:
        """Atualiza o estado e retorna ``True`` se for hora de parar.

        :param val_loss: perda de validação na época atual.
        :param model: modelo PyTorch — pesos serão "fotografados" se
            ``val_loss`` for o novo melhor.
        :param epoch: índice da época (0-based).
        :returns: ``True`` se ``counter > patience`` (parar treino).
        """
        improved = val_loss < (self.best_val - self.min_delta)
        if improved:
            self.best_val = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return self.counter > self.patience


# --------------------------------------------------------------------------- #
# Helper de reprodutibilidade
# --------------------------------------------------------------------------- #
def _set_seed(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade relevantes.

    Cobre ``random`` (puro Python), ``numpy.random`` e ``torch`` (CPU + CUDA).
    Pequenas variações ainda podem ocorrer em CUDA por causa de algoritmos
    não-determinísticos — para o nosso caso CPU, a seed é suficiente.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Função pública: train_lstm
# --------------------------------------------------------------------------- #
def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    loss_fn: nn.Module | None = None,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
    log_every: int = 10,
) -> tuple[nn.Module, pd.DataFrame]:
    """Treina um modelo PyTorch com Adam + early stopping.

    :param model: instância de qualquer ``nn.Module`` que aceite tensor
        ``(batch, seq_len, n_features)`` e devolva ``(batch, n_out)``.
    :param train_loader: ``DataLoader`` do split de treino (com ``shuffle=True``
        para randomização dos batches dentro da época).
    :param val_loader: ``DataLoader`` do split de validação (``shuffle=False``).
    :param max_epochs: limite superior de épocas (teto de segurança).
    :param lr: taxa de aprendizado do otimizador Adam.
    :param weight_decay: coeficiente $\\lambda$ da regularização $L_2$ aplicada
        a TODOS os pesos. Valores típicos: ``1e-5`` a ``1e-3``.
    :param patience: épocas sem melhora aceitas antes do early stopping.
    :param loss_fn: função de perda a otimizar. ``None`` (padrão) usa
        ``nn.MSELoss()`` — adequado para regressão pontual. Para regressão
        quantílica use :class:`PinballLoss`, e.g.::

            loss_fn = PinballLoss([0.1, 0.5, 0.9])

    :param grad_clip: norma máxima do gradiente global por iteração.
        Use ``None`` ou ``0`` para desabilitar.
    :param seed: semente para reprodutibilidade. Cada seed diferente gera
        uma inicialização de pesos diferente → multi-seed permite medir
        a variância de inicialização (essencial em bases pequenas).
    :param device: ``"cpu"`` ou ``"cuda"`` — o modelo e os tensores são
        movidos para esse device.
    :param verbose: se ``True``, imprime ``train_loss`` / ``val_loss`` a
        cada ``log_every`` épocas.
    :param log_every: intervalo de log quando ``verbose=True``.

    :returns: tupla ``(best_model, history_df)``:

        * ``best_model``: o próprio ``model`` (mutado) com os melhores
          pesos (menor ``val_loss``) RESTAURADOS.
        * ``history_df``: ``pd.DataFrame`` com colunas
          ``epoch, train_loss, val_loss``. Usado pelos plots de curva
          de aprendizado e como referência de convergência.
    """
    _set_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)

    history: list[dict] = []

    for epoch in range(max_epochs):
        # ---------- TREINO ----------
        model.train()
        train_losses: list[float] = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip,
                )
            optimizer.step()
            train_losses.append(float(loss.item()))

        # ---------- VALIDAÇÃO ----------
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                pred = model(Xb)
                val_losses.append(float(loss_fn(pred, yb).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            print(
                f"  época {epoch:>3d}: train={train_loss:.5f}  "
                f"val={val_loss:.5f}  best={stopper.best_val:.5f} "
                f"(época {stopper.best_epoch})"
            )

        if stopper.step(val_loss, model, epoch):
            if verbose:
                print(
                    f"  Early stopping na época {epoch} — sem melhora há "
                    f"{stopper.patience} épocas. Melhor val={stopper.best_val:.5f} "
                    f"na época {stopper.best_epoch}."
                )
            break

    # Restaura os melhores pesos antes de devolver o modelo
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model, pd.DataFrame(history)


# --------------------------------------------------------------------------- #
# Helper: predict em batches (sem gradiente)
# --------------------------------------------------------------------------- #
def predict_in_batches(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Roda o modelo em modo eval sobre todos os batches do loader.

    :param model: modelo treinado.
    :param loader: ``DataLoader`` (geralmente com ``shuffle=False`` para
        preservar a ordem das amostras).
    :param device: device para inferência.

    :returns: ``np.ndarray`` de predições ``(n,)`` (achatado).
    """
    model.eval()
    model = model.to(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy().ravel()
            preds.append(out)
    if not preds:
        return np.array([], dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)
