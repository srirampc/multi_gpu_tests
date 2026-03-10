import typing as t

import torch
import torch.nn as nn
from typing_extensions import override


# ── A: Manual Elman RNN ───────────────────────────────────────────
@t.final
class ManualRNNModel(nn.Module):
    """
    Hand-rolled Elman RNN.  Each branch is a set of plain weight matrices.
    h_t = tanh/relu( x_t @ W_x  +  h_{t-1} @ W_h  +  b )

    Weights for branch i are created with .to(dev_i) so they live on the
    correct device from the start.  Input tensors are moved with .to()
    inside forward() — a single explicit line per branch, no magic needed.
    """

    def __init__(
        self, units: int, device0: str, device1: str, out_device: str, params: dict[str, t.Any]
    ):
        super().__init__()

        self.units = units
        self.device0 = device0
        self.device1 = device1
        self.out_device = out_device

        # Embedding lives on DEV0 (or CPU); both branches read from it
        self.embedding = nn.Embedding(params["VOCAB_SIZE"], params["EMBED_DIM"]).to(device0)

        # ── Branch 1 weights on DEV0 ──────────────────────────────
        self.W_x0 = nn.Parameter(torch.randn(params["EMBED_DIM"], units) * 0.01)
        self.W_h0 = nn.Parameter(torch.randn(units, units) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(units))
        # Move branch 1 parameters to DEV0
        self.W_x0 = nn.Parameter(self.W_x0.to(device0))
        self.W_h0 = nn.Parameter(self.W_h0.to(device0))
        self.b0 = nn.Parameter(self.b0.to(device0))

        # ── Branch 2 weights on DEV1 ──────────────────────────────
        self.W_x1 = nn.Parameter(torch.randn(params["EMBED_DIM"], units) * 0.01)
        self.W_h1 = nn.Parameter(torch.randn(units, units) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(units))
        self.W_x1 = nn.Parameter(self.W_x1.to(device1))
        self.W_h1 = nn.Parameter(self.W_h1.to(device1))
        self.b1 = nn.Parameter(self.b1.to(device1))

        # ── Output head on DEV_HEAD ───────────────────────────────
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(units * 2, 1).to(out_device)

    def _rnn_branch(self, emb, W_x, W_h, b, activation):
        """Unroll one Elman RNN branch over time."""
        batch = emb.size(0)
        # h lives on the same device as the weights
        h = torch.zeros(batch, self.units, device=W_x.device)
        for tx in range(emb.size(1)):
            x_t = emb[:, tx, :]  # (batch, embed_dim)
            h = activation(x_t @ W_x + h @ W_h + b)
        return h  # (batch, rnn_units)

    @override
    def forward(self, x):
        # x: (batch, seq_len) — long tensor on CPU from DataLoader

        # Embedding on DEV0
        emb = self.embedding(x.to(self.device0))  # (batch, seq_len, embed_dim)

        # Branch 1 on DEV0 — tanh
        b1 = self._rnn_branch(emb, self.W_x0, self.W_h0, self.b0, torch.tanh)

        # Branch 2 on DEV1 — relu
        # .to(DEV1): one explicit line, always works, no framework magic needed
        b2 = self._rnn_branch(emb.to(self.device1), self.W_x1, self.W_h1, self.b1, torch.relu)

        # Merge: move b2 to DEV_HEAD, concatenate, classify
        merged = torch.cat([b1, b2.to(self.out_device)], dim=-1)
        return self.fc(self.dropout(merged)).squeeze(-1)


# ── B: torch.nn.RNN (built-in) ────────────────────────────────────


@t.final
class BuiltinRNNModel(nn.Module):
    """
    Uses torch.nn.RNN which wraps a cuDNN-optimised kernel.
    Placing a module on a device is a single .to(device) call in __init__.
    In forward(), moving tensors between devices is just tensor.to(device).
    """

    def __init__(
        self, units: int, device0: str, device1: str, out_device: str, params: dict[str, t.Any]
    ):
        super().__init__()
        self.units = units
        self.device0 = device0
        self.device1 = device1
        self.out_device = out_device

        self.embedding = nn.Embedding(params["VOCAB_SIZE"], params["EMBED_DIM"]).to(device0)

        # Branch 1 on DEV0 — tanh (torch.nn.RNN default)
        self.rnn0 = nn.RNN(params["EMBED_DIM"], units, batch_first=True, nonlinearity="tanh").to(
            device0
        )

        # Branch 2 on DEV1 — relu
        self.rnn1 = nn.RNN(params["EMBED_DIM"], units, batch_first=True, nonlinearity="relu").to(
            device1
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(units * 2, 1).to(out_device)

    @override
    def forward(self, x):
        emb = self.embedding(x.to(self.device0))  # (batch, seq, embed_dim) on DEV0

        # Branch 1 — output shape: (batch, seq, units); take last step
        _, h0 = self.rnn0(emb)  # h0: (1, batch, units)
        b1 = h0.squeeze(0)  # (batch, units) on DEV0

        # Branch 2 — move embedding to DEV1 first
        _, h1 = self.rnn1(emb.to(self.device1))
        b2 = h1.squeeze(0)  # (batch, units) on DEV1

        merged = torch.cat([b1, b2.to(self.out_device)], dim=-1)
        return self.fc(self.dropout(merged)).squeeze(-1)


# ── C: Custom RNNCell inside a manual loop ────────────────────────
@t.final
class CustomRNNCell(nn.Module):
    """One time-step of an Elman RNN.  Same formula as ManualRNNModel
    but packaged as a reusable cell — mirrors the TF CustomSimpleRNNCell.
    The cell is placed on a device via .to(device) in the parent model.
    """

    def __init__(self, input_size: int, hidden_size: int, activation: str = "tanh"):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = torch.tanh if activation == "tanh" else torch.relu
        self.W_x = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size))

    @override
    def forward(self, x_t, h):
        return self.activation(x_t @ self.W_x + h @ self.W_h + self.b)


@t.final
class CustomCellModel(nn.Module):
    """Drives two CustomRNNCells with an explicit time loop.
    Mirrors Model C from the TF version but without any placement headaches.
    """

    def __init__(
        self, units: int, device0: str, device1: str, out_device: str, params: dict[str, t.Any]
    ):
        super().__init__()
        self.units = units
        self.device0 = device0
        self.device1 = device1
        self.out_device = out_device

        self.embedding = nn.Embedding(params["VOCAB_SIZE"], params["EMBED_DIM"]).to(device0)

        # Cell for branch 1 on DEV0 — tanh
        self.cell0 = CustomRNNCell(params["EMBED_DIM"], units, activation="tanh").to(device0)
        # Cell for branch 2 on DEV1 — relu
        self.cell1 = CustomRNNCell(params["EMBED_DIM"], units, activation="relu").to(device1)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(units * 2, 1).to(out_device)

    def _unroll(self, emb, cell):
        """Unroll a CustomRNNCell over the time dimension."""
        h = torch.zeros(emb.size(0), cell.hidden_size, device=emb.device)
        for tx in range(emb.size(1)):
            h = cell(emb[:, tx, :], h)
        return h

    @override
    def forward(self, x):
        emb = self.embedding(x.to(self.device0))

        b1 = self._unroll(emb, self.cell0)  # on DEV0
        b2 = self._unroll(emb.to(self.device1), self.cell1)  # on DEV1

        merged = torch.cat([b1, b2.to(self.out_device)], dim=-1)
        return self.fc(self.dropout(merged)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# 4. TRAINING & EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, criterion, out_device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)  # x_batch still on CPU
        loss = criterion(logits.to(out_device), y_batch.to(out_device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        preds = (torch.sigmoid(logits.to("cpu")) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        n += x_batch.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, out_device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    criterion = nn.BCEWithLogitsLoss()
    for x_batch, y_batch in loader:
        logits = model(x_batch)
        loss = criterion(logits.to(out_device), y_batch.to(out_device))
        total_loss += loss.item() * x_batch.size(0)
        preds = (torch.sigmoid(logits.to("cpu")) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        n += x_batch.size(0)
    return total_loss / n, correct / n
