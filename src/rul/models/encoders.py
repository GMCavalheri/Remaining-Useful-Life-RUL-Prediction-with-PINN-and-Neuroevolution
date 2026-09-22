"""Encoders: ``(batch, window_length, n_features) -> (batch, embedding_dim)``.

Every model in this project — every baseline (Phase 3) and later the PINN
(Phase 4) — is "an encoder plus a head" (see :mod:`rul.models.heads`). This
module holds three interchangeable encoders, all normalized to the same
input/output contract, which is exactly what the neuroevolution search
(Phase 5) needs: swapping the encoder type is then a genome gene, not a
different code path.

- :class:`MLPEncoder` — flattens time and features together. No notion of
  "time" at all; every window position is just another input dimension. The
  cheapest model, and a useful lower bound: if a fancier encoder can't beat
  it, the extra structure isn't earning its keep.
- :class:`CNNEncoder` — 1D convolutions slide a small kernel along the time
  axis, so it detects local *patterns of change* (e.g. a sensor drifting
  over a few consecutive cycles) rather than caring about the absolute
  window position.
- :class:`RNNEncoder` — an LSTM or GRU processes the window step by step,
  carrying a hidden state forward; well suited to a trajectory where later
  cycles matter more for predicting imminent failure than earlier ones.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        window_length: int,
        n_features: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        output_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        dims = [window_length * n_features, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # Drop the trailing activation/dropout after the final projection.
        self.net = nn.Sequential(*layers[:-1] if dropout == 0 else layers[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class CNNEncoder(nn.Module):
    def __init__(
        self,
        window_length: int,
        n_features: int,
        channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 5,
        output_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        in_channels = n_features
        conv_layers: list[nn.Module] = []
        for out_channels in channels:
            conv_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
                nn.ReLU(),
            ]
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        # Global average pool over time -> fixed-size embedding regardless
        # of window_length, then project to output_dim.
        self.project = nn.Linear(in_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, F) -> (B, F, L) for Conv1d, which expects channels first.
        h = self.conv(x.transpose(1, 2))
        h = h.mean(dim=2)  # global average pool over time -> (B, C)
        return self.project(h)


class RNNEncoder(nn.Module):
    def __init__(
        self,
        window_length: int,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        output_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last = self.rnn(x)
        # LSTM returns (h_n, c_n); GRU returns h_n directly.
        h_n = last[0] if isinstance(last, tuple) else last
        last_layer_hidden = h_n[-1]  # (B, hidden_size), final layer's state
        return self.project(last_layer_hidden)


def build_encoder(
    encoder_type: str, window_length: int, n_features: int, **kwargs
) -> nn.Module:
    """Factory used by baseline configs and the neuroevolution genome alike."""
    encoders = {"mlp": MLPEncoder, "cnn": CNNEncoder, "lstm": RNNEncoder, "gru": RNNEncoder}
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder_type {encoder_type!r}, expected one of {list(encoders)}")

    if encoder_type in ("lstm", "gru"):
        kwargs.setdefault("rnn_type", encoder_type)
    return encoders[encoder_type](window_length=window_length, n_features=n_features, **kwargs)
