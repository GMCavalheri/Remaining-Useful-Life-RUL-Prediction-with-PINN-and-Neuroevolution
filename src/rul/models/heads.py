"""Prediction heads and the encoder+head wrapper used by every model.

RUL is physically bounded below by zero — an engine cannot have "negative
cycles" left. A plain linear output layer has no way to know that, so
:class:`RULHead` applies ``softplus`` to the raw linear output:

.. math::

    \\widehat{\\text{RUL}} = \\log(1 + e^{z}), \\qquad z = w^\\top h + b

``softplus`` is a smooth (everywhere-differentiable) approximation of
``relu`` — unlike a hard clip to zero, it has a nonzero gradient everywhere,
so a large negative raw prediction can still be corrected by gradient
descent instead of getting stuck in a flat, zero-gradient region.
"""

from __future__ import annotations

import torch
from torch import nn


class RULHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.linear(embedding)).squeeze(-1)


class RULRegressor(nn.Module):
    """``encoder -> RULHead``. The plain (non-physics-informed) baseline model."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = RULHead(encoder.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
