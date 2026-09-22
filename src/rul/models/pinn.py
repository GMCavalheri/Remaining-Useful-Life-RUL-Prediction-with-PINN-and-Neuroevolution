"""The PINN: a shared encoder with two heads (RUL + health parameters).

.. math::

    h = \\text{encoder}(x), \\qquad
    \\widehat{\\text{RUL}} = \\text{RULHead}(h), \\qquad
    \\widehat{T} = \\text{HealthHead}(h)

The same :func:`rul.models.encoders.build_encoder` used by the Phase 3
baselines is reused unchanged here — the only architectural difference
between a baseline and the PINN is the *second head* plus the physics loss
terms applied during training (:mod:`rul.physics.losses`); the encoder
itself, and therefore the neuroevolution genome (Phase 5), does not need to
know or care which one it's attached to.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rul.models.encoders import build_encoder
from rul.models.heads import RULHead


class HealthHead(nn.Module):
    """Predicts the health-parameter vector (unconstrained: modifiers can be +/-)."""

    def __init__(self, embedding_dim: int, n_health: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, n_health)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(embedding)


class PINNModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_health: int):
        super().__init__()
        self.encoder = encoder
        self.rul_head = RULHead(encoder.output_dim)
        self.health_head = HealthHead(encoder.output_dim, n_health)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return self.rul_head(embedding), self.health_head(embedding)


def build_pinn(
    model_cfg: dict[str, Any], window_length: int, n_features: int, n_health: int
) -> PINNModel:
    """Build a :class:`PINNModel` from a config's ``model`` section.

    Mirrors ``rul.baselines.build_baseline``'s per-encoder-type kwarg
    mapping (see e.g. ``rul.baselines.cnn``) rather than passing the whole
    config through, so an unexpected key in a config fails loudly instead
    of silently reaching an unrelated encoder constructor.
    """
    encoder_type = model_cfg["encoder"]
    common = dict(
        window_length=window_length,
        n_features=n_features,
        output_dim=model_cfg.get("embedding_dim", 32),
        dropout=model_cfg.get("dropout", 0.0),
    )
    if encoder_type == "mlp":
        encoder = build_encoder(
            "mlp", hidden_dims=tuple(model_cfg.get("hidden_dims", (128, 64))), **common
        )
    elif encoder_type == "cnn":
        encoder = build_encoder(
            "cnn",
            channels=tuple(model_cfg.get("channels", (32, 64))),
            kernel_size=model_cfg.get("kernel_size", 5),
            **common,
        )
    elif encoder_type in ("lstm", "gru"):
        encoder = build_encoder(
            encoder_type,
            hidden_size=model_cfg.get("hidden_size", 64),
            num_layers=model_cfg.get("num_layers", 1),
            **common,
        )
    else:
        raise ValueError(f"Unknown encoder {encoder_type!r} for PINN")

    return PINNModel(encoder, n_health)
