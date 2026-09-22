"""MLP baseline: no notion of time, every (cycle, feature) is an independent input."""

from __future__ import annotations

from typing import Any

from rul.models.encoders import build_encoder
from rul.models.heads import RULRegressor


def build_mlp_baseline(
    model_cfg: dict[str, Any], window_length: int, n_features: int
) -> RULRegressor:
    encoder = build_encoder(
        "mlp",
        window_length=window_length,
        n_features=n_features,
        hidden_dims=tuple(model_cfg.get("hidden_dims", (128, 64))),
        output_dim=model_cfg.get("embedding_dim", 32),
        dropout=model_cfg.get("dropout", 0.0),
    )
    return RULRegressor(encoder)
