"""1D-CNN baseline: local temporal patterns via convolution along the cycle axis."""

from __future__ import annotations

from typing import Any

from rul.models.encoders import build_encoder
from rul.models.heads import RULRegressor


def build_cnn_baseline(
    model_cfg: dict[str, Any], window_length: int, n_features: int
) -> RULRegressor:
    encoder = build_encoder(
        "cnn",
        window_length=window_length,
        n_features=n_features,
        channels=tuple(model_cfg.get("channels", (32, 64))),
        kernel_size=model_cfg.get("kernel_size", 5),
        output_dim=model_cfg.get("embedding_dim", 32),
        dropout=model_cfg.get("dropout", 0.0),
    )
    return RULRegressor(encoder)
