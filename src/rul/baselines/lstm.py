"""Recurrent (LSTM/GRU) baseline: carries a hidden state across the window's cycles."""

from __future__ import annotations

from typing import Any

from rul.models.encoders import build_encoder
from rul.models.heads import RULRegressor


def build_rnn_baseline(
    model_cfg: dict[str, Any], window_length: int, n_features: int
) -> RULRegressor:
    encoder = build_encoder(
        model_cfg.get("rnn_type", "lstm"),
        window_length=window_length,
        n_features=n_features,
        hidden_size=model_cfg.get("hidden_size", 64),
        num_layers=model_cfg.get("num_layers", 1),
        output_dim=model_cfg.get("embedding_dim", 32),
        dropout=model_cfg.get("dropout", 0.0),
    )
    return RULRegressor(encoder)
