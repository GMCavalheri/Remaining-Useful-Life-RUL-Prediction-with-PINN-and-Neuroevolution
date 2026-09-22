from typing import Any

from torch import nn

from rul.baselines.cnn import build_cnn_baseline
from rul.baselines.lstm import build_rnn_baseline
from rul.baselines.mlp import build_mlp_baseline

_BUILDERS = {
    "mlp": build_mlp_baseline,
    "cnn": build_cnn_baseline,
    "lstm": build_rnn_baseline,
    "gru": build_rnn_baseline,
}


def build_baseline(model_cfg: dict[str, Any], window_length: int, n_features: int) -> nn.Module:
    """Build a plain (non-physics-informed) baseline model from a config's ``model`` section.

    ``model_cfg["type"]`` selects the encoder family; every other key is
    that encoder's own hyperparameters (see the individual ``build_*``
    functions for defaults).
    """
    model_type = model_cfg["type"]
    if model_type not in _BUILDERS:
        raise ValueError(f"Unknown model type {model_type!r}, expected one of {list(_BUILDERS)}")
    return _BUILDERS[model_type](model_cfg, window_length, n_features)
