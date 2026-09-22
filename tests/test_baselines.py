import pytest
import torch

from rul.baselines import build_baseline

WINDOW, FEATURES = 20, 7


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"type": "mlp", "hidden_dims": [16], "embedding_dim": 8},
        {"type": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8},
        {"type": "lstm", "hidden_size": 8, "embedding_dim": 8},
        {"type": "gru", "hidden_size": 8, "embedding_dim": 8},
    ],
)
def test_build_baseline_forward_shape(model_cfg):
    model = build_baseline(model_cfg, window_length=WINDOW, n_features=FEATURES)
    x = torch.randn(3, WINDOW, FEATURES)
    out = model(x)
    assert out.shape == (3,)
    assert torch.all(out >= 0)


def test_build_baseline_unknown_type_raises():
    with pytest.raises(ValueError):
        build_baseline({"type": "transformer"}, window_length=WINDOW, n_features=FEATURES)
