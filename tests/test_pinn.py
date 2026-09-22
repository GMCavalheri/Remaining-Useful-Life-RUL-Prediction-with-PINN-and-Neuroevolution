import pytest
import torch

from rul.models.pinn import build_pinn

WINDOW, FEATURES, N_HEALTH = 20, 7, 10


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"encoder": "mlp", "hidden_dims": [16], "embedding_dim": 8},
        {"encoder": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8},
        {"encoder": "lstm", "hidden_size": 8, "embedding_dim": 8},
        {"encoder": "gru", "hidden_size": 8, "embedding_dim": 8},
    ],
)
def test_build_pinn_output_shapes(model_cfg):
    model = build_pinn(model_cfg, window_length=WINDOW, n_features=FEATURES, n_health=N_HEALTH)
    x = torch.randn(4, WINDOW, FEATURES)
    rul_pred, health_pred = model(x)

    assert rul_pred.shape == (4,)
    assert torch.all(rul_pred >= 0)  # softplus RUL head
    assert health_pred.shape == (4, N_HEALTH)


def test_build_pinn_unknown_encoder_raises():
    with pytest.raises(ValueError):
        build_pinn(
            {"encoder": "transformer"}, window_length=WINDOW, n_features=FEATURES, n_health=N_HEALTH
        )


def test_pinn_gradients_flow_to_both_heads():
    model = build_pinn(
        {"encoder": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8},
        window_length=WINDOW,
        n_features=FEATURES,
        n_health=N_HEALTH,
    )
    x = torch.randn(4, WINDOW, FEATURES)
    rul_pred, health_pred = model(x)
    (rul_pred.sum() + health_pred.sum()).backward()

    assert model.rul_head.linear.weight.grad is not None
    assert model.health_head.linear.weight.grad is not None
    assert any(p.grad is not None for p in model.encoder.parameters())
