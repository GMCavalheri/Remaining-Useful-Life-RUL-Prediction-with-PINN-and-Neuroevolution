import pytest
import torch

from rul.models.encoders import CNNEncoder, MLPEncoder, RNNEncoder, build_encoder
from rul.models.heads import RULHead, RULRegressor

BATCH, WINDOW, FEATURES = 4, 20, 7


@pytest.fixture(params=["mlp", "cnn", "lstm", "gru"])
def encoder(request):
    return build_encoder(request.param, window_length=WINDOW, n_features=FEATURES, output_dim=16)


def test_encoder_output_shape(encoder):
    x = torch.randn(BATCH, WINDOW, FEATURES)
    out = encoder(x)
    assert out.shape == (BATCH, 16)


def test_encoder_gradients_flow(encoder):
    x = torch.randn(BATCH, WINDOW, FEATURES, requires_grad=True)
    out = encoder(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_unknown_encoder_type_raises():
    with pytest.raises(ValueError):
        build_encoder("transformer", window_length=WINDOW, n_features=FEATURES)


def test_rul_head_is_nonnegative():
    head = RULHead(embedding_dim=8)
    embedding = torch.randn(BATCH, 8) * 100  # even large-magnitude inputs
    out = head(embedding)
    assert torch.all(out >= 0)
    assert out.shape == (BATCH,)


def test_rul_regressor_end_to_end(encoder):
    model = RULRegressor(encoder)
    x = torch.randn(BATCH, WINDOW, FEATURES)
    out = model(x)
    assert out.shape == (BATCH,)
    assert torch.all(out >= 0)


def test_cnn_encoder_handles_different_window_lengths():
    encoder = CNNEncoder(window_length=WINDOW, n_features=FEATURES, output_dim=16)
    for length in (5, 20, 100):
        x = torch.randn(2, length, FEATURES)
        assert encoder(x).shape == (2, 16)


def test_mlp_encoder_flattens_window_and_features():
    encoder = MLPEncoder(window_length=WINDOW, n_features=FEATURES, output_dim=16)
    x = torch.randn(BATCH, WINDOW, FEATURES)
    assert encoder(x).shape == (BATCH, 16)


def test_rnn_encoder_supports_multiple_layers():
    encoder = RNNEncoder(
        window_length=WINDOW, n_features=FEATURES, hidden_size=12, num_layers=2, output_dim=16
    )
    x = torch.randn(BATCH, WINDOW, FEATURES)
    assert encoder(x).shape == (BATCH, 16)
