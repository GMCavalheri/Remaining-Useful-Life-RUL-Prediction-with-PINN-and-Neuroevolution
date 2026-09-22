"""End-to-end smoke tests: real datamodule -> PINN -> physics loss -> trainer."""

import torch

from rul.data.datamodule import build_datamodule
from rul.models.pinn import build_pinn
from rul.physics.losses import make_pinn_loss
from rul.training.trainer import evaluate, train_model


def _config(raw_dir):
    return {
        "seed": 0,
        "data": {
            "subset": "DS02",
            "raw_dir": str(raw_dir),
            "window_length": 5,
            "downsample_factor": 1,
            "val_fraction": 0.34,
            "feature_groups": ("W", "X_s"),
        },
    }


def _build(raw_dir, lambda_mono, lambda_health):
    torch.manual_seed(0)
    dm = build_datamodule(_config(raw_dir))
    train_loader, val_loader, test_loader = dm.loaders(batch_size=8)

    n_health = dm.train.health.shape[1]
    model = build_pinn(
        {"encoder": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8},
        window_length=5,
        n_features=len(dm.feature_names),
        n_health=n_health,
    )
    health_std = torch.from_numpy(dm.health_scaler.std)
    loss_fn = make_pinn_loss(lambda_mono, lambda_health, health_std)
    return model, train_loader, val_loader, test_loader, loss_fn


def test_pinn_trains_with_both_physics_terms(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    model, train_loader, val_loader, test_loader, loss_fn = _build(
        raw_dir, lambda_mono=1.0, lambda_health=1.0
    )

    result = train_model(
        model, train_loader, val_loader, epochs=3, lr=1e-2, early_stopping_patience=10,
        device=torch.device("cpu"), loss_fn=loss_fn,
    )
    assert len(result.history) == 3
    assert result.best_val_rmse < float("inf")

    test_metrics = evaluate(result.model, test_loader, device=torch.device("cpu"))
    assert test_metrics["rmse"] >= 0


def test_pinn_no_physics_ablation_still_trains(synthetic_ncmapss_path):
    """lambda_mono=lambda_health=0 -- the health head exists but contributes
    no gradient, exercising the same code path as configs/pinn_no_physics.yaml."""
    raw_dir = synthetic_ncmapss_path.parent
    model, train_loader, val_loader, test_loader, loss_fn = _build(
        raw_dir, lambda_mono=0.0, lambda_health=0.0
    )

    result = train_model(
        model, train_loader, val_loader, epochs=2, lr=1e-2, early_stopping_patience=10,
        device=torch.device("cpu"), loss_fn=loss_fn,
    )
    assert len(result.history) == 2
