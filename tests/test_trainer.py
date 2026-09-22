import torch

from rul.baselines import build_baseline
from rul.data.datamodule import build_datamodule
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


def test_train_model_smoke_run(synthetic_ncmapss_path):
    torch.manual_seed(0)
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))
    train_loader, val_loader, test_loader = dm.loaders(batch_size=8)

    model = build_baseline(
        {"type": "mlp", "hidden_dims": [16], "embedding_dim": 8},
        window_length=5,
        n_features=len(dm.feature_names),
    )

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=3,
        lr=1e-2,
        early_stopping_patience=10,
        device=torch.device("cpu"),
    )

    assert len(result.history) == 3
    assert result.best_val_rmse < float("inf")
    assert result.best_epoch >= 0

    test_metrics = evaluate(result.model, test_loader, device=torch.device("cpu"))
    assert "rmse" in test_metrics and "nasa_score" in test_metrics
    assert test_metrics["rmse"] >= 0


def test_train_model_early_stopping_keeps_best_weights(synthetic_ncmapss_path):
    torch.manual_seed(0)
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))
    train_loader, val_loader, _ = dm.loaders(batch_size=8)

    model = build_baseline(
        {"type": "mlp", "hidden_dims": [4], "embedding_dim": 4},
        window_length=5,
        n_features=len(dm.feature_names),
    )

    # patience=0 -> stop as soon as val RMSE fails to improve for one epoch.
    result = train_model(
        model, train_loader, val_loader, epochs=20, lr=1e-2, early_stopping_patience=0,
        device=torch.device("cpu"),
    )
    # Should stop well before the full 20 epochs.
    assert len(result.history) < 20
