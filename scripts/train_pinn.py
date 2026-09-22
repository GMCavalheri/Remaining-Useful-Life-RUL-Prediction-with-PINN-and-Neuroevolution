"""Train the PINN (or one of its physics-ablation variants) and report metrics.

Usage
-----
    uv run python scripts/train_pinn.py --config configs/pinn_full.yaml
    uv run python scripts/train_pinn.py --config configs/pinn_no_physics.yaml
    uv run python scripts/train_pinn.py --config configs/pinn_mono_only.yaml
    uv run python scripts/train_pinn.py --config configs/pinn_health_only.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rul.config import load_config
from rul.data.datamodule import build_datamodule
from rul.models.pinn import build_pinn
from rul.physics.losses import make_pinn_loss
from rul.seed import get_device, set_seed
from rul.training.trainer import evaluate, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], deterministic=config.get("deterministic", True))
    device = get_device()

    dm = build_datamodule(config)
    train_loader, val_loader, test_loader = dm.loaders(config["training"]["batch_size"])

    window_length = config["data"]["window_length"]
    n_features = len(dm.feature_names)
    n_health = dm.train.health.shape[1]

    model_cfg = config["model"]
    model = build_pinn(model_cfg, window_length, n_features, n_health)

    health_std = torch.from_numpy(dm.health_scaler.std).to(device)
    loss_fn = make_pinn_loss(
        lambda_mono=model_cfg.get("lambda_mono", 0.0),
        lambda_health=model_cfg.get("lambda_health", 0.0),
        health_std=health_std,
    )

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        device=device,
        loss_fn=loss_fn,
    )

    test_metrics = evaluate(result.model, test_loader, device)
    print(f"Best epoch: {result.best_epoch} (val RMSE {result.best_val_rmse:.3f})")
    print(f"Test metrics: {test_metrics}")

    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config_path": str(args.config),
        "lambda_mono": model_cfg.get("lambda_mono", 0.0),
        "lambda_health": model_cfg.get("lambda_health", 0.0),
        "best_epoch": result.best_epoch,
        "best_val_rmse": result.best_val_rmse,
        "test_metrics": test_metrics,
        "history": result.history,
    }
    report_path = output_dir / f"{args.config.stem}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
