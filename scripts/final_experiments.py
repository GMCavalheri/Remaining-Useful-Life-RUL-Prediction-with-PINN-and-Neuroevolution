"""Milestone 6: retrain the final candidates over multiple seeds and compare.

Since ``rul.data.windowing.split_units_train_val`` uses ``config["seed"]``
to choose *which* dev unit becomes the validation unit, re-running a config
across several seeds isn't just "different weight initializations" -- it's
a different train/val unit split each time, closer in spirit to k-fold
cross-validation than to bare seed variance. That directly stress-tests
Phase 5's finding that the evolved champion's low validation RMSE (found
against a *single* held-out unit, after 96 genome evaluations competing to
minimize error on exactly that unit) might not reflect genuine held-out
generalization. The (fixed, NASA-defined) *test* units never change across
seeds, so test-set numbers stay directly comparable across the table.

Usage
-----
    uv run python scripts/final_experiments.py \\
        --configs configs/baseline_cnn.yaml configs/pinn_mono_only.yaml \\
                  configs/pinn_champion.yaml \\
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rul.baselines import build_baseline
from rul.config import load_config
from rul.data.datamodule import build_datamodule
from rul.models.pinn import build_pinn
from rul.physics.losses import make_pinn_loss
from rul.seed import get_device, set_seed
from rul.training.trainer import evaluate, train_model


def _build_model_and_loss(config, dm, device):
    model_cfg = config["model"]
    window_length = config["data"]["window_length"]
    n_features = len(dm.feature_names)

    if model_cfg["type"] == "pinn":
        n_health = dm.train.health.shape[1]
        model = build_pinn(model_cfg, window_length, n_features, n_health)
        health_std = torch.from_numpy(dm.health_scaler.std).to(device)
        loss_fn = make_pinn_loss(
            model_cfg.get("lambda_mono", 0.0), model_cfg.get("lambda_health", 0.0), health_std
        )
    else:
        model = build_baseline(model_cfg, window_length, n_features)
        loss_fn = None  # trainer's default plain-MSE loss
    return model, loss_fn


def run_one(config_path: Path, seed: int, device: torch.device) -> dict:
    config = load_config(config_path)
    config["seed"] = seed  # override the config's own seed for this run
    set_seed(seed, deterministic=config.get("deterministic", True))

    dm = build_datamodule(config)
    train_loader, val_loader, test_loader = dm.loaders(config["training"]["batch_size"])
    model, loss_fn = _build_model_and_loss(config, dm, device)

    kwargs = dict(
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        device=device,
    )
    if loss_fn is not None:
        kwargs["loss_fn"] = loss_fn
    result = train_model(model, train_loader, val_loader, **kwargs)

    test_metrics = evaluate(result.model, test_loader, device)
    return {
        "seed": seed,
        "val_unit": sorted(set(dm.val.unit.tolist())),
        "best_val_rmse": result.best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_nasa_score": test_metrics["nasa_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=Path("results/final_experiments.json"))
    args = parser.parse_args()

    device = get_device()
    summary = {}

    for config_path in args.configs:
        name = config_path.stem
        runs = [run_one(config_path, seed, device) for seed in args.seeds]
        test_rmses = np.array([r["test_rmse"] for r in runs])
        test_scores = np.array([r["test_nasa_score"] for r in runs])

        summary[name] = {
            "runs": runs,
            "test_rmse_mean": float(test_rmses.mean()),
            "test_rmse_std": float(test_rmses.std()),
            "test_nasa_score_mean": float(test_scores.mean()),
            "test_nasa_score_std": float(test_scores.std()),
        }
        print(
            f"{name}: test RMSE {test_rmses.mean():.3f} +/- {test_rmses.std():.3f}, "
            f"NASA score {test_scores.mean():.3f} +/- {test_scores.std():.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
