"""Milestone 7: regenerate the report figures from saved run outputs.

Reads ``results/evolution/evolution_checkpoint.json`` (Phase 5) and
``results/final_experiments.json`` (Phase 6), retrains the champion once
more to get per-cycle predictions for one test unit, and writes PNGs to
``assets/`` (committed, unlike ``results/*`` which is regenerable scratch
output -- see the project README).

Usage
-----
    uv run python scripts/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rul.config import load_config
from rul.data.datamodule import build_datamodule
from rul.evolution.nsga import fast_non_dominated_sort
from rul.models.pinn import build_pinn
from rul.physics.losses import make_pinn_loss
from rul.seed import get_device, set_seed
from rul.training.trainer import train_model

ASSETS_DIR = Path("assets")


def plot_evolution_convergence(checkpoint_path: Path) -> None:
    data = json.loads(checkpoint_path.read_text())
    history = data["history"]
    generations = [h["generation"] for h in history]
    best = [h["best_val_rmse"] for h in history]
    mean = [h["mean_val_rmse"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(generations, best, marker="o", label="best val RMSE")
    ax.plot(generations, mean, marker="o", linestyle="--", label="population mean val RMSE")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation RMSE (cycles)")
    ax.set_title("NSGA-II search convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "evolution_convergence.png", dpi=150)
    plt.close(fig)


def plot_pareto_front(checkpoint_path: Path) -> None:
    data = json.loads(checkpoint_path.read_text())
    population = data["population"]
    objectives = [(ind["fitness"]["val_rmse"], ind["fitness"]["n_params"]) for ind in population]
    fronts = fast_non_dominated_sort(objectives)

    fig, ax = plt.subplots(figsize=(6, 4))
    for rank, front in enumerate(fronts):
        xs = [objectives[i][1] for i in front]
        ys = [objectives[i][0] for i in front]
        label = "front 0 (Pareto-optimal)" if rank == 0 else f"front {rank}"
        ax.scatter(xs, ys, label=label, s=50 if rank == 0 else 25, zorder=3 - rank)
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Validation RMSE (cycles)")
    ax.set_title("Final population: accuracy vs. model size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "pareto_front.png", dpi=150)
    plt.close(fig)


def plot_final_comparison(summary_path: Path) -> None:
    data = json.loads(summary_path.read_text())
    names = list(data.keys())
    means = [data[n]["test_rmse_mean"] for n in names]
    stds = [data[n]["test_rmse_std"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=stds, capsize=6)
    ax.set_ylabel("Test RMSE (cycles), mean +/- std over seeds")
    ax.set_title("Final comparison (multi-seed)")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "final_comparison.png", dpi=150)
    plt.close(fig)


def plot_rul_predictions(config_path: Path, test_unit: float) -> None:
    config = load_config(config_path)
    set_seed(config["seed"], deterministic=config.get("deterministic", True))
    device = get_device()

    dm = build_datamodule(config)
    train_loader, val_loader, test_loader = dm.loaders(config["training"]["batch_size"])
    n_health = dm.train.health.shape[1]
    model_cfg = config["model"]
    model = build_pinn(model_cfg, config["data"]["window_length"], len(dm.feature_names), n_health)
    health_std = torch.from_numpy(dm.health_scaler.std).to(device)
    loss_fn = make_pinn_loss(
        model_cfg.get("lambda_mono", 0.0), model_cfg.get("lambda_health", 0.0), health_std
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

    mask = dm.test.unit.numpy() == test_unit
    X = dm.test.X[mask]
    y_true = dm.test.y[mask].numpy()
    cycles = dm.test.cycle[mask].numpy()

    result.model.eval()
    with torch.no_grad():
        y_pred, _ = result.model(X.to(device))
    y_pred = y_pred.cpu().numpy()

    order = np.argsort(cycles)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cycles[order], y_true[order], marker="o", label="true RUL")
    ax.plot(cycles[order], y_pred[order], marker="x", label="predicted RUL")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Champion PINN: RUL over the trajectory, test unit {test_unit:.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"rul_predictions_unit{test_unit:.0f}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    evo_checkpoint = Path("results/evolution/evolution_checkpoint.json")
    if evo_checkpoint.exists():
        plot_evolution_convergence(evo_checkpoint)
        plot_pareto_front(evo_checkpoint)
        print("Wrote evolution_convergence.png, pareto_front.png")

    final_summary = Path("results/final_experiments.json")
    if final_summary.exists():
        plot_final_comparison(final_summary)
        print("Wrote final_comparison.png")

    champion_config = Path("configs/pinn_champion.yaml")
    if champion_config.exists():
        plot_rul_predictions(champion_config, test_unit=11.0)
        print("Wrote rul_predictions_unit11.png")


if __name__ == "__main__":
    main()
