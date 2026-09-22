"""Fitness evaluation: one genome -> (validation RMSE, parameter count).

Both objectives are minimized. A full, converged training run (Phase 3/4's
~100 epochs) per genome would make even a small search prohibitively slow,
so fitness evaluation uses a small ``epochs`` *budget* with early stopping —
"good enough to rank genomes against each other," not "the number to
report." Once the search converges on a Pareto front, the final champion(s)
should be retrained with the full budget and multiple seeds (a later,
separate step — not done inside the search loop).

``window_length`` is a gene, which means it changes the *data* (window
size), not just the model — so a genome's fitness evaluation may need to
rebuild the datamodule for a window length not seen yet. Since only 4
window lengths are in the search space (see
:mod:`rul.evolution.genome`), :class:`DataModuleCache` builds each one at
most once per search rather than once per genome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from rul.data.datamodule import RULDataModule, build_datamodule
from rul.evolution.genome import Genome
from rul.models.pinn import build_pinn
from rul.physics.losses import make_pinn_loss
from rul.training.trainer import evaluate, train_model


class DataModuleCache:
    """Builds a :class:`RULDataModule` per distinct ``window_length``, once."""

    def __init__(self, base_config: dict[str, Any]):
        self._base_config = base_config
        self._cache: dict[int, RULDataModule] = {}

    def get(self, window_length: int) -> RULDataModule:
        if window_length not in self._cache:
            config = copy.deepcopy(self._base_config)
            config["data"]["window_length"] = window_length
            self._cache[window_length] = build_datamodule(config)
        return self._cache[window_length]


@dataclass
class FitnessResult:
    val_rmse: float
    n_params: int
    val_nasa_score: float


def evaluate_genome(
    genome: Genome,
    dm_cache: DataModuleCache,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    seed: int,
) -> FitnessResult:
    """Train ``genome`` for a small epoch budget and return its fitness."""
    torch.manual_seed(seed)
    dm = dm_cache.get(genome.window_length)

    n_health = dm.train.health.shape[1]
    model = build_pinn(genome.to_model_cfg(), genome.window_length, len(dm.feature_names), n_health)
    n_params = sum(p.numel() for p in model.parameters())

    if len(dm.train) == 0 or len(dm.val) == 0:
        # genome.window_length exceeded some unit's cycle count, leaving no
        # windows to train/validate on (see rul.data.windowing.make_windows'
        # per-unit skip-if-too-short behavior, and the WindowScaler fix for
        # the same case). This genome simply cannot be trained -- give it
        # the worst possible fitness rather than crash the whole search;
        # NSGA-II's dominance-based selection (rul.evolution.nsga) then
        # discards it on its own, the same as any other unfit individual.
        return FitnessResult(val_rmse=float("inf"), n_params=n_params, val_nasa_score=float("inf"))

    train_loader, val_loader, _test_loader = dm.loaders(config["training"]["batch_size"])
    health_std = torch.from_numpy(dm.health_scaler.std).to(device)
    loss_fn = make_pinn_loss(genome.lambda_mono, genome.lambda_health, health_std)

    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=genome.lr,
        early_stopping_patience=max(3, epochs // 4),
        device=device,
        loss_fn=loss_fn,
    )
    val_metrics = evaluate(result.model, val_loader, device)
    return FitnessResult(
        val_rmse=val_metrics["rmse"], n_params=n_params, val_nasa_score=val_metrics["nasa_score"]
    )
