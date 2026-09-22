"""Plain-MSE training loop with early stopping, shared by every baseline.

The training objective for Phase 3's baselines is ordinary regression:

.. math::

    \\mathcal{L}_{\\text{MSE}} = \\frac{1}{B} \\sum_{i=1}^{B}
        \\left( \\widehat{\\text{RUL}}_i - \\text{RUL}_i \\right)^2

Phase 4's PINN reuses this same loop with an extended loss (MSE plus the
physics penalty terms), so the loop itself takes a ``loss_fn`` rather than
hard-coding MSE, keeping the two milestones' code shared instead of
duplicated.

Early stopping watches validation RMSE (not the training loss) so that a
model which starts overfitting — training loss still falling, but the model
no longer generalizing to unseen units — is caught and its best-so-far
weights are what gets returned/evaluated, not whatever the last epoch
happened to produce.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from rul.training.metrics import nasa_score, rmse


@dataclass
class TrainResult:
    model: nn.Module
    best_val_rmse: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)


def _default_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, _health_pred, _health_true
) -> torch.Tensor:
    return nn.functional.mse_loss(y_pred, y_true)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for X, y, _health in loader:
        X = X.to(device)
        y_pred = model(X)
        preds.append(y_pred.cpu())
        targets.append(y)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return {
        "rmse": rmse(y_pred, y_true),
        "nasa_score": nasa_score(y_pred, y_true, reduction="mean"),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    device: torch.device,
    loss_fn: Callable = _default_loss,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y, health in train_loader:
            X, y, health = X.to(device), y.to(device), health.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y, None, health)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        train_loss = sum(train_losses) / len(train_losses)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model, best_val_rmse=best_val_rmse, best_epoch=best_epoch, history=history
    )
