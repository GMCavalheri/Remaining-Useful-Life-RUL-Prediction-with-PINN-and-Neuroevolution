"""Physics-informed loss terms, as *soft* priors added on top of plain MSE.

These are not exact PDE residuals (N-CMAPSS's true degradation dynamics are
a proprietary simulator, not a closed-form equation we have access to).
Instead, both terms encode a physically-motivated inequality/consistency
that a good RUL predictor should approximately satisfy, and penalize its
violation. The combined PINN loss (:func:`make_pinn_loss`) is

.. math::

    \\mathcal{L} = \\mathcal{L}_{\\text{MSE}}
        + \\lambda_{\\text{mono}} \\, \\mathcal{L}_{\\text{mono}}
        + \\lambda_{\\text{health}} \\, \\mathcal{L}_{\\text{health}}

with :math:`\\lambda_{\\text{mono}} = 0` and/or :math:`\\lambda_{\\text{health}}
= 0` recovering the ablated variants (Phase 4's "no physics" / "one term
only" configs).
"""

from __future__ import annotations

from typing import Callable

import torch


def monotonic_degradation_loss(
    rul_pred: torch.Tensor, unit: torch.Tensor, cycle: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Penalize RUL predictions that *increase* later in the same unit's life.

    An engine's remaining life is, by definition, non-increasing in cycle
    number: two windows ``i, j`` from the *same* unit with
    ``cycle_i < cycle_j`` must satisfy
    :math:`\\widehat{\\text{RUL}}_i \\geq \\widehat{\\text{RUL}}_j`. This
    compares every same-unit pair within a batch (not just consecutive
    cycles — a shuffled ``DataLoader`` batch rarely contains adjacent
    cycles, but *any* correctly-ordered pair still constrains the model):

    .. math::

        \\mathcal{L}_{\\text{mono}} = \\frac{1}{|P|} \\sum_{(i,j) \\in P}
            \\max\\!\\left(0,\\ \\widehat{\\text{RUL}}_j - \\widehat{\\text{RUL}}_i\\right)

    where :math:`P = \\{(i,j) : \\text{unit}_i = \\text{unit}_j,\\ \\text{cycle}_i <
    \\text{cycle}_j\\}`. A pair contributes 0 whenever the prediction is
    already correctly ordered; only violations produce gradient (via
    ``relu``, the same "one-sided" penalty shape as a hinge loss).
    """
    same_unit = unit.unsqueeze(1) == unit.unsqueeze(0)  # (B, B)
    earlier = cycle.unsqueeze(1) < cycle.unsqueeze(0)  # [i,j]: cycle_i < cycle_j
    mask = same_unit & earlier

    # diff[i, j] = rul_pred_j - rul_pred_i; positive means a violation
    # (the later cycle j was predicted MORE remaining life than earlier i).
    diff = rul_pred.unsqueeze(0) - rul_pred.unsqueeze(1)
    violation = torch.relu(diff) * mask

    n_pairs = mask.sum()
    if n_pairs == 0:
        return rul_pred.sum() * 0.0  # zero, but still connected to the graph
    return violation.sum() / (n_pairs + eps)


def health_consistency_loss(
    health_pred: torch.Tensor,
    health_true: torch.Tensor,
    health_std: torch.Tensor,
    min_std: float = 1e-3,
) -> torch.Tensor:
    """Auxiliary supervised loss for the health-parameter head.

    N-CMAPSS's ``T`` arrays are the simulator's ground-truth degradation
    state (efficiency/flow modifiers per engine module) — information a
    real sensor cannot measure directly, but available here because the
    data is simulated. Supervising an auxiliary head to predict it is a
    standard way to inject physical structure into the *shared* encoder: if
    the embedding is also predictive of the underlying health state, not
    just of RUL, the encoder is pushed toward representing the actual
    degradation process rather than an arbitrary regression shortcut.

    Different health parameters live on very different natural scales (see
    :func:`rul.data.datamodule.build_datamodule`'s ``health_scaler``), so
    this standardizes by each parameter's *training-set* standard deviation
    before computing MSE — otherwise the largest-scale column would
    dominate the loss regardless of ``lambda_health``:

    .. math::

        \\mathcal{L}_{\\text{health}} = \\frac{1}{BH} \\sum_{i,h}
            \\left( \\frac{\\widehat{T}_{i,h} - T_{i,h}}
            {\\max(\\sigma_h,\\sigma_{\\min})} \\right)^2

    ``min_std`` clamps the denominator rather than merely adding a tiny
    epsilon: several N-CMAPSS health parameters (e.g. ``fan_eff_mod`` in
    DS02) are *exactly constant* across every training sample, i.e.
    :math:`\\sigma_h = 0`. An epsilon like ``1e-8`` "fixes" the division by
    zero but does not fix the loss: any nonzero prediction on that column
    (which is inevitable — the head has no way to know a column is
    constant) gets divided by ``1e-8`` and squared, producing a term ~1e16
    times larger than the RUL MSE and making the "physics" loss the only
    thing gradient descent optimizes. Clamping to a non-degenerate floor
    bounds a constant column's contribution instead of letting it explode.
    """
    denom = torch.clamp(health_std, min=min_std)
    diff = (health_pred - health_true) / denom
    return (diff**2).mean()


def make_pinn_loss(
    lambda_mono: float, lambda_health: float, health_std: torch.Tensor
) -> Callable:
    """Build a ``loss_fn`` matching :func:`rul.training.trainer.train_model`'s signature.

    Setting ``lambda_mono=0`` and/or ``lambda_health=0`` ablates that term
    (its loss is not even computed), which is how ``configs/pinn_*.yaml``'s
    ablation variants (no-physics / mono-only / health-only / full) are
    expressed.
    """

    def loss_fn(rul_pred, rul_true, health_pred, health_true, unit, cycle):
        loss = torch.nn.functional.mse_loss(rul_pred, rul_true)
        if lambda_mono > 0:
            loss = loss + lambda_mono * monotonic_degradation_loss(rul_pred, unit, cycle)
        if lambda_health > 0:
            loss = loss + lambda_health * health_consistency_loss(
                health_pred, health_true, health_std
            )
        return loss

    return loss_fn
