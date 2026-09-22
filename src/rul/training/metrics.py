"""RMSE and the NASA asymmetric scoring function.

RMSE treats an early warning (predicting less RUL than truth) and a late
warning (predicting more RUL than truth) as equally bad. For a maintenance
decision that isn't true: predicting *more* life than an engine actually has
left means it fails before the scheduled maintenance — a safety incident.
Predicting *less* life than it actually has just means unnecessary early
maintenance — a cost, not a hazard. The NASA scoring function used across
the (N-)CMAPSS literature (Saxena, Goebel, Simon & Eklund, 2008,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation") encodes exactly that asymmetry.

For each sample, let :math:`d = \\widehat{\\text{RUL}} - \\text{RUL}`
(prediction minus truth):

.. math::

    s(d) =
    \\begin{cases}
        e^{-d/13} - 1 & d < 0 \\quad \\text{(early: predicted less life than truth)} \\\\
        e^{\\,d/10} - 1 & d \\geq 0 \\quad \\text{(late: predicted more life than truth)}
    \\end{cases}

Both branches are 0 at :math:`d=0` and grow away from it, but the ``late``
branch (denominator 10) grows faster than the ``early`` branch (denominator
13) for the same |d| — a late prediction of a given magnitude costs more
than an early one of the same magnitude. The dataset score is
:math:`\\sum_i s(d_i)`, which is what papers report; it is *not*
normalized by sample count, so it is only comparable across runs on the
same test set size (we also expose a ``mean`` reduction for that reason).
"""

from __future__ import annotations

import numpy as np


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(
    y_pred: np.ndarray, y_true: np.ndarray, reduction: str = "sum"
) -> float:
    """NASA asymmetric RUL scoring function (lower is better).

    Parameters
    ----------
    reduction:
        ``"sum"`` (the value reported in the literature) or ``"mean"``
        (per-sample average, comparable across differently sized test sets).
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")

    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    d = y_pred - y_true

    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum()) if reduction == "sum" else float(s.mean())
