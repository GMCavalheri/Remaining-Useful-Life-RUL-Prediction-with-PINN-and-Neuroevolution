"""Feature standardization, fit on the training split only.

Given windows of shape ``(N, L, F)``, we standardize each of the ``F``
feature channels independently:

.. math::

    \\hat{x}_{i,\\ell,f} = \\frac{x_{i,\\ell,f} - \\mu_f}{\\sigma_f + \\varepsilon}

where :math:`\\mu_f, \\sigma_f` are the per-feature mean and standard
deviation computed over every ``(i, \\ell)`` position **of the training
windows only**. Fitting on train and applying to val/test (rather than
fitting on the pooled data) is the same unit-wise-leakage discipline as the
train/val/test split itself: the model must never have, even indirectly
through normalization statistics, information derived from validation or
test units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, X_train: np.ndarray, eps: float = 1e-8) -> "WindowScaler":
        """Fit per-feature mean/std from an ``(N, L, F)`` training array."""
        flat = X_train.reshape(-1, X_train.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / (self.std + self.eps)).astype(np.float32)
