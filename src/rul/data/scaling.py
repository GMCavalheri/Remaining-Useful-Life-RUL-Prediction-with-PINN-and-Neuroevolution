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
        """Fit per-feature mean/std from an ``(N, L, F)`` training array.

        ``N == 0`` is a real, reachable case here (not just a hypothetical
        edge case): the neuroevolution search (Phase 5) treats
        ``window_length`` as a gene, and a sampled value can exceed every
        training unit's cycle count, leaving zero windows (see
        ``rul.data.windowing.make_windows``'s per-unit skip-if-too-short
        behavior). ``np.mean``/``np.std`` of an empty array silently
        produce ``NaN`` with only a runtime warning -- which would then
        poison every downstream computation without ever raising. Falling
        back to identity statistics (mean 0, std 1) instead keeps
        ``transform`` a well-defined no-op; the genome that produced zero
        windows still can't train (see
        ``rul.evolution.fitness.evaluate_genome``'s explicit empty-dataset
        check), but it fails loudly and locally there, not via silent NaNs
        three functions away.
        """
        flat = X_train.reshape(-1, X_train.shape[-1])
        n_features = flat.shape[-1]
        if flat.shape[0] == 0:
            return cls(mean=np.zeros(n_features), std=np.ones(n_features), eps=eps)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / (self.std + self.eps)).astype(np.float32)
