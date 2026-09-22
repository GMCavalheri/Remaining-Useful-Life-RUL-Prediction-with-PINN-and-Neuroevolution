"""Reproducibility utilities.

Every experiment in this project (a baseline run, a PINN training run, or a
single individual's fitness evaluation inside the evolutionary search) is
seeded explicitly. Determinism matters here for two distinct reasons:

1. Scientific validity: results are reported as mean +/- std over multiple
   seeds (see the project plan's evaluation milestone). If a "seed" did not
   actually fix the random state, that std would be meaningless.
2. Evolutionary search reproducibility: two runs of the same genome under the
   same seed must produce the same fitness, otherwise the search is
   optimizing noise rather than architecture/hyperparameter choices.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Parameters
    ----------
    seed:
        The seed value shared across all RNG sources.
    deterministic:
        If True, also request deterministic (non-benchmarked) CUDA/cuDNN
        kernels. This trades some throughput for exact reproducibility and
        should be on for final reported runs, and may be turned off for
        cheap neuroevolution fitness evaluations where wall-clock time
        matters more than bit-exact repeatability.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
