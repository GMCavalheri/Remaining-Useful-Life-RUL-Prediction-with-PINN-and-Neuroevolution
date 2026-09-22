import numpy as np
import torch

from rul.seed import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a_np = np.random.rand(5)
    a_torch = torch.rand(5)

    set_seed(42)
    b_np = np.random.rand(5)
    b_torch = torch.rand(5)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)


def test_different_seeds_differ():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(1)
    b = np.random.rand(5)
    assert not np.allclose(a, b)
