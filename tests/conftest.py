"""Shared test fixtures.

Real N-CMAPSS files are multi-GB, so unit tests never touch one directly.
Instead, :func:`synthetic_ncmapss_path` writes a tiny ``.h5`` file with the
exact same group/column layout (see ``rul.data.ncmapss``) so the loading,
windowing and splitting code is exercised end to end without the real
dataset being present (e.g. in CI).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

W_COLS = ["alt", "Mach", "TRA", "T2"]
XS_COLS = ["T24", "T30", "P30"]
XV_COLS = ["Wf", "Nf_R"]
T_COLS = ["HPC_eff", "HPC_flow"]
A_COLS = ["unit", "cycle", "Fc", "hs"]


def _make_split_arrays(rng: np.random.Generator, unit_lengths: dict[int, int]):
    """Build row-aligned arrays for a set of units, each degrading linearly."""
    W, X_s, X_v, T, A, Y = [], [], [], [], [], []
    for unit, n_rows in unit_lengths.items():
        cycles = np.repeat(np.arange(1, n_rows // 5 + 2), 5)[:n_rows]
        rul = (cycles.max() - cycles).astype(np.float32)

        W.append(rng.normal(size=(n_rows, len(W_COLS))).astype(np.float32))
        X_s.append(rng.normal(size=(n_rows, len(XS_COLS))).astype(np.float32))
        X_v.append(rng.normal(size=(n_rows, len(XV_COLS))).astype(np.float32))
        # Health parameters decay monotonically with cycle (toy physics).
        health = 1.0 - 0.01 * cycles[:, None] * np.ones((1, len(T_COLS)), dtype=np.float32)
        T.append(health.astype(np.float32))
        A.append(
            np.stack(
                [
                    np.full(n_rows, unit, dtype=np.float32),
                    cycles.astype(np.float32),
                    np.ones(n_rows, dtype=np.float32),
                    np.ones(n_rows, dtype=np.float32),
                ],
                axis=1,
            )
        )
        Y.append(rul.reshape(-1, 1))

    return (
        np.concatenate(W),
        np.concatenate(X_s),
        np.concatenate(X_v),
        np.concatenate(T),
        np.concatenate(A),
        np.concatenate(Y),
    )


@pytest.fixture
def synthetic_ncmapss_path(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "N-CMAPSS_DS02-006.h5"

    dev = _make_split_arrays(rng, {1: 120, 2: 150, 3: 100})
    test = _make_split_arrays(rng, {11: 80})

    with h5py.File(path, "w") as f:
        groups = [("W", W_COLS), ("X_s", XS_COLS), ("X_v", XV_COLS), ("T", T_COLS), ("A", A_COLS)]
        for group, cols in groups:
            f.create_dataset(f"{group}_var", data=np.array(cols, dtype="S32"))

        for name, arr in zip(["W", "X_s", "X_v", "T", "A", "Y"], dev):
            f.create_dataset(f"{name}_dev", data=arr)
        for name, arr in zip(["W", "X_s", "X_v", "T", "A", "Y"], test):
            f.create_dataset(f"{name}_test", data=arr)

    return path
