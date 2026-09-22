"""Per-cycle aggregation, sliding-window construction, and unit-wise splitting.

N-CMAPSS is sampled at 1 Hz *within* each flight cycle. A cycle is one
flight, so it can contain **thousands to tens of thousands of raw rows**
(DS02's cycles average ~11,500 rows each). RUL, however, is a per-*cycle*
quantity: it does not change within a flight. Feeding a model raw 1 Hz rows
would mean "windows of 50 seconds" instead of "windows of 50 flights", and
even lightly downsampled, a single unit's ~11,000-row cycles keep the
row-level dataset enormous (DS02's 9 units alone produce hundreds of
thousands of highly redundant, overlapping windows if windowed at the row
level — enough to exhaust memory on an ordinary workstation).

So building model-ready data here is two stages:

1. **Per-cycle aggregation** (:func:`aggregate_by_cycle`): within each
   ``(unit, cycle)`` group, average every feature over that cycle's rows
   (after first keeping only every ``downsample_factor``-th raw row, purely
   as a speed/precision knob on the averaging — it does not change how many
   *cycles* come out). This turns "thousands of samples per flight" into
   "one representative operating point per flight", which is the natural
   granularity for a degradation trajectory that unfolds over tens to
   hundreds of cycles, not over seconds.
2. **Windowing** (:func:`make_windows`): slide a ``window_length``-cycle
   window over each unit's now-short per-cycle sequence, labelled with the
   RUL at the window's last cycle.

Splitting is done **unit-wise**: a unit's rows are 100% train, 100% val, or
100% test, never split across sets. Splitting by row instead would leak
information (adjacent rows/cycles of the same degradation trajectory are
nearly identical), silently inflating validation/test scores. N-CMAPSS files
ship NASA's own train/test unit partition (the ``dev``/``test`` split loaded
by :mod:`rul.data.ncmapss`); this module's :func:`split_units_train_val` is
used only to carve a validation set out of the ``dev`` units, by unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rul.data.ncmapss import NCMAPSSSplit


@dataclass(frozen=True)
class CycleData:
    """One row per ``(unit, cycle)``, aggregated from raw 1 Hz samples.

    Attributes mirror :class:`WindowedData` but without the window
    dimension: ``features[i]`` is the mean feature vector for
    ``(unit[i], cycle[i])``.
    """

    features: np.ndarray
    y: np.ndarray
    health: np.ndarray
    unit: np.ndarray
    cycle: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class WindowedData:
    """Model-ready windows built from one :class:`NCMAPSSSplit`.

    Attributes
    ----------
    X:
        Shape ``(n_windows, window_length, n_features)``, float32. Feature
        columns are the concatenation of the requested ``feature_groups``
        (e.g. flight conditions ``W`` + measured sensors ``X_s``), each
        value being that cycle's mean over its raw 1 Hz samples.
    y:
        Shape ``(n_windows,)``, float32. RUL at the window's last cycle.
    health:
        Shape ``(n_windows, n_health_params)``, float32. The ``T`` (health
        parameter) row at the window's last cycle — used as an auxiliary
        supervision target by the PINN's health head (Milestone 4), not as
        a model input.
    unit:
        Shape ``(n_windows,)``. Engine unit id each window belongs to.
    cycle:
        Shape ``(n_windows,)``. Flight cycle number at the window's last
        step, used to order a unit's windows when computing the
        monotonic-degradation physics loss.
    feature_names:
        Column name for each of the ``n_features`` columns of ``X``, in
        order.
    """

    X: np.ndarray
    y: np.ndarray
    health: np.ndarray
    unit: np.ndarray
    cycle: np.ndarray
    feature_names: list[str]


def aggregate_by_cycle(
    split: NCMAPSSSplit,
    *,
    feature_groups: tuple[str, ...] = ("W", "X_s"),
    downsample_factor: int = 1,
) -> CycleData:
    """Collapse each unit's raw 1 Hz rows to one mean row per flight cycle.

    ``downsample_factor`` thins the raw rows *before* averaging (a stride,
    not a filter) purely to cut compute on files with ~11k rows/cycle; it
    does not reduce the number of cycles produced, since a downsampled
    cycle group must still yield at least one row via a strided slice.
    """
    unit_col = split.columns["A"].index("unit")
    cycle_col = split.columns["A"].index("cycle")

    feature_names = [name for group in feature_groups for name in split.columns[group]]

    features_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    health_chunks: list[np.ndarray] = []
    unit_chunks: list[np.ndarray] = []
    cycle_chunks: list[np.ndarray] = []

    for uid in split.unit_ids:
        mask = split.A[:, unit_col] == uid
        # Rows for a unit are stored in time order already; sort defensively.
        order = np.argsort(split.A[mask, cycle_col], kind="stable")

        features_full = np.concatenate(
            [getattr(split, group)[mask][order] for group in feature_groups], axis=1
        )
        y_full = split.Y[mask][order].reshape(-1)
        health_full = split.T[mask][order]
        cycle_full = split.A[mask, cycle_col][order]

        features_ds = features_full[::downsample_factor]
        y_ds = y_full[::downsample_factor]
        health_ds = health_full[::downsample_factor]
        cycle_ds = cycle_full[::downsample_factor]

        # `cycle_ds` is sorted ascending (rows were sorted by cycle above,
        # and striding preserves order), so `return_index` gives each
        # cycle's first-occurrence row and `return_inverse`/`return_counts`
        # let us sum-then-divide into a per-cycle mean in one vectorized pass.
        uniq_cycles, first_idx, inverse, counts = np.unique(
            cycle_ds, return_index=True, return_inverse=True, return_counts=True
        )

        sums = np.zeros((len(uniq_cycles), features_ds.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, features_ds)
        feature_means = (sums / counts[:, None]).astype(np.float32)

        # RUL and health parameters are constant within a cycle in
        # N-CMAPSS, so the first occurrence is exact, not an approximation.
        y_agg = y_ds[first_idx].astype(np.float32)
        health_agg = health_ds[first_idx].astype(np.float32)

        features_chunks.append(feature_means)
        y_chunks.append(y_agg)
        health_chunks.append(health_agg)
        unit_chunks.append(np.full(len(uniq_cycles), uid))
        cycle_chunks.append(uniq_cycles)

    return CycleData(
        features=np.concatenate(features_chunks).astype(np.float32),
        y=np.concatenate(y_chunks),
        health=np.concatenate(health_chunks),
        unit=np.concatenate(unit_chunks),
        cycle=np.concatenate(cycle_chunks),
        feature_names=feature_names,
    )


def make_windows(
    split: NCMAPSSSplit,
    *,
    feature_groups: tuple[str, ...] = ("W", "X_s"),
    window_length: int = 50,
    stride: int = 1,
    downsample_factor: int = 1,
) -> WindowedData:
    """Aggregate to per-cycle rows, then build sliding windows for every unit.

    A unit with fewer cycles than ``window_length`` contributes no windows
    (it is skipped, not padded) — this can happen for short trajectories and
    is preferable to inventing padded data for a physical degradation signal.
    """
    cycles = aggregate_by_cycle(
        split, feature_groups=feature_groups, downsample_factor=downsample_factor
    )

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    health_chunks: list[np.ndarray] = []
    unit_chunks: list[np.ndarray] = []
    cycle_chunks: list[np.ndarray] = []

    for uid in split.unit_ids:
        mask = cycles.unit == uid
        features = cycles.features[mask]
        y = cycles.y[mask]
        health = cycles.health[mask]
        cycle = cycles.cycle[mask]

        n_cycles = features.shape[0]
        if n_cycles < window_length:
            continue

        for end in range(window_length - 1, n_cycles, stride):
            start = end - window_length + 1
            X_chunks.append(features[start : end + 1])
            y_chunks.append(y[end])
            health_chunks.append(health[end])
            unit_chunks.append(uid)
            cycle_chunks.append(cycle[end])

    if not X_chunks:
        n_features = len(cycles.feature_names)
        n_health = split.T.shape[1]
        return WindowedData(
            X=np.empty((0, window_length, n_features), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            health=np.empty((0, n_health), dtype=np.float32),
            unit=np.empty((0,), dtype=split.A.dtype),
            cycle=np.empty((0,), dtype=split.A.dtype),
            feature_names=cycles.feature_names,
        )

    return WindowedData(
        X=np.stack(X_chunks).astype(np.float32),
        y=np.asarray(y_chunks, dtype=np.float32),
        health=np.stack(health_chunks).astype(np.float32),
        unit=np.asarray(unit_chunks),
        cycle=np.asarray(cycle_chunks),
        feature_names=cycles.feature_names,
    )


def split_units_train_val(
    unit_ids: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split unique unit ids into train/val by unit.

    At least one unit is kept for validation whenever ``val_fraction > 0``
    and more than one unit is available, so a nonzero ``val_fraction`` never
    silently degenerates to an empty validation set.
    """
    units = np.array(sorted(set(unit_ids.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)

    n_val = int(round(len(units) * val_fraction))
    if val_fraction > 0 and n_val == 0 and len(units) > 1:
        n_val = 1

    val_units = shuffled[:n_val]
    train_units = shuffled[n_val:]
    return train_units, val_units
