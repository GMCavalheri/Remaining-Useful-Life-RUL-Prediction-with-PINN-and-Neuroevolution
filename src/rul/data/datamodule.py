"""Ties loading, unit-wise splitting, windowing and scaling into DataLoaders.

The split hierarchy, top to bottom:

1. NASA's own ``dev``/``test`` unit partition (loaded by
   :mod:`rul.data.ncmapss`) — ``test`` is untouched until final evaluation.
2. ``dev`` units are further split, by unit, into train/val
   (:func:`rul.data.windowing.split_units_train_val`), using
   ``config["data"]["val_fraction"]``.
3. Each split is windowed independently (:func:`rul.data.windowing.make_windows`)
   so a window never spans across a train/val boundary.
4. A :class:`~rul.data.scaling.WindowScaler` is fit on the *train* windows
   only and applied to train/val/test alike.

``config["data"]["test_fraction"]`` from ``configs/base.yaml`` is accepted
but unused while NASA's own unit-level test partition is used as-is; it is
kept for a future custom (non-NASA-default) split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rul.data.ncmapss import load_ncmapss
from rul.data.scaling import WindowScaler
from rul.data.windowing import WindowedData, make_windows, split_units_train_val


class RULWindowDataset(Dataset):
    """Wraps a :class:`WindowedData` as a torch ``Dataset``.

    Each item is ``(X, y, health, unit, cycle)`` where ``X`` is
    ``(window_length, n_features)``, ``y`` is a scalar RUL, ``health`` is the
    health parameter vector at the window's last step (auxiliary target for
    the PINN's health head), and ``unit``/``cycle`` identify the window (its
    engine unit and last cycle number) — used by the monotonic-degradation
    physics loss to compare windows of the *same* unit at different points
    in its trajectory within a batch.
    """

    def __init__(self, data: WindowedData):
        self.X = torch.from_numpy(data.X)
        self.y = torch.from_numpy(data.y)
        self.health = torch.from_numpy(data.health)
        self.unit = torch.from_numpy(data.unit.astype(np.float32))
        self.cycle = torch.from_numpy(data.cycle.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.health[idx], self.unit[idx], self.cycle[idx]


@dataclass
class RULDataModule:
    train: RULWindowDataset
    val: RULWindowDataset
    test: RULWindowDataset
    scaler: WindowScaler
    health_scaler: WindowScaler
    feature_names: list[str]

    def loaders(
        self, batch_size: int, num_workers: int = 0
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            self.val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            self.test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader


def build_datamodule(config: dict[str, Any]) -> RULDataModule:
    """Build train/val/test datasets from a loaded YAML config (see ``configs/base.yaml``)."""
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    subset = data_cfg["subset"]
    h5_path = raw_dir / f"N-CMAPSS_{subset}-006.h5"

    ncmapss_file = load_ncmapss(h5_path)

    feature_groups = tuple(data_cfg.get("feature_groups", ("W", "X_s")))
    window_length = data_cfg["window_length"]
    downsample_factor = data_cfg["downsample_factor"]

    train_units, val_units = split_units_train_val(
        ncmapss_file.dev.unit_ids, data_cfg["val_fraction"], config["seed"]
    )

    def _windows_for_units(split, units: np.ndarray) -> WindowedData:
        unit_col = split.columns["A"].index("unit")
        mask = np.isin(split.A[:, unit_col], units)
        subset_split = type(split)(
            W=split.W[mask],
            X_s=split.X_s[mask],
            X_v=split.X_v[mask],
            T=split.T[mask],
            A=split.A[mask],
            Y=split.Y[mask],
            columns=split.columns,
        )
        return make_windows(
            subset_split,
            feature_groups=feature_groups,
            window_length=window_length,
            downsample_factor=downsample_factor,
        )

    train_windows = _windows_for_units(ncmapss_file.dev, train_units)
    val_windows = _windows_for_units(ncmapss_file.dev, val_units)
    test_windows = make_windows(
        ncmapss_file.test,
        feature_groups=feature_groups,
        window_length=window_length,
        downsample_factor=downsample_factor,
    )

    # Health targets (T, the auxiliary head's supervision target) are NOT
    # written back into the dataset scaled -- the PINN's physics loss
    # (Phase 4) standardizes them on the fly with health_scaler.std, since
    # different health parameters have wildly different natural scales
    # (e.g. HPT_eff_mod ~1e-2 vs. several columns that are ~0 for this
    # subset) and an unscaled MSE would let the largest-scale column
    # dominate the loss regardless of lambda_health.
    health_scaler = WindowScaler.fit(train_windows.health)

    scaler = WindowScaler.fit(train_windows.X)
    train_windows = WindowedData(
        X=scaler.transform(train_windows.X),
        y=train_windows.y,
        health=train_windows.health,
        unit=train_windows.unit,
        cycle=train_windows.cycle,
        feature_names=train_windows.feature_names,
    )
    val_windows = WindowedData(
        X=scaler.transform(val_windows.X),
        y=val_windows.y,
        health=val_windows.health,
        unit=val_windows.unit,
        cycle=val_windows.cycle,
        feature_names=val_windows.feature_names,
    )
    test_windows = WindowedData(
        X=scaler.transform(test_windows.X),
        y=test_windows.y,
        health=test_windows.health,
        unit=test_windows.unit,
        cycle=test_windows.cycle,
        feature_names=test_windows.feature_names,
    )

    return RULDataModule(
        train=RULWindowDataset(train_windows),
        val=RULWindowDataset(val_windows),
        test=RULWindowDataset(test_windows),
        scaler=scaler,
        health_scaler=health_scaler,
        feature_names=train_windows.feature_names,
    )
