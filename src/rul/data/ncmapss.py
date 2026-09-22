"""Loading N-CMAPSS ``.h5`` files.

N-CMAPSS ("New" C-MAPSS) is NASA's turbofan run-to-failure dataset generated
under realistic flight conditions (Arias Chao et al., 2021, "Aircraft Engine
Run-to-Failure Dataset under Real Flight Conditions for Prognostics and
Diagnostics", *Data*). Each ``N-CMAPSS_DSxx-yyy.h5`` file stores a fixed set
of HDF5 datasets, split into a NASA-defined ``dev`` (development/train) and
``test`` partition of *engine units* — the test units are held out at the
unit level, so there is no risk of a unit's cycles leaking across the split:

======  =========================================  =====================
Group   Meaning                                     Columns (``*_var``)
======  =========================================  =====================
``W``   Flight/scenario conditions                  alt, Mach, TRA, T2
``X_s`` Measured (physical) sensor readings          14 sensors, e.g. T24, T30, P30, Nf, Nc, ...
``X_v`` Virtual (model-based) sensor readings         14 sensors
``T``   Unobservable engine health parameters        varies by subset
``A``   Auxiliary: unit id, cycle, flight class, health state
``Y``   Target: Remaining Useful Life (cycles)        1 column
======  =========================================  =====================

Every array shares the same first (row) dimension: one row per 1 Hz sample
within a flight cycle. ``*_var`` datasets store the column names for the
corresponding array as fixed-length byte strings.

This module only *reads* that structure; it does no windowing or splitting
(see :mod:`rul.data.windowing`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# The five column-carrying arrays that each have a matching `<name>_var`
# dataset listing their column names, plus the scalar target `Y`.
_VAR_GROUPS = ("W", "X_s", "X_v", "T", "A")
_ALL_ARRAYS = _VAR_GROUPS + ("Y",)
_SPLITS = ("dev", "test")


@dataclass(frozen=True)
class NCMAPSSSplit:
    """One partition (``dev`` or ``test``) of an N-CMAPSS file.

    All arrays are aligned row-wise: row ``i`` of ``W``, ``X_s``, ``X_v``,
    ``T``, ``A`` and ``Y`` all describe the same 1 Hz sample.
    """

    W: np.ndarray
    X_s: np.ndarray
    X_v: np.ndarray
    T: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    columns: dict[str, list[str]]

    @property
    def unit_ids(self) -> np.ndarray:
        """Unique engine unit identifiers present in this split."""
        unit_col = self.columns["A"].index("unit")
        return np.unique(self.A[:, unit_col])


@dataclass(frozen=True)
class NCMAPSSFile:
    dev: NCMAPSSSplit
    test: NCMAPSSSplit
    path: Path


def _decode_var_names(dataset: h5py.Dataset) -> list[str]:
    """Decode an ``*_var`` dataset of fixed-length byte strings to str."""
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in dataset[:].ravel()]


def load_ncmapss(path: str | Path) -> NCMAPSSFile:
    """Load one N-CMAPSS ``.h5`` file into dev/test :class:`NCMAPSSSplit`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (with a hint to run the download step).
    KeyError
        If the file is missing an expected dataset (i.e. is not actually an
        N-CMAPSS file in the documented layout).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"N-CMAPSS file not found: {path}. Run `scripts/download_data.py` first."
        )

    with h5py.File(path, "r") as f:
        columns = {group: _decode_var_names(f[f"{group}_var"]) for group in _VAR_GROUPS}

        splits: dict[str, NCMAPSSSplit] = {}
        for split in _SPLITS:
            arrays = {name: np.asarray(f[f"{name}_{split}"]) for name in _ALL_ARRAYS}
            splits[split] = NCMAPSSSplit(
                W=arrays["W"],
                X_s=arrays["X_s"],
                X_v=arrays["X_v"],
                T=arrays["T"],
                A=arrays["A"],
                Y=arrays["Y"],
                columns=columns,
            )

    return NCMAPSSFile(dev=splits["dev"], test=splits["test"], path=path)
