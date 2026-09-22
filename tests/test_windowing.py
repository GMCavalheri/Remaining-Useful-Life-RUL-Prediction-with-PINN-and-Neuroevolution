import numpy as np

from rul.data.ncmapss import load_ncmapss
from rul.data.windowing import aggregate_by_cycle, make_windows, split_units_train_val


def test_aggregate_by_cycle_one_row_per_unit_cycle(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)
    cycles = aggregate_by_cycle(f.dev, feature_groups=("W", "X_s"))

    # unit 1 has 120 rows in groups of 5 -> 24 cycles; same logic for others.
    n_cycles_per_unit = {
        uid: (cycles.unit == uid).sum() for uid in np.unique(cycles.unit)
    }
    assert n_cycles_per_unit == {1.0: 24, 2.0: 30, 3.0: 20}
    assert cycles.features.shape[1] == 4 + 3  # W cols + X_s cols


def test_aggregate_by_cycle_matches_manual_mean(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)
    cycles = aggregate_by_cycle(f.dev, feature_groups=("W",))

    unit_col = f.dev.columns["A"].index("unit")
    cycle_col = f.dev.columns["A"].index("cycle")
    raw_mask = (f.dev.A[:, unit_col] == 1.0) & (f.dev.A[:, cycle_col] == 1.0)
    expected_mean = f.dev.W[raw_mask].mean(axis=0)

    agg_mask = (cycles.unit == 1.0) & (cycles.cycle == 1.0)
    np.testing.assert_allclose(cycles.features[agg_mask][0], expected_mean, rtol=1e-5)


def test_make_windows_shapes(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)
    windows = make_windows(
        f.dev, feature_groups=("W", "X_s"), window_length=10, downsample_factor=1
    )

    assert windows.X.shape[1] == 10
    assert windows.X.shape[2] == 4 + 3  # W cols + X_s cols
    assert windows.feature_names == ["alt", "Mach", "TRA", "T2", "T24", "T30", "P30"]
    assert windows.X.shape[0] == windows.y.shape[0] == windows.unit.shape[0]
    assert windows.health.shape[1] == 2
    # unit cycles: 24, 30, 20 with window_length=10 -> (24-9)+(30-9)+(20-9) windows
    assert windows.X.shape[0] == 15 + 21 + 11


def test_make_windows_skips_units_with_too_few_cycles(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)
    # every unit has well under 1000 cycles after aggregation.
    windows = make_windows(f.dev, window_length=1000)
    assert windows.X.shape[0] == 0
    assert windows.X.shape[1:] == (1000, 4 + 3)


def test_windows_label_is_nonnegative_and_finite(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)
    windows = make_windows(f.dev, feature_groups=("W",), window_length=5, downsample_factor=1)
    assert np.all(np.isfinite(windows.y))
    assert np.all(windows.y >= 0)


def test_downsample_factor_preserves_cycle_count_but_changes_values(synthetic_ncmapss_path):
    """downsample_factor thins the *raw rows averaged per cycle*, not the
    number of cycles — each 5-row cycle group in the fixture always yields
    exactly one surviving row at any stride <= 5, so window *count* is
    unchanged while the averaged feature *values* differ (mean of 5 samples
    vs. a single sample)."""
    f = load_ncmapss(synthetic_ncmapss_path)
    full = make_windows(f.dev, window_length=5, downsample_factor=1)
    down = make_windows(f.dev, window_length=5, downsample_factor=5)

    assert down.X.shape[0] == full.X.shape[0]
    assert not np.allclose(down.X, full.X)


def test_split_units_train_val_no_overlap_and_covers_all():
    units = np.array([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
    train, val = split_units_train_val(units, val_fraction=0.3, seed=0)

    assert set(train.tolist()).isdisjoint(set(val.tolist()))
    assert set(train.tolist()) | set(val.tolist()) == set(units.tolist())
    assert len(val) == 2  # round(8 * 0.3) == 2


def test_split_units_train_val_deterministic():
    units = np.arange(20)
    train1, val1 = split_units_train_val(units, 0.25, seed=7)
    train2, val2 = split_units_train_val(units, 0.25, seed=7)
    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)


def test_split_units_nonzero_fraction_never_empty_val():
    units = np.array([1, 2])
    _, val = split_units_train_val(units, val_fraction=0.01, seed=0)
    assert len(val) == 1
