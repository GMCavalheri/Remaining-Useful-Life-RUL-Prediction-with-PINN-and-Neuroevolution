import numpy as np

from rul.data.datamodule import build_datamodule


def _config(raw_dir):
    return {
        "seed": 0,
        "data": {
            "subset": "DS02",
            "raw_dir": str(raw_dir),
            "window_length": 10,
            "downsample_factor": 1,
            "val_fraction": 0.34,  # 1 of 3 dev units -> val
            "feature_groups": ("W", "X_s"),
        },
    }


def test_build_datamodule_unit_wise_no_leakage(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))

    train_units = set(dm.train.unit.tolist())
    val_units = set(dm.val.unit.tolist())
    test_units = set(dm.test.unit.tolist())

    assert train_units.isdisjoint(val_units)
    # test units come from NASA's own test partition (unit 11), always
    # disjoint from dev (units 1,2,3) by construction of the fixture.
    assert test_units.isdisjoint(train_units | val_units)
    assert test_units == {11.0}


def test_scaler_fit_only_on_train(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))

    # Scaled train features should be ~standardized; scaler stats came from
    # train only, so this is a meaningful check (not tautological for val/test).
    train_X = dm.train.X.numpy()
    flat = train_X.reshape(-1, train_X.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-2)


def test_dataloaders_yield_batches(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))
    train_loader, val_loader, test_loader = dm.loaders(batch_size=8)

    X, y, health, unit, cycle = next(iter(train_loader))
    assert X.shape[1] == 10
    assert X.shape[2] == len(dm.feature_names)
    assert y.shape[0] == X.shape[0]
    assert health.shape[0] == X.shape[0]
    assert unit.shape[0] == X.shape[0]
    assert cycle.shape[0] == X.shape[0]


def test_health_scaler_fit_only_on_train(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    dm = build_datamodule(_config(raw_dir))

    train_health = dm.train.health.numpy()
    manual_mean = train_health.mean(axis=0)
    manual_std = train_health.std(axis=0)
    np.testing.assert_allclose(dm.health_scaler.mean, manual_mean, rtol=1e-5)
    np.testing.assert_allclose(dm.health_scaler.std, manual_std, rtol=1e-5)

    # health arrays themselves stay RAW (unscaled) in the dataset -- the
    # scaler is exposed for the physics loss to standardize on the fly.
    assert not np.allclose(train_health.mean(axis=0), 0.0, atol=1e-2)
