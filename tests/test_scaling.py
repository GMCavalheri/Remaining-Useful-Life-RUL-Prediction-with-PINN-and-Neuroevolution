import numpy as np

from rul.data.scaling import WindowScaler


def test_scaler_fit_transform_train_has_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(200, 10, 3)).astype(np.float32)

    scaler = WindowScaler.fit(X)
    X_scaled = scaler.transform(X)

    flat = X_scaled.reshape(-1, 3)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-2)


def test_scaler_applies_train_statistics_to_other_splits():
    X_train = np.zeros((10, 5, 2), dtype=np.float32)
    X_train[..., 0] = 1.0  # mean 1, std 0
    X_train[..., 1] = 3.0

    scaler = WindowScaler.fit(X_train)

    X_val = np.ones((4, 5, 2), dtype=np.float32) * 2.0
    X_val_scaled = scaler.transform(X_val)

    # Uses TRAIN mean/std, not val's own — val's own mean would give 0 here.
    expected_channel0 = (2.0 - 1.0) / (0.0 + scaler.eps)
    assert np.isclose(X_val_scaled[0, 0, 0], expected_channel0, rtol=1e-3)
