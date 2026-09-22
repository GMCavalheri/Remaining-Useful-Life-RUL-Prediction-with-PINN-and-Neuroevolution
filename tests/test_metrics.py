import numpy as np

from rul.training.metrics import nasa_score, rmse


def test_rmse_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 2.0, 2.0])
    # errors: -1, 0, 1 -> squared: 1, 0, 1 -> mean 2/3 -> sqrt
    assert np.isclose(rmse(y_pred, y_true), np.sqrt(2.0 / 3.0))


def test_nasa_score_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert nasa_score(y, y) == 0.0


def test_nasa_score_penalizes_late_more_than_early():
    y_true = np.array([50.0])
    # "late": predicted MORE life than truth (dangerous - engine fails before expected)
    late = nasa_score(np.array([60.0]), y_true)
    # "early": predicted LESS life than truth (conservative)
    early = nasa_score(np.array([40.0]), y_true)
    assert late > early > 0


def test_nasa_score_matches_hand_computed_formula():
    y_pred = np.array([55.0, 45.0])
    y_true = np.array([50.0, 50.0])
    # d = y_pred - y_true = [5, -5]
    expected = (np.exp(5 / 10) - 1) + (np.exp(5 / 13) - 1)
    assert np.isclose(nasa_score(y_pred, y_true, reduction="sum"), expected)


def test_nasa_score_mean_vs_sum_reduction():
    y_pred = np.array([55.0, 45.0])
    y_true = np.array([50.0, 50.0])
    total = nasa_score(y_pred, y_true, reduction="sum")
    mean = nasa_score(y_pred, y_true, reduction="mean")
    assert np.isclose(mean, total / 2)


def test_nasa_score_invalid_reduction_raises():
    import pytest

    with pytest.raises(ValueError):
        nasa_score(np.array([1.0]), np.array([1.0]), reduction="bogus")
