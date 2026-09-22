import torch

from rul.physics.losses import health_consistency_loss, make_pinn_loss, monotonic_degradation_loss


def test_monotonic_loss_zero_when_already_monotonic():
    # unit 1: cycle 1 -> RUL 20, cycle 2 -> RUL 10 (correctly decreasing)
    rul_pred = torch.tensor([20.0, 10.0])
    unit = torch.tensor([1.0, 1.0])
    cycle = torch.tensor([1.0, 2.0])
    loss = monotonic_degradation_loss(rul_pred, unit, cycle)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_monotonic_loss_positive_when_violated():
    # unit 1: cycle 1 -> RUL 10, cycle 2 -> RUL 20 (WRONG: increased)
    rul_pred = torch.tensor([10.0, 20.0])
    unit = torch.tensor([1.0, 1.0])
    cycle = torch.tensor([1.0, 2.0])
    loss = monotonic_degradation_loss(rul_pred, unit, cycle)
    assert torch.isclose(loss, torch.tensor(10.0))  # relu(20-10)/1 pair


def test_monotonic_loss_ignores_different_units():
    # No same-unit pair exists -> zero loss regardless of RUL values.
    rul_pred = torch.tensor([10.0, 20.0])
    unit = torch.tensor([1.0, 2.0])
    cycle = torch.tensor([1.0, 2.0])
    loss = monotonic_degradation_loss(rul_pred, unit, cycle)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_monotonic_loss_gradient_flows_even_when_no_pairs():
    rul_pred = torch.tensor([10.0, 20.0], requires_grad=True)
    unit = torch.tensor([1.0, 2.0])
    cycle = torch.tensor([1.0, 2.0])
    loss = monotonic_degradation_loss(rul_pred, unit, cycle)
    loss.backward()  # should not raise
    assert rul_pred.grad is not None


def test_monotonic_loss_averages_over_multiple_violating_pairs():
    # unit 1, 3 cycles, RUL predictions strictly increasing (all pairs violate).
    rul_pred = torch.tensor([0.0, 10.0, 20.0])
    unit = torch.tensor([1.0, 1.0, 1.0])
    cycle = torch.tensor([1.0, 2.0, 3.0])
    loss = monotonic_degradation_loss(rul_pred, unit, cycle)
    # pairs (0,1): relu(10-0)=10; (0,2): relu(20-0)=20; (1,2): relu(20-10)=10
    expected = (10 + 20 + 10) / 3
    assert torch.isclose(loss, torch.tensor(expected))


def test_health_consistency_loss_matches_manual_computation():
    health_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    health_true = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    health_std = torch.tensor([1.0, 2.0])
    loss = health_consistency_loss(health_pred, health_true, health_std)
    # ((1/1)^2 + (2/2)^2 + (3/1)^2 + (4/2)^2) / 4 = (1+1+9+4)/4
    expected = (1 + 1 + 9 + 4) / 4
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_health_consistency_loss_clamps_zero_std_columns():
    """A health parameter that is exactly constant in training data (std=0,
    e.g. DS02's fan_eff_mod) must not blow up the loss for an arbitrary
    prediction on that column -- min_std clamps the denominator instead of
    dividing by (near) zero."""
    health_pred = torch.tensor([[5.0]])  # arbitrary nonzero prediction
    health_true = torch.tensor([[0.0]])  # the column's one observed (constant) value
    health_std = torch.tensor([0.0])  # zero variance in training data

    loss = health_consistency_loss(health_pred, health_true, health_std, min_std=1e-3)
    expected = (5.0 / 1e-3) ** 2  # clamped denominator, not (5.0 / 1e-8)**2
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-4)
    assert loss.item() < 1e8  # sane magnitude, not ~1e16 as with a raw epsilon


def test_make_pinn_loss_reduces_to_mse_when_lambdas_zero():
    loss_fn = make_pinn_loss(lambda_mono=0.0, lambda_health=0.0, health_std=torch.tensor([1.0]))
    rul_pred = torch.tensor([10.0, 20.0])  # violates monotonicity, but lambda_mono=0
    rul_true = torch.tensor([15.0, 15.0])
    health_pred = torch.tensor([[100.0]])  # would be huge health loss if it counted
    health_true = torch.tensor([[0.0]])
    unit = torch.tensor([1.0, 1.0])
    cycle = torch.tensor([1.0, 2.0])

    loss = loss_fn(rul_pred, rul_true, health_pred, health_true, unit, cycle)
    expected_mse = torch.nn.functional.mse_loss(rul_pred, rul_true)
    assert torch.isclose(loss, expected_mse)


def test_make_pinn_loss_adds_physics_terms_when_enabled():
    loss_fn = make_pinn_loss(lambda_mono=1.0, lambda_health=0.0, health_std=torch.tensor([1.0]))
    rul_pred = torch.tensor([10.0, 20.0])  # violates monotonicity
    rul_true = torch.tensor([15.0, 15.0])
    unit = torch.tensor([1.0, 1.0])
    cycle = torch.tensor([1.0, 2.0])

    loss_with_mono = loss_fn(rul_pred, rul_true, None, None, unit, cycle)
    expected_mse = torch.nn.functional.mse_loss(rul_pred, rul_true)
    assert loss_with_mono > expected_mse
