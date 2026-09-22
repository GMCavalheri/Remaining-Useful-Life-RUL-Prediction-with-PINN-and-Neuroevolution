# Remaining Useful Life Prediction with PINN and Neuroevolution

Predicting the Remaining Useful Life (RUL) of turbofan engines on NASA's
[N-CMAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
dataset with a Physics-Informed Neural Network (PINN), whose architecture and
hyperparameters are tuned by a neuroevolutionary (NSGA-II) search.

## Project status

Work in progress, built incrementally in milestones. See
[`docs` / commit history] for the running log; each milestone's commit
message explains the reasoning behind it.

## Approach

- **Dataset:** N-CMAPSS (turbofan degradation trajectories), starting with
  subset DS02.
- **Models:** standard sequence baselines (MLP / 1D-CNN / LSTM-GRU) trained
  with plain regression loss, compared against a PINN that adds soft
  physics-informed penalties (monotonic degradation, health-parameter
  consistency).
- **Neuroevolution:** an NSGA-II search over encoder type, depth, width,
  window length, learning rate, and physics-loss weights, optimizing
  validation RMSE and the NASA asymmetric scoring function.
- **Evaluation:** RMSE and NASA score on held-out engine units (split at the
  unit level to avoid data leakage), reported as mean +/- std over multiple
  seeds.

## Repository layout

```
src/rul/
  data/       # N-CMAPSS loading, windowing, unit-wise splitting
  models/     # encoders and the PINN head
  physics/    # physics-informed loss terms
  training/   # training loop and metrics
  evolution/  # genome, operators, NSGA-II search
  baselines/  # non-physics-informed reference models
scripts/      # CLI entry points (train / evolve / evaluate / report)
configs/      # YAML experiment configs
tests/        # unit tests
```

## Setup

```bash
uv sync --extra dev --extra evolution
```

## Running tests

```bash
make test
```

## Reproducing an experiment

```bash
uv run python scripts/train.py --config configs/<experiment>.yaml
```
