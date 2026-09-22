import torch

from rul.evolution.fitness import DataModuleCache, evaluate_genome
from rul.evolution.genome import Genome


def _config(raw_dir):
    return {
        "seed": 0,
        "data": {
            "subset": "DS02",
            "raw_dir": str(raw_dir),
            "window_length": 5,
            "downsample_factor": 1,
            "val_fraction": 0.34,
            "feature_groups": ("W", "X_s"),
        },
        "training": {"batch_size": 8},
    }


def _genome(**overrides) -> Genome:
    defaults = dict(
        encoder="cnn",
        embedding_dim=8,
        window_length=5,
        dropout=0.0,
        lr=1e-2,
        lambda_mono=0.0,
        lambda_health=0.0,
        mlp_hidden_dim=16,
        cnn_channels=8,
        cnn_kernel_size=3,
        rnn_hidden_size=8,
        rnn_num_layers=1,
    )
    defaults.update(overrides)
    return Genome(**defaults)


def test_datamodule_cache_reuses_same_window_length(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    cache = DataModuleCache(_config(raw_dir))

    dm1 = cache.get(5)
    dm2 = cache.get(5)
    dm3 = cache.get(10)

    assert dm1 is dm2  # same window_length -> cached, not rebuilt
    assert dm1 is not dm3  # different window_length -> different datamodule


def test_evaluate_genome_returns_sane_fitness(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    cache = DataModuleCache(_config(raw_dir))
    genome = _genome()

    fitness = evaluate_genome(
        genome, cache, _config(raw_dir), device=torch.device("cpu"), epochs=2, seed=0
    )
    assert fitness.val_rmse >= 0
    assert fitness.n_params > 0
    assert fitness.val_nasa_score >= 0


def test_evaluate_genome_different_encoders_different_param_counts(synthetic_ncmapss_path):
    raw_dir = synthetic_ncmapss_path.parent
    cache = DataModuleCache(_config(raw_dir))

    mlp_fitness = evaluate_genome(
        _genome(encoder="mlp"), cache, _config(raw_dir), torch.device("cpu"), epochs=1, seed=0
    )
    lstm_fitness = evaluate_genome(
        _genome(encoder="lstm"), cache, _config(raw_dir), torch.device("cpu"), epochs=1, seed=0
    )
    assert mlp_fitness.n_params != lstm_fitness.n_params
