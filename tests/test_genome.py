import numpy as np

from rul.evolution.genome import (
    _CHOICE_DOMAINS,
    _LOG_UNIFORM_DOMAINS,
    _UNIFORM_DOMAINS,
    crossover,
    mutate,
    sample_genome,
)


def test_sample_genome_within_domains():
    rng = np.random.default_rng(0)
    for _ in range(50):
        genome = sample_genome(rng)
        for name, choices in _CHOICE_DOMAINS.items():
            assert getattr(genome, name) in choices
        for name, (low, high) in _UNIFORM_DOMAINS.items():
            assert low <= getattr(genome, name) <= high
        for name, (low, high) in _LOG_UNIFORM_DOMAINS.items():
            assert low <= getattr(genome, name) <= high


def test_sample_genome_deterministic_given_seed():
    g1 = sample_genome(np.random.default_rng(42))
    g2 = sample_genome(np.random.default_rng(42))
    assert g1 == g2


def test_to_model_cfg_selects_only_relevant_genes():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    genome.encoder = "cnn"
    genome.cnn_channels = 32
    genome.cnn_kernel_size = 5

    cfg = genome.to_model_cfg()
    assert cfg["encoder"] == "cnn"
    assert cfg["channels"] == (32,)
    assert cfg["kernel_size"] == 5
    assert "hidden_dims" not in cfg
    assert "hidden_size" not in cfg


def test_to_model_cfg_mlp():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    genome.encoder = "mlp"
    genome.mlp_hidden_dim = 64
    cfg = genome.to_model_cfg()
    assert cfg["hidden_dims"] == (64,)


def test_to_model_cfg_rnn():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    genome.encoder = "lstm"
    genome.rnn_hidden_size = 32
    genome.rnn_num_layers = 2
    cfg = genome.to_model_cfg()
    assert cfg["hidden_size"] == 32
    assert cfg["num_layers"] == 2


def test_mutate_with_rate_zero_is_identity():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    mutated = mutate(genome, np.random.default_rng(1), rate=0.0)
    assert mutated == genome


def test_mutate_with_rate_one_changes_most_genes():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    mutated = mutate(genome, np.random.default_rng(1), rate=1.0)
    # With rate=1.0 every gene is resampled; at least the continuous ones
    # (near-zero chance of landing on the exact same float) should differ.
    assert mutated.lr != genome.lr
    assert mutated.dropout != genome.dropout


def test_mutate_does_not_modify_original():
    rng = np.random.default_rng(0)
    genome = sample_genome(rng)
    original = str(genome)
    mutate(genome, np.random.default_rng(1), rate=1.0)
    assert str(genome) == original


def test_crossover_gene_comes_from_one_parent_or_other():
    rng = np.random.default_rng(0)
    parent_a = sample_genome(np.random.default_rng(1))
    parent_b = sample_genome(np.random.default_rng(2))
    child = crossover(parent_a, parent_b, rng)

    for name in _CHOICE_DOMAINS:
        assert getattr(child, name) in (getattr(parent_a, name), getattr(parent_b, name))
