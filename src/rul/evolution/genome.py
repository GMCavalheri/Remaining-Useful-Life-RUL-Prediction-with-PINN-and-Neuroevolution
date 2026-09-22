"""The genome: what neuroevolution searches over.

Every individual is a full PINN configuration — architecture *and*
hyperparameters *and* the two physics-loss weights — encoded as a flat,
fixed-length set of independent genes. Each gene has its own domain and its
own sampling/mutation rule (a categorical choice, a continuous range, or a
log-uniform range for learning rate); this is the same flat-vector
representation used by most practical NAS/hyperparameter-search systems,
traded off against a richer (but much more complex) variable-length
encoding that could, say, evolve a variable number of layers.

Only the genes relevant to the sampled ``encoder`` actually affect the
built model (e.g. ``cnn_channels`` is ignored for an ``lstm`` individual) —
carrying the unused genes along anyway is what makes uniform crossover
between two different-encoder parents well-defined without special-casing.

Phase 4 hand-picked ``lambda_mono=1.0`` and (after a measured rebalance)
``lambda_health=0.02`` and found the health term still hurt on this tiny
dataset. Rather than keep hand-tuning, both weights are genes here — this
is the search the project plan always intended to replace manual tuning
with.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

# (gene name -> domain). "choice" domains are lists; "log_uniform" and
# "uniform" domains are (low, high) tuples.
_CHOICE_DOMAINS: dict[str, list] = {
    "encoder": ["mlp", "cnn", "lstm", "gru"],
    "embedding_dim": [16, 32, 64],
    "window_length": [10, 20, 30, 50],
    "mlp_hidden_dim": [32, 64, 128],
    "cnn_channels": [16, 32, 64],
    "cnn_kernel_size": [3, 5, 7],
    "rnn_hidden_size": [16, 32, 64],
    "rnn_num_layers": [1, 2],
}
_UNIFORM_DOMAINS: dict[str, tuple[float, float]] = {
    "dropout": (0.0, 0.3),
    "lambda_mono": (0.0, 2.0),
    "lambda_health": (0.0, 0.1),
}
_LOG_UNIFORM_DOMAINS: dict[str, tuple[float, float]] = {
    "lr": (1e-4, 1e-2),
}


@dataclass
class Genome:
    encoder: str
    embedding_dim: int
    window_length: int
    dropout: float
    lr: float
    lambda_mono: float
    lambda_health: float
    mlp_hidden_dim: int
    cnn_channels: int
    cnn_kernel_size: int
    rnn_hidden_size: int
    rnn_num_layers: int

    def to_model_cfg(self) -> dict[str, Any]:
        """The subset of genes relevant to ``self.encoder``, as a ``build_pinn`` config."""
        cfg = {
            "encoder": self.encoder,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "lambda_mono": self.lambda_mono,
            "lambda_health": self.lambda_health,
        }
        if self.encoder == "mlp":
            cfg["hidden_dims"] = (self.mlp_hidden_dim,)
        elif self.encoder == "cnn":
            cfg["channels"] = (self.cnn_channels,)
            cfg["kernel_size"] = self.cnn_kernel_size
        else:  # lstm / gru
            cfg["hidden_size"] = self.rnn_hidden_size
            cfg["num_layers"] = self.rnn_num_layers
        return cfg


def sample_genome(rng: np.random.Generator) -> Genome:
    values: dict[str, Any] = {}
    for name, choices in _CHOICE_DOMAINS.items():
        values[name] = choices[rng.integers(len(choices))]
    for name, (low, high) in _UNIFORM_DOMAINS.items():
        values[name] = float(rng.uniform(low, high))
    for name, (low, high) in _LOG_UNIFORM_DOMAINS.items():
        values[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return Genome(**values)


def mutate(genome: Genome, rng: np.random.Generator, rate: float = 0.2) -> Genome:
    """Independently resample each gene with probability ``rate``."""
    child = copy.deepcopy(genome)
    for name, choices in _CHOICE_DOMAINS.items():
        if rng.random() < rate:
            setattr(child, name, choices[rng.integers(len(choices))])
    for name, (low, high) in _UNIFORM_DOMAINS.items():
        if rng.random() < rate:
            setattr(child, name, float(rng.uniform(low, high)))
    for name, (low, high) in _LOG_UNIFORM_DOMAINS.items():
        if rng.random() < rate:
            setattr(child, name, float(np.exp(rng.uniform(np.log(low), np.log(high)))))
    return child


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover: each gene independently comes from one parent or the other."""
    values = {}
    for f in fields(Genome):
        source = parent_a if rng.random() < 0.5 else parent_b
        values[f.name] = getattr(source, f.name)
    return Genome(**values)
