"""The NSGA-II generational loop tying genome + fitness + selection together.

Each generation:

1. **Vary**: produce ``population_size`` offspring from the current
   population via binary tournament selection (prefer better Pareto rank,
   break ties by crowding distance — see :mod:`rul.evolution.nsga`) +
   uniform crossover + per-gene mutation.
2. **Evaluate**: train every offspring for the fitness epoch budget
   (:func:`rul.evolution.fitness.evaluate_genome`).
3. **Select**: pool parents + offspring (``2N`` individuals — this is the
   elitist :math:`(\\mu+\\lambda)` scheme NSGA-II uses, so a strong parent is
   never lost just because its offspring happened to be worse) and keep the
   best ``population_size`` by :func:`rul.evolution.nsga.select_next_generation`.

The loop checkpoints after every generation (genomes + their fitness +
generation index) so a long search can be resumed rather than restarted
after an interruption.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rul.evolution.fitness import DataModuleCache, FitnessResult, evaluate_genome
from rul.evolution.genome import Genome, crossover, mutate, sample_genome
from rul.evolution.nsga import crowding_distance, fast_non_dominated_sort, select_next_generation

Individual = tuple[Genome, FitnessResult]


def _rank_and_crowding(objectives: list[tuple[float, ...]]) -> tuple[list[int], np.ndarray]:
    fronts = fast_non_dominated_sort(objectives)
    ranks = [0] * len(objectives)
    distances = np.zeros(len(objectives))
    for rank, front in enumerate(fronts):
        front_objectives = [objectives[i] for i in front]
        front_distances = crowding_distance(front_objectives)
        for i, idx in enumerate(front):
            ranks[idx] = rank
            distances[idx] = front_distances[i]
    return ranks, distances


def _tournament_select(
    population: list[Individual], ranks: list[int], distances: np.ndarray, rng: np.random.Generator
) -> Genome:
    i, j = rng.integers(len(population), size=2)
    if ranks[i] < ranks[j] or (ranks[i] == ranks[j] and distances[i] > distances[j]):
        return population[i][0]
    return population[j][0]


def _objectives(population: list[Individual]) -> list[tuple[float, float]]:
    return [(fitness.val_rmse, fitness.n_params) for _, fitness in population]


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "evolution_checkpoint.json"


def _save_checkpoint(
    output_dir: Path, generation: int, population: list[Individual], history: list[dict]
) -> None:
    payload = {
        "generation": generation,
        "population": [
            {"genome": asdict(genome), "fitness": asdict(fitness)} for genome, fitness in population
        ],
        "history": history,
    }
    _checkpoint_path(output_dir).write_text(json.dumps(payload, indent=2))


def _load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_evolution(
    config: dict[str, Any],
    *,
    population_size: int,
    n_generations: int,
    fitness_epochs: int,
    seed: int,
    output_dir: Path,
    device: torch.device,
    mutation_rate: float = 0.2,
) -> list[dict]:
    """Run (or resume) an NSGA-II search; returns the per-generation history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dm_cache = DataModuleCache(config)

    checkpoint = _load_checkpoint(output_dir)
    if checkpoint is not None:
        start_generation = checkpoint["generation"] + 1
        population = [
            (Genome(**ind["genome"]), FitnessResult(**ind["fitness"]))
            for ind in checkpoint["population"]
        ]
        history = checkpoint["history"]
        print(f"Resuming from checkpoint at generation {checkpoint['generation']}")
    else:
        start_generation = 0
        genomes = [sample_genome(rng) for _ in range(population_size)]
        fitnesses = [
            evaluate_genome(g, dm_cache, config, device, fitness_epochs, seed) for g in genomes
        ]
        population = list(zip(genomes, fitnesses))
        history = []
        _record_and_checkpoint(output_dir, 0, population, history)
        start_generation = 1

    for generation in range(start_generation, n_generations):
        objectives = _objectives(population)
        ranks, distances = _rank_and_crowding(objectives)

        offspring_genomes = []
        for _ in range(population_size):
            parent_a = _tournament_select(population, ranks, distances, rng)
            parent_b = _tournament_select(population, ranks, distances, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, rate=mutation_rate)
            offspring_genomes.append(child)

        offspring_fitness = [
            evaluate_genome(g, dm_cache, config, device, fitness_epochs, seed + generation)
            for g in offspring_genomes
        ]
        offspring = list(zip(offspring_genomes, offspring_fitness))

        combined = population + offspring
        combined_objectives = _objectives(combined)
        selected_indices = select_next_generation(combined_objectives, population_size)
        population = [combined[i] for i in selected_indices]

        _record_and_checkpoint(output_dir, generation, population, history)

    return history


def _record_and_checkpoint(
    output_dir: Path, generation: int, population: list[Individual], history: list[dict]
) -> None:
    rmses = [fitness.val_rmse for _, fitness in population]
    n_params = [fitness.n_params for _, fitness in population]
    history.append(
        {
            "generation": generation,
            "best_val_rmse": min(rmses),
            "mean_val_rmse": sum(rmses) / len(rmses),
            "min_n_params": min(n_params),
            "max_n_params": max(n_params),
        }
    )
    _save_checkpoint(output_dir, generation, population, history)
    print(
        f"Generation {generation}: best val RMSE {min(rmses):.3f}, "
        f"pop size {len(population)}, param range [{min(n_params)}, {max(n_params)}]"
    )
