"""Run (or resume) the NSGA-II neuroevolution search.

Usage
-----
    uv run python scripts/evolve.py --config configs/evolve_small.yaml

Re-running the same command resumes from ``<output_dir>/evolution_checkpoint.json``
if one exists, instead of starting over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rul.config import load_config
from rul.evolution.search import run_evolution
from rul.seed import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], deterministic=config.get("deterministic", True))
    device = get_device()

    evo_cfg = config["evolution"]
    history = run_evolution(
        config,
        population_size=evo_cfg["population_size"],
        n_generations=evo_cfg["n_generations"],
        fitness_epochs=evo_cfg["fitness_epochs"],
        seed=config["seed"],
        output_dir=Path(config.get("output_dir", "results")) / "evolution",
        device=device,
        mutation_rate=evo_cfg.get("mutation_rate", 0.2),
    )
    print(f"Done. {len(history)} generations recorded.")


if __name__ == "__main__":
    main()
