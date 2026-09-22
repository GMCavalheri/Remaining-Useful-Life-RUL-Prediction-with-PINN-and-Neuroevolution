import torch

from rul.evolution.search import run_evolution


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


def test_run_evolution_smoke(synthetic_ncmapss_path, tmp_path):
    raw_dir = synthetic_ncmapss_path.parent
    output_dir = tmp_path / "evolution"

    history = run_evolution(
        _config(raw_dir),
        population_size=2,
        n_generations=2,
        fitness_epochs=2,
        seed=0,
        output_dir=output_dir,
        device=torch.device("cpu"),
    )

    assert len(history) == 2
    assert (output_dir / "evolution_checkpoint.json").exists()
    # RMSE should never be worse in later generations than generation 0's
    # best (elitist selection is supposed to never lose a good individual).
    assert history[-1]["best_val_rmse"] <= history[0]["best_val_rmse"] + 1e-6


def test_run_evolution_resumes_from_checkpoint(synthetic_ncmapss_path, tmp_path):
    raw_dir = synthetic_ncmapss_path.parent
    output_dir = tmp_path / "evolution"

    history_partial = run_evolution(
        _config(raw_dir), population_size=2, n_generations=1, fitness_epochs=2, seed=0,
        output_dir=output_dir, device=torch.device("cpu"),
    )
    assert len(history_partial) == 1

    # Re-running with a larger n_generations should RESUME (reuse gen 0's
    # recorded result) rather than starting over.
    history_full = run_evolution(
        _config(raw_dir), population_size=2, n_generations=3, fitness_epochs=2, seed=0,
        output_dir=output_dir, device=torch.device("cpu"),
    )
    assert len(history_full) == 3
    assert history_full[0] == history_partial[0]
