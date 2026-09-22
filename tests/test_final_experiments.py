import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from final_experiments import run_one  # noqa: E402


def _write_config(tmp_path, raw_dir, model_section):
    import yaml

    config = {
        "seed": 0,
        "deterministic": False,
        "data": {
            "subset": "DS02",
            "raw_dir": str(raw_dir),
            "window_length": 5,
            "downsample_factor": 1,
            "val_fraction": 0.34,
            "feature_groups": ["W", "X_s"],
        },
        "training": {"batch_size": 8, "epochs": 2, "lr": 1e-2, "early_stopping_patience": 10},
        "model": model_section,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


def test_run_one_baseline(synthetic_ncmapss_path, tmp_path):
    raw_dir = synthetic_ncmapss_path.parent
    config_path = _write_config(
        tmp_path, raw_dir, {"type": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8}
    )
    result = run_one(config_path, seed=0, device=torch.device("cpu"))

    assert result["seed"] == 0
    assert result["test_rmse"] >= 0
    assert isinstance(result["val_unit"], list) and len(result["val_unit"]) == 1


def test_run_one_pinn(synthetic_ncmapss_path, tmp_path):
    raw_dir = synthetic_ncmapss_path.parent
    config_path = _write_config(
        tmp_path,
        raw_dir,
        {
            "type": "pinn",
            "encoder": "cnn",
            "channels": [8],
            "kernel_size": 3,
            "embedding_dim": 8,
            "lambda_mono": 1.0,
            "lambda_health": 0.01,
        },
    )
    result = run_one(config_path, seed=0, device=torch.device("cpu"))
    assert result["test_rmse"] >= 0


def test_run_one_different_seeds_pick_different_val_units(synthetic_ncmapss_path, tmp_path):
    raw_dir = synthetic_ncmapss_path.parent
    config_path = _write_config(
        tmp_path, raw_dir, {"type": "cnn", "channels": [8], "kernel_size": 3, "embedding_dim": 8}
    )
    val_units = {
        tuple(run_one(config_path, seed=s, device=torch.device("cpu"))["val_unit"])
        for s in range(4)
    }
    # With only 3 dev units in the fixture, 4 different seeds should not
    # all land on the same held-out unit.
    assert len(val_units) > 1
