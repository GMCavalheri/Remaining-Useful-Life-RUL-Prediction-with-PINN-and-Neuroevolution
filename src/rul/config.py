"""Minimal YAML-backed configuration loading.

Every script (``scripts/train.py``, ``scripts/evolve.py``, ...) takes a
``--config path/to/file.yaml`` argument rather than a pile of CLI flags. This
keeps every experiment's full configuration as a single artifact that can be
committed, diffed, and cited in the results tables (milestone 7 of the
project plan: "reproduction commands").

Configs are plain nested dicts loaded from YAML; ``load_config`` additionally
supports a single-level ``defaults: <path>`` key so specific configs (e.g.
``configs/evolve_small.yaml``) can inherit from a base config and override
only what differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``defaults:`` base path.

    ``defaults`` is resolved relative to the config file's own directory and
    merged before the file's own keys are applied on top (so a specific
    config always wins over its base).
    """
    path = Path(path)
    with path.open("r") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    defaults = config.pop("defaults", None)
    if defaults is not None:
        base_path = (path.parent / defaults).resolve()
        base_config = load_config(base_path)
        return _deep_update(base_config, config)

    return config
