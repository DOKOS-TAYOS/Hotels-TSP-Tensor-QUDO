"""Load instance generation configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from instance_gen_process.models import InstanceConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _parse_range(raw_value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if raw_value is None:
        return default
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"Expected a range with two values, got: {raw_value!r}")
    low = float(raw_value[0])
    high = float(raw_value[1])
    if low >= high:
        raise ValueError(f"Invalid range bounds: {raw_value!r}")
    return (low, high)


def load_instance_config(path: Path | str | None = None) -> InstanceConfig:
    """Load and validate `InstanceConfig` from YAML."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Instance config file not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Instance config must be a YAML mapping.")

    config = InstanceConfig(
        num_items=int(raw_config.get("num_items", 8)),
        max_weight=float(raw_config.get("max_weight", 50_000)),
        max_volume=float(raw_config.get("max_volume", 200)),
        cg_min=float(raw_config.get("cg_min", -15)),
        cg_max=float(raw_config.get("cg_max", 15)),
        weight_range=_parse_range(raw_config.get("weight_range"), (100.0, 1000.0)),
        volume_range=_parse_range(raw_config.get("volume_range"), (1.0, 10.0)),
        seed=int(raw_config.get("seed", 42)),
    )

    if config.num_items <= 0:
        raise ValueError("num_items must be a positive integer.")
    if config.cg_min >= config.cg_max:
        raise ValueError("cg_min must be strictly smaller than cg_max.")
    if config.max_weight <= 0 or config.max_volume <= 0:
        raise ValueError("max_weight and max_volume must be strictly positive.")

    return config

