"""Load instance generation configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from instance_gen_process.models import InstanceConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _parse_range(raw_value: Any, field_name: str) -> tuple[float, float]:
    """Parse a two-element range of floats from YAML data.

    Args:
        raw_value: Raw value from YAML (list or tuple of two numbers).
        field_name: Field name for error messages.

    Returns:
        Tuple (low, high) with low <= high.

    Raises:
        ValueError: If value is missing, not a 2-element sequence, or low > high.
    """
    if raw_value is None:
        raise ValueError(f"Missing required field: {field_name}")
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{field_name}: expected a range with two values, got: {raw_value!r}")
    low = float(raw_value[0])
    high = float(raw_value[1])
    if low > high:
        raise ValueError(f"{field_name}: invalid range bounds: {raw_value!r}")
    return (low, high)


def _parse_int_range(raw_value: Any, field_name: str) -> tuple[int, int]:
    """Parse a two-element range of integers from YAML data.

    Args:
        raw_value: Raw value from YAML (list or tuple of two integers).
        field_name: Field name for error messages.

    Returns:
        Tuple (low, high) with low <= high.

    Raises:
        ValueError: If value is missing, not a 2-element sequence, or low > high.
    """
    if raw_value is None:
        raise ValueError(f"Missing required field: {field_name}")
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{field_name}: expected a range with two values, got: {raw_value!r}")
    low = int(raw_value[0])
    high = int(raw_value[1])
    if low > high:
        raise ValueError(f"{field_name}: invalid range bounds: {raw_value!r}")
    return (low, high)


def load_instance_config(path: Path | str | None = None) -> InstanceConfig:
    """Load and validate `InstanceConfig` from YAML.

    Args:
        path: Path to YAML config file. If None, uses DEFAULT_CONFIG_PATH.

    Returns:
        InstanceConfig with n_cities, n_precedences_range, price ranges, seed.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "n_cities" not in data:
        raise ValueError("Missing required field: n_cities")
    n_cities = int(data["n_cities"])
    if n_cities < 3:
        raise ValueError(
            "n_cities must be at least 3 (depot + 2 available cities). "
            "With n_cities=2 the TQUDO formulation degenerates to zero cost."
        )

    n_precedences_range = _parse_int_range(
        data.get("n_precedences_range"), field_name="n_precedences_range"
    )
    if n_precedences_range[0] < 0 or n_precedences_range[1] < 0:
        raise ValueError("n_precedences_range must be non-negative")

    n_available = n_cities - 1
    max_precedences = n_available * (n_available - 1) // 2
    if n_precedences_range[1] > max_precedences:
        raise ValueError(
            "n_precedences_range upper bound exceeds the maximum feasible number "
            f"of precedence constraints ({max_precedences}) for n_cities={n_cities}"
        )

    prices_range_hotels = _parse_range(
        data.get("prices_range_hotels"), field_name="prices_range_hotels"
    )
    prices_range_travels = _parse_range(
        data.get("prices_range_travels"), field_name="prices_range_travels"
    )

    if "seed" not in data:
        raise ValueError("Missing required field: seed")
    seed = int(data["seed"])

    return InstanceConfig(
        n_cities=n_cities,
        n_precedences_range=n_precedences_range,
        prices_range_hotels=prices_range_hotels,
        prices_range_travels=prices_range_travels,
        seed=seed,
    )
