"""Tests for instance config loading and generation."""

from pathlib import Path
import random
import shutil
from uuid import uuid4

from instance_gen_process import generate_random_instance, load_instance_config


def _workspace_tmp_dir(prefix: str) -> Path:
    base_dir = Path(__file__).resolve().parent / ".tmp"
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"{prefix}_{uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def _cleanup_workspace_tmp_dir(temp_dir: Path) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)
    base_dir = temp_dir.parent
    if base_dir.exists() and not any(base_dir.iterdir()):
        base_dir.rmdir()


def test_load_instance_config_and_generate() -> None:
    tmp_path = _workspace_tmp_dir("instance_config")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 8",
                    "n_precedence_range: [2, 5]",
                    "prices_range_hotels: [30., 150.]",
                    "prices_range_travels: [30., 150.]",
                    "seed: 123",
                ]
            ),
            encoding="utf-8",
        )

        config = load_instance_config(config_path)
        rng = random.Random(config.seed)
        instance = generate_random_instance(config, rng)

        assert config.n_cities == 8
        assert config.n_precedences_range == (2, 5)
        assert instance.n_cities == 8
        assert len(instance.precedences) >= 2
        assert len(instance.precedences) <= 5
        n_available = instance.n_cities - 1
        assert instance.prices_hotels.shape == (n_available, n_available)
        assert instance.prices_travels.shape == (instance.n_cities, instance.n_cities, instance.n_cities)
    finally:
        _cleanup_workspace_tmp_dir(tmp_path)
