"""Tests for instance config loading and generation."""

import random

import pytest

from conftest import cleanup_workspace_tmp_dir, workspace_tmp_dir
from instance_gen_process import generate_random_instance, load_instance_config


def test_load_instance_config_and_generate() -> None:
    tmp_path = workspace_tmp_dir("instance_config")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 5",
                    "n_precedences_range: [2, 5]",
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

        assert config.n_cities == 5
        assert config.n_precedences_range == (2, 5)
        assert instance.n_cities == 5
        assert len(instance.precedences) >= 2
        assert len(instance.precedences) <= 5
        n_available = instance.n_cities - 1
        assert instance.prices_hotels.shape == (n_available, n_available)
        assert instance.prices_travels.shape == (instance.n_cities, instance.n_cities, instance.n_cities)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_instance_config_rejects_n_cities_below_two() -> None:
    tmp_path = workspace_tmp_dir("instance_config_invalid_cities")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 1",
                    "n_precedences_range: [0, 1]",
                    "prices_range_hotels: [30., 150.]",
                    "prices_range_travels: [30., 150.]",
                    "seed: 123",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="at least 2"):
            load_instance_config(config_path)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_instance_config_rejects_precedence_upper_bound_above_maximum() -> None:
    tmp_path = workspace_tmp_dir("instance_config_invalid_precedences")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 5",
                    "n_precedences_range: [2, 7]",
                    "prices_range_hotels: [30., 150.]",
                    "prices_range_travels: [30., 150.]",
                    "seed: 123",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="maximum feasible"):
            load_instance_config(config_path)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_instance_config_accepts_valid_precedence_upper_bound_edge_case() -> None:
    tmp_path = workspace_tmp_dir("instance_config_edge_precedences")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 5",
                    "n_precedences_range: [0, 6]",
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

        assert config.n_precedences_range == (0, 6)
        assert len(instance.precedences) <= 6
    finally:
        cleanup_workspace_tmp_dir(tmp_path)
