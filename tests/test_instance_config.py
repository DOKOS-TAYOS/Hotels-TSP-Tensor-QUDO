"""Tests for instance config loading and generation."""

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
        instance = generate_random_instance(config, config.seed)

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

        with pytest.raises(ValueError, match="at least 3"):
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
        instance = generate_random_instance(config, config.seed)

        assert config.n_precedences_range == (0, 6)
        assert len(instance.precedences) <= 6
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_instance_config_accepts_fixed_ranges_for_deterministic_experiments() -> None:
    tmp_path = workspace_tmp_dir("instance_config_fixed_ranges")
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_cities: 5",
                    "n_precedences_range: [2, 2]",
                    "prices_range_hotels: [30., 30.]",
                    "prices_range_travels: [40., 40.]",
                    "seed: 123",
                ]
            ),
            encoding="utf-8",
        )

        config = load_instance_config(config_path)
        instance = generate_random_instance(config, config.seed)

        assert config.n_precedences_range == (2, 2)
        assert config.prices_range_hotels == (30.0, 30.0)
        assert config.prices_range_travels == (40.0, 40.0)
        assert len(instance.precedences) == 2
        assert instance.prices_hotels.min() == 30.0
        assert instance.prices_hotels.max() == 30.0
        assert instance.prices_travels[0, instance.n_cities - 1, 0] == 40.0
    finally:
        cleanup_workspace_tmp_dir(tmp_path)
