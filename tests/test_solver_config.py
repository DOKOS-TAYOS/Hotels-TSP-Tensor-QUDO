"""Tests for solver config loading and workflow compatibility checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import cleanup_workspace_tmp_dir, workspace_tmp_dir
from instance_gen_process import (
    load_solver_config,
    solver_config_to_run_config,
    validate_solver_instance_compatibility,
)
from instance_gen_process.models import InstanceConfig


def _write_solver_config(path: Path, extra_lines: list[str]) -> None:
    """Write a minimally valid solver config plus the provided overrides."""
    base_lines = [
        "n_instances: 1",
        "solver: cirq",
        "formulation: qubo",
        "optimizer: COBYLA",
    ]
    path.write_text("\n".join(base_lines + extra_lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("extra_lines", "match"),
    [
        (["qaoa_depth: 0"], "qaoa_depth"),
        (["qaoa_max_iter: 0"], "qaoa_max_iter"),
        (["qaoa_shots: 0"], "qaoa_shots"),
        (["qaoa_sample_shots: 0"], "qaoa_sample_shots"),
        (["qaoa_delta_t: 0"], "qaoa_delta_t"),
        (["qaoa_optimizer_tol: 0"], "qaoa_optimizer_tol"),
        (["max_iterations: -1"], "max_iterations"),
    ],
)
def test_load_solver_config_rejects_invalid_numeric_controls(
    extra_lines: list[str],
    match: str,
) -> None:
    tmp_path = workspace_tmp_dir("solver_config_invalid_numeric")
    config_path = tmp_path / "solver_config.yaml"
    try:
        _write_solver_config(config_path, extra_lines)

        with pytest.raises(ValueError, match=match):
            load_solver_config(config_path)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_solver_config_rejects_cobyla_budget_below_parameter_count() -> None:
    tmp_path = workspace_tmp_dir("solver_config_cobyla_budget")
    config_path = tmp_path / "solver_config.yaml"
    try:
        _write_solver_config(
            config_path,
            [
                "qaoa_depth: 2",
                "qaoa_max_iter: 5",
            ],
        )

        with pytest.raises(ValueError, match="at least 6"):
            load_solver_config(config_path)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_solver_config_qaoa_delta_t_and_optimizer_tol_defaults() -> None:
    tmp_path = workspace_tmp_dir("solver_config_qaoa_float_defaults")
    config_path = tmp_path / "solver_config.yaml"
    try:
        _write_solver_config(config_path, [])
        config = load_solver_config(config_path)
        assert config["qaoa_delta_t"] == pytest.approx(0.55)
        assert config["qaoa_optimizer_tol"] == pytest.approx(1e-6)
        run_cfg = solver_config_to_run_config(config)
        assert run_cfg.delta_t == pytest.approx(0.55)
        assert run_cfg.optimizer_tol == pytest.approx(1e-6)
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_load_solver_config_accepts_cudaq_tqudo_combination() -> None:
    tmp_path = workspace_tmp_dir("solver_config_cudaq_tqudo")
    config_path = tmp_path / "solver_config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "n_instances: 1",
                    "solver: cudaq",
                    "formulation: tqudo",
                    "optimizer: COBYLA",
                ]
            ),
            encoding="utf-8",
        )

        config = load_solver_config(config_path)
        assert config["solver"] == "cudaq"
        assert config["formulation"] == "tqudo"
    finally:
        cleanup_workspace_tmp_dir(tmp_path)


def test_validate_solver_instance_compatibility_rejects_invalid_tqudo_dimension() -> None:
    instance_config = InstanceConfig(
        n_cities=6,
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"formulation": "tqudo_virtual"}

    with pytest.raises(ValueError, match="power of two"):
        validate_solver_instance_compatibility(instance_config, solver_config)


def test_validate_solver_instance_compatibility_rejects_cudaq_native_tqudo() -> None:
    """CUDA-Q does not support native TQUDO (real qudits)."""
    instance_config = InstanceConfig(
        n_cities=5,
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"solver": "cudaq", "formulation": "tqudo"}

    with pytest.raises(ValueError, match="not supported by the CUDA-Q"):
        validate_solver_instance_compatibility(instance_config, solver_config)


def test_validate_solver_instance_compatibility_accepts_cudaq_tqudo_virtual() -> None:
    """CUDA-Q accepts tqudo_virtual when dimension is power of two."""
    instance_config = InstanceConfig(
        n_cities=5,
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"solver": "cudaq", "formulation": "tqudo_virtual"}

    validate_solver_instance_compatibility(instance_config, solver_config)


def test_validate_solver_instance_compatibility_rejects_sa_tqudo_virtual() -> None:
    """SA does not support tqudo_virtual."""
    instance_config = InstanceConfig(
        n_cities=5,
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"solver": "simulated_annealing", "formulation": "tqudo_virtual"}

    with pytest.raises(ValueError, match="not supported by simulated annealing"):
        validate_solver_instance_compatibility(instance_config, solver_config)


def test_validate_solver_instance_compatibility_accepts_compatible_tqudo_dimension() -> None:
    instance_config = InstanceConfig(
        n_cities=5,
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"solver": "cirq", "formulation": "tqudo"}

    validate_solver_instance_compatibility(instance_config, solver_config)


def test_validate_solver_instance_compatibility_cirq_accepts_non_power_of_two() -> None:
    """Native-qudit Cirq backend supports arbitrary dimensions (not just power-of-2)."""
    instance_config = InstanceConfig(
        n_cities=6,  # n_available = 5, NOT a power of two
        n_precedences_range=(1, 1),
        prices_range_hotels=(30.0, 30.0),
        prices_range_travels=(40.0, 40.0),
        seed=123,
    )
    solver_config = {"solver": "cirq", "formulation": "tqudo"}

    # Should NOT raise — Cirq native qudits support any dimension.
    validate_solver_instance_compatibility(instance_config, solver_config)
