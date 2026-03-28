"""Map validated solver configuration dicts to ``SolverRunConfig``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solvers.base import SolverRunConfig


def solver_config_to_run_config(config: dict[str, Any]) -> SolverRunConfig:
    """Map a validated solver dict to ``SolverRunConfig``.

    Args:
        config: Output of ``load_solver_config`` or ``parse_solver_config_dict``.

    Returns:
        Immutable run configuration passed to ``SolverProtocol.solve``.

    """
    from solvers.base import SolverRunConfig
    from solvers.noise import NoiseConfig

    return SolverRunConfig(
        max_iterations=config["max_iterations"],
        timeout_seconds=config["timeout_seconds"],
        formulation=config["formulation"],
        restriction_config=config["restriction"],
        qaoa_depth=config["qaoa_depth"],
        qaoa_max_iter=config["qaoa_max_iter"],
        delta_t=config["qaoa_delta_t"],
        optimizer_tol=config["qaoa_optimizer_tol"],
        qaoa_shots=config["qaoa_shots"],
        qaoa_sample_shots=config["qaoa_sample_shots"],
        seed=config["seed"],
        optimizer=config["optimizer"],
        noise_config=config.get("noise_config", NoiseConfig()),
        sa_t_initial=config["sa_t_initial"],
        sa_t_final=config["sa_t_final"],
        sa_alpha=config["sa_alpha"],
        brute_force_max_assignments_tqudo=config["brute_force_max_assignments_tqudo"],
        brute_force_max_assignments_qubo=config["brute_force_max_assignments_qubo"],
    )
