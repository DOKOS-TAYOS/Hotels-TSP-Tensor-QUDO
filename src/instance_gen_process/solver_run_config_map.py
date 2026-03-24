"""Map validated solver dicts to :class:`~solvers.base.SolverRunConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solvers.base import SolverRunConfig


def solver_config_to_run_config(config: dict[str, Any]) -> SolverRunConfig:
    """Map a dict from :func:`~instance_gen_process.solver_config_parse.load_solver_config` / :func:`~instance_gen_process.solver_config_parse.parse_solver_config_dict` to run config.

    Args:
        config: Validated solver configuration dictionary.

    Returns:
        Frozen run configuration for :meth:`~solvers.base.SolverProtocol.solve`.
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
