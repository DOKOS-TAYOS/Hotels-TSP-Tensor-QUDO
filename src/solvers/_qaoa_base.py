"""Shared base class for QAOA-based solver backends (Cirq and CUDA-Q)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from instance_gen_process import (
    ProblemInstance,
    generate_QUBO_from_problem,
    generate_TQUDO_from_problem,
)
from instance_gen_process.models import RestrictionConfig
from solvers.base import SolverResult, SolverRunConfig
from utils.constraints import (
    qubo_binary_to_sequence,
    validate_solution_constraints_qubo,
    validate_solution_constraints_tqudo,
)
from utils.costs import calculate_real_cost


def _default_restriction() -> RestrictionConfig:
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


_OPTIONAL_METADATA_KEYS = (
    "initial_energy",
    "energy_history",
    "optimal_angles",
)


class BaseQAOASolver(ABC):
    """Common solve/metadata logic shared by Cirq and CUDA-Q solvers.

    Subclasses must set ``solver_name`` and implement the three abstract
    methods: ``_get_tqudo_runner``, ``_get_tqudo_virtual_runner``, and
    ``_get_qubo_runner``.  Returning ``None`` from any runner signals that the
    formulation is not supported by this backend.
    """

    solver_name: str

    @abstractmethod
    def _get_tqudo_runner(self) -> Callable[..., dict] | None:
        """Return ``run_qaoa`` for native TQUDO, or None if unsupported."""

    @abstractmethod
    def _get_tqudo_virtual_runner(self) -> Callable[..., dict] | None:
        """Return ``run_qaoa`` for TQUDO virtual (qubit emulation), or None."""

    @abstractmethod
    def _get_qubo_runner(self) -> Callable[..., dict] | None:
        """Return ``run_qaoa`` for QUBO, or None if unsupported."""

    @abstractmethod
    def _serialize_samples(self, samples: Any) -> dict[str, int] | None:
        """Convert backend-specific samples to a sorted ``{bitstring: count}`` dict."""

    @abstractmethod
    def _noise_qubit_count(
        self, instance: ProblemInstance, formulation: str,
    ) -> tuple[int, dict[str, Any]]:
        """Return ``(n_qubits, extra_kwargs)`` for ``warn_if_large_system``."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, instance: ProblemInstance, run_config: SolverRunConfig,
    ) -> SolverResult:
        """Run QAOA and return a standardized result."""
        restriction = run_config.restriction_config or _default_restriction()
        formulation = run_config.formulation

        n_qubits, noise_kwargs = self._noise_qubit_count(instance, formulation)
        run_config.noise_config.warn_if_large_system(n_qubits, **noise_kwargs)

        start = time.perf_counter()

        if formulation in ("tqudo", "tqudo_virtual"):
            result = self._solve_tqudo(instance, restriction, run_config)
        else:
            result = self._solve_qubo(instance, restriction, run_config)

        runtime_seconds = time.perf_counter() - start
        metadata = self._build_metadata(result, instance)
        return SolverResult(
            solver_name=self.solver_name,
            objective_value=result["energy"],
            feasible=result["feasible"],
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(
        self, result: dict, instance: ProblemInstance,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "best_sequence": result.get("best_sequence"),
            "best_bitstring": result.get("best_bitstring"),
            "best_binary": result.get("best_binary"),
        }
        for key in _OPTIONAL_METADATA_KEYS:
            if key in result:
                metadata[key] = result[key]
        if "initial_samples" in result:
            metadata["initial_samples"] = self._serialize_samples(result["initial_samples"])
        if "final_samples" in result:
            metadata["final_samples"] = self._serialize_samples(result["final_samples"])
        if result["feasible"] and result.get("best_sequence") is not None:
            metadata["real_cost"] = float(
                calculate_real_cost(instance, result["best_sequence"])
            )
        return metadata

    def _run_qaoa_tqudo(
        self, run_qaoa_fn: Callable[..., dict],
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
    ) -> dict:
        problem = generate_TQUDO_from_problem(instance, restriction)
        raw = run_qaoa_fn(
            problem.Etab,
            problem.Ettprimeab,
            depth=run_config.qaoa_depth,
            max_iter=run_config.qaoa_max_iter,
            n_shots=run_config.qaoa_shots,
            sample_shots=run_config.qaoa_sample_shots,
            seed=run_config.seed,
            optimizer=run_config.optimizer,
            delta_t=run_config.delta_t,
            noise_config=run_config.noise_config,
        )
        best_sequence_array = raw["best_sequence"]
        best_sequence = best_sequence_array.tolist() if best_sequence_array is not None else None
        feasible = (
            best_sequence is not None
            and validate_solution_constraints_tqudo(instance, best_sequence)
        )
        depth = run_config.qaoa_depth
        params = raw["params"]
        s = problem.energy_scale
        return {
            "energy": float(raw["energy"]) * s,
            "feasible": feasible,
            "best_sequence": best_sequence,
            "best_bitstring": raw["best_bitstring"],
            "initial_energy": raw["initial_energy"] * s,
            "energy_history": [e * s for e in raw["energy_history"]],
            "initial_samples": raw.get("initial_samples"),
            "final_samples": raw.get("final_samples"),
            "optimal_angles": {
                "gamma": params[:depth].tolist(),
                "beta": params[depth:].tolist(),
            },
        }

    def _solve_tqudo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
    ) -> dict:
        formulation = run_config.formulation
        if formulation == "tqudo_virtual":
            runner = self._get_tqudo_virtual_runner()
        else:
            runner = self._get_tqudo_runner()
        if runner is None:
            raise ValueError(
                f"Formulation '{formulation}' is not supported by the "
                f"'{self.solver_name}' backend."
            )
        return self._run_qaoa_tqudo(runner, instance, restriction, run_config)

    def _solve_qubo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
    ) -> dict:
        runner = self._get_qubo_runner()
        if runner is None:
            raise ValueError(
                f"Formulation 'qubo' is not supported by the "
                f"'{self.solver_name}' backend."
            )
        problem = generate_QUBO_from_problem(instance, restriction)
        raw = runner(
            problem.qubo_matrix,
            depth=run_config.qaoa_depth,
            max_iter=run_config.qaoa_max_iter,
            n_shots=run_config.qaoa_shots,
            sample_shots=run_config.qaoa_sample_shots,
            seed=run_config.seed,
            optimizer=run_config.optimizer,
            delta_t=run_config.delta_t,
            noise_config=run_config.noise_config,
        )
        n_available = instance.n_cities - 1
        best_binary = raw["best_binary"]
        best_sequence = qubo_binary_to_sequence(best_binary, n_available)
        feasible = (
            best_sequence is not None
            and validate_solution_constraints_qubo(instance, best_binary)
        )
        depth = run_config.qaoa_depth
        params = raw["params"]
        s = problem.energy_scale
        return {
            "energy": float(raw["energy"]) * s,
            "feasible": feasible,
            "best_sequence": best_sequence.tolist() if best_sequence is not None else None,
            "best_bitstring": raw["best_bitstring"],
            "best_binary": best_binary.tolist(),
            "initial_energy": raw["initial_energy"] * s,
            "energy_history": [e * s for e in raw["energy_history"]],
            "initial_samples": raw.get("initial_samples"),
            "final_samples": raw.get("final_samples"),
            "optimal_angles": {
                "gamma": params[:depth].tolist(),
                "beta": params[depth:].tolist(),
            },
        }
