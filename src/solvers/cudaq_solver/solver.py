"""CUDA-Q backend for QAOA (TQUDO and QUBO formulations)."""

from __future__ import annotations

import time
from typing import Any

from instance_gen_process import ProblemInstance, generate_QUBO_from_problem, generate_TQUDO_from_problem
from instance_gen_process.models import RestrictionConfig
from solvers.base import SolverResult, SolverRunConfig
from utils.constraints import qubo_binary_to_sequence, validate_solution_constraints_qubo, validate_solution_constraints_tqudo
from utils.costs import calculate_real_cost


def _default_restriction() -> RestrictionConfig:
    """Default penalty coefficients for constraint encoding."""
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


def _serialize_samples(samples: Any) -> dict[str, int] | None:
    """Convert a cudaq.SampleResult to a JSON-serializable dict.

    Returns a mapping of bitstring -> count, or None if samples is None.
    Sorted by count descending for readability.
    """
    if samples is None:
        return None
    counts: dict[str, int] = {bs: int(cnt) for bs, cnt in samples.items()}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


class CudaqSolver:
    """CUDA-Q solver using QAOA for TQUDO or QUBO formulations."""

    solver_name = "cudaq"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Run QAOA and return standardized result."""
        restriction = run_config.restriction_config or _default_restriction()
        start = time.perf_counter()

        if run_config.formulation == "tqudo":
            result = self._solve_tqudo(instance, restriction, run_config)
        else:
            result = self._solve_qubo(instance, restriction, run_config)

        runtime_seconds = time.perf_counter() - start
        metadata: dict = {
            "best_sequence": result.get("best_sequence"),
            "best_bitstring": result.get("best_bitstring"),
            "best_binary": result.get("best_binary"),
        }
        if "initial_energy" in result:
            metadata["initial_energy"] = result["initial_energy"]
        if "energy_history" in result:
            metadata["energy_history"] = result["energy_history"]
        if "optimal_angles" in result:
            metadata["optimal_angles"] = result["optimal_angles"]
        if "initial_samples" in result:
            metadata["initial_samples"] = _serialize_samples(result["initial_samples"])
        if "final_samples" in result:
            metadata["final_samples"] = _serialize_samples(result["final_samples"])
        if result["feasible"] and result.get("best_sequence") is not None:
            metadata["real_cost"] = float(
                calculate_real_cost(instance, result["best_sequence"])
            )
        return SolverResult(
            solver_name=self.solver_name,
            objective_value=result["energy"],
            feasible=result["feasible"],
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    def _solve_tqudo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
    ) -> dict:
        """Solve using TQUDO QAOA."""
        from solvers.cudaq_solver.qaoa_circuit_tqudo import run_qaoa

        problem = generate_TQUDO_from_problem(instance, restriction)
        raw = run_qaoa(
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

    def _solve_qubo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
    ) -> dict:
        """Solve using QUBO QAOA."""
        from solvers.cudaq_solver.qaoa_circuit_qubo import run_qaoa

        problem = generate_QUBO_from_problem(instance, restriction)
        raw = run_qaoa(
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
