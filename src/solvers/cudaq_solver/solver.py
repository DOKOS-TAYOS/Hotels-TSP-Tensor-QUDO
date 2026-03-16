"""CUDA-Q backend for QAOA (TQUDO and QUBO formulations)."""

from __future__ import annotations

import time

from instance_gen_process import ProblemInstance, generate_QUBO_from_problem, generate_TQUDO_from_problem
from instance_gen_process.models import RestrictionConfig
from solvers.base import SolverResult, SolverRunConfig
from utils.constraints import qubo_binary_to_sequence, validate_solution_constraints_qubo, validate_solution_constraints_tqudo


def _default_restriction() -> RestrictionConfig:
    """Default penalty coefficients for constraint encoding."""
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


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
        return SolverResult(
            solver_name=self.solver_name,
            objective_value=result["energy"],
            feasible=result["feasible"],
            runtime_seconds=runtime_seconds,
            metadata={
                "best_sequence": result.get("best_sequence"),
                "best_bitstring": result.get("best_bitstring"),
                "best_binary": result.get("best_binary"),
            },
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
        )
        best_sequence = raw["best_sequence"].tolist()
        feasible = validate_solution_constraints_tqudo(instance, best_sequence)
        return {
            "energy": float(raw["energy"]),
            "feasible": feasible,
            "best_sequence": best_sequence,
            "best_bitstring": raw["best_bitstring"],
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
            n_shots=run_config.qaoa_sample_shots,
            seed=run_config.seed,
        )
        n_available = instance.n_cities - 1
        best_binary = raw["best_binary"]
        best_sequence = qubo_binary_to_sequence(best_binary, n_available)
        feasible = (
            best_sequence is not None
            and validate_solution_constraints_qubo(instance, best_binary)
        )
        return {
            "energy": float(raw["energy"]),
            "feasible": feasible,
            "best_sequence": best_sequence.tolist() if best_sequence is not None else None,
            "best_bitstring": raw["best_bitstring"],
            "best_binary": best_binary.tolist(),
        }

