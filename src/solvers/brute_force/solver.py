"""Enumerate all assignments in the QUBO or TQUDO configuration space (exact global minimum)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from instance_gen_process import generate_QUBO_from_problem, generate_TQUDO_from_problem
from instance_gen_process.models import ProblemInstance, RestrictionConfig
from solvers.base import SolverResult, SolverRunConfig
from solvers.brute_force.limits import QUBO_MAX_BINARY_VARS, TQUDO_MAX_N_AVAILABLE
from utils.constraints import (
    qubo_binary_to_sequence,
    validate_solution_constraints_qubo,
    validate_solution_constraints_tqudo,
)
from utils.costs import calculate_qubo_cost, calculate_real_cost, calculate_tqudo_cost


def _default_restriction() -> RestrictionConfig:
    return RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)


def _lex_less_seq(a: list[int], b: list[int]) -> bool:
    return tuple(a) < tuple(b)


class BruteForceSolver:
    """Exhaustive search: every assignment in the formulation space (QUBO: ``2^n_vars``; TQUDO: ``n^n``)."""

    solver_name = "brute_force"

    def solve(self, instance: ProblemInstance, run_config: SolverRunConfig) -> SolverResult:
        """Return the global minimum energy and metadata (best feasible if distinct)."""
        formulation = run_config.formulation
        if formulation not in ("qubo", "tqudo"):
            raise ValueError(
                f"brute_force only supports formulation 'qubo' or 'tqudo', got {formulation!r}."
            )

        restriction = run_config.restriction_config or _default_restriction()
        n_available = instance.n_cities - 1
        if formulation == "tqudo" and n_available > TQUDO_MAX_N_AVAILABLE:
            raise ValueError(
                f"brute_force TQUDO requires n_cities - 1 <= {TQUDO_MAX_N_AVAILABLE} "
                f"(enumeration size n^n); got n_available={n_available}."
            )
        if formulation == "qubo":
            n_vars_precheck = n_available * n_available
            if n_vars_precheck > QUBO_MAX_BINARY_VARS:
                raise ValueError(
                    f"brute_force QUBO requires (n_cities - 1)^2 <= {QUBO_MAX_BINARY_VARS} "
                    f"binary variables (2^n_vars configs); got n_vars={n_vars_precheck}."
                )

        start = time.perf_counter()

        if formulation == "tqudo":
            result = self._solve_tqudo(instance, restriction, run_config, n_available)
        else:
            result = self._solve_qubo(instance, restriction, run_config, n_available)

        runtime_seconds = time.perf_counter() - start
        return SolverResult(
            solver_name=self.solver_name,
            objective_value=result["objective_value"],
            feasible=result["feasible"],
            runtime_seconds=runtime_seconds,
            metadata=result["metadata"],
        )

    def _solve_tqudo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
        n_available: int,
    ) -> dict[str, Any]:
        cap = run_config.brute_force_max_assignments_tqudo
        cardinal = n_available**n_available
        if cardinal > cap:
            raise ValueError(
                f"TQUDO brute_force: {cardinal} assignments exceed "
                f"brute_force_max_assignments_tqudo={cap}."
            )

        problem = generate_TQUDO_from_problem(instance, restriction)
        best_cost = float("inf")
        best_seq: list[int] | None = None
        best_feasible_cost = float("inf")
        best_feasible_seq: list[int] | None = None

        seq_arr = np.zeros(n_available, dtype=int)
        for idx in range(cardinal):
            x = idx
            for t in range(n_available):
                seq_arr[t] = x % n_available
                x //= n_available
            seq_list = seq_arr.tolist()
            cost = calculate_tqudo_cost(problem, seq_arr)
            if best_seq is None or cost < best_cost or (
                cost == best_cost and _lex_less_seq(seq_list, best_seq)
            ):
                best_cost = cost
                best_seq = seq_list

            if validate_solution_constraints_tqudo(instance, seq_list):
                if best_feasible_seq is None or cost < best_feasible_cost or (
                    cost == best_feasible_cost and _lex_less_seq(seq_list, best_feasible_seq)
                ):
                    best_feasible_cost = cost
                    best_feasible_seq = seq_list

        assert best_seq is not None
        feasible = validate_solution_constraints_tqudo(instance, best_seq)
        metadata: dict[str, Any] = {
            "best_sequence": best_seq,
            "configs_evaluated": cardinal,
        }
        if best_feasible_seq is not None:
            metadata["best_feasible_objective_value"] = best_feasible_cost
            metadata["best_feasible_sequence"] = best_feasible_seq
            metadata["best_feasible_real_cost"] = float(
                calculate_real_cost(instance, best_feasible_seq)
            )
        if feasible:
            metadata["real_cost"] = float(calculate_real_cost(instance, best_seq))

        return {"objective_value": best_cost, "feasible": feasible, "metadata": metadata}

    def _solve_qubo(
        self,
        instance: ProblemInstance,
        restriction: RestrictionConfig,
        run_config: SolverRunConfig,
        n_available: int,
    ) -> dict[str, Any]:
        n_vars = n_available * n_available
        cap = run_config.brute_force_max_assignments_qubo
        cardinal = 1 << n_vars
        if cardinal > cap:
            raise ValueError(
                f"QUBO brute_force: {cardinal} assignments exceed "
                f"brute_force_max_assignments_qubo={cap}."
            )

        problem = generate_QUBO_from_problem(instance, restriction)
        x = np.zeros(n_vars, dtype=np.float64)
        best_cost = float("inf")
        best_i: int | None = None
        best_feasible_cost = float("inf")
        best_feasible_i: int | None = None

        for i in range(cardinal):
            v = i
            for b in range(n_vars):
                x[b] = float(v & 1)
                v >>= 1
            cost = calculate_qubo_cost(problem, x)
            if best_i is None or cost < best_cost or (cost == best_cost and i < best_i):
                best_cost = cost
                best_i = i

            if validate_solution_constraints_qubo(instance, x):
                if best_feasible_i is None or cost < best_feasible_cost or (
                    cost == best_feasible_cost and i < best_feasible_i
                ):
                    best_feasible_cost = cost
                    best_feasible_i = i

        assert best_i is not None
        v = best_i
        for b in range(n_vars):
            x[b] = float(v & 1)
            v >>= 1

        best_binary = x.tolist()
        bitstring = "".join("1" if bit > 0.5 else "0" for bit in best_binary)
        feasible = validate_solution_constraints_qubo(instance, x)

        metadata: dict[str, Any] = {
            "best_binary": best_binary,
            "best_bitstring": bitstring,
            "configs_evaluated": cardinal,
        }

        seq_decoded = qubo_binary_to_sequence(x, n_available)
        if seq_decoded is not None:
            metadata["best_sequence"] = seq_decoded.tolist()

        if best_feasible_i is not None:
            v2 = best_feasible_i
            xf = np.zeros(n_vars, dtype=np.float64)
            for b in range(n_vars):
                xf[b] = float(v2 & 1)
                v2 >>= 1
            metadata["best_feasible_objective_value"] = best_feasible_cost
            metadata["best_feasible_binary"] = xf.tolist()
            metadata["best_feasible_bitstring"] = "".join(
                "1" if xf[b] > 0.5 else "0" for b in range(n_vars)
            )
            seq_f = qubo_binary_to_sequence(xf, n_available)
            if seq_f is not None:
                sq = seq_f.tolist()
                metadata["best_feasible_sequence"] = sq
                metadata["best_feasible_real_cost"] = float(calculate_real_cost(instance, sq))

        if feasible and seq_decoded is not None:
            metadata["real_cost"] = float(calculate_real_cost(instance, seq_decoded.tolist()))

        return {"objective_value": best_cost, "feasible": feasible, "metadata": metadata}
