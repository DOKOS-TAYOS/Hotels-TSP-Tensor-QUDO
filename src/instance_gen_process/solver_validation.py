"""Compatibility checks between instance size and solver / formulation."""

from __future__ import annotations

from typing import Any

from instance_gen_process.models import InstanceConfig
from utils.qaoa_helpers import is_power_of_two

_QUBO_QUBIT_WARN_THRESHOLD = 30


def validate_solver_instance_compatibility(
    instance_config: InstanceConfig,
    solver_config: dict[str, Any],
) -> None:
    """Raise if the instance size and solver formulation are incompatible.

    Enforces backend capabilities: Cirq supports ``qubo``, ``tqudo``, and
    ``tqudo_virtual``; CUDA-Q supports ``qubo`` and ``tqudo_virtual``;
    simulated annealing supports ``qubo`` and ``tqudo``. Also enforces qubit
    count heuristics and power-of-two rules for virtual TQUDO.

    Args:
        instance_config: Instance generation parameters (e.g. ``n_cities``).
        solver_config: Loaded solver dict with ``solver`` and ``formulation``.

    Raises:
        ValueError: If the combination is unsupported or impractical.
    """
    formulation = solver_config.get("formulation", "tqudo")
    solver = solver_config.get("solver", "cudaq")
    n_available = instance_config.n_cities - 1

    if formulation == "qubo" and solver in ("cudaq", "cirq"):
        n_qubits = n_available * n_available
        if n_qubits > _QUBO_QUBIT_WARN_THRESHOLD:
            raise ValueError(
                f"QUBO formulation with {instance_config.n_cities} cities requires "
                f"{n_qubits} qubits for quantum simulation, which exceeds the "
                f"practical limit (~{_QUBO_QUBIT_WARN_THRESHOLD}). "
                "Use formulation='tqudo' or 'tqudo_virtual' for quantum backends, "
                "or solver='simulated_annealing' for QUBO at this scale."
            )

    if formulation == "tqudo" and solver == "cudaq":
        raise ValueError(
            "Native TQUDO (real qudits) is not supported by the CUDA-Q backend. "
            "Use formulation='tqudo_virtual' for qubit-emulated TQUDO on CUDA-Q, "
            "or use solver='cirq' for native qudit support."
        )

    if formulation == "tqudo_virtual" and solver == "simulated_annealing":
        raise ValueError(
            "TQUDO virtual (qubit emulation) is not supported by simulated annealing. "
            "Use formulation='tqudo' or 'qubo' instead."
        )

    if formulation == "tqudo_virtual" and not is_power_of_two(n_available):
        raise ValueError(
            f"TQUDO virtual (qubit emulation) requires n_cities - 1 to be a "
            "power of two. Use formulation='tqudo' with solver='cirq' (native "
            "qudits) for arbitrary dimensions. "
            f"Got n_cities={instance_config.n_cities} (n_cities - 1 = {n_available})."
        )

    if solver == "brute_force":
        from solvers.brute_force.limits import QUBO_MAX_BINARY_VARS, TQUDO_MAX_N_AVAILABLE

        if formulation not in ("qubo", "tqudo"):
            raise ValueError(
                "solver='brute_force' only supports formulation 'qubo' or 'tqudo', "
                f"got {formulation!r}."
            )
        cap_t = int(solver_config.get("brute_force_max_assignments_tqudo", 8**8))
        cap_q = int(solver_config.get("brute_force_max_assignments_qubo", 2**QUBO_MAX_BINARY_VARS))
        if formulation == "tqudo":
            if n_available > TQUDO_MAX_N_AVAILABLE:
                raise ValueError(
                    f"brute_force TQUDO requires n_cities - 1 <= {TQUDO_MAX_N_AVAILABLE} "
                    f"(full space has n^n assignments, max {TQUDO_MAX_N_AVAILABLE}**"
                    f"{TQUDO_MAX_N_AVAILABLE}); got n_cities={instance_config.n_cities}."
                )
            cardinal = n_available**n_available
            if cardinal > cap_t:
                raise ValueError(
                    f"brute_force TQUDO would enumerate {cardinal} assignments "
                    f"(n_cities={instance_config.n_cities}); exceeds "
                    f"brute_force_max_assignments_tqudo={cap_t}."
                )
        if formulation == "qubo":
            n_vars = n_available * n_available
            if n_vars > QUBO_MAX_BINARY_VARS:
                raise ValueError(
                    f"brute_force QUBO requires at most {QUBO_MAX_BINARY_VARS} binary variables "
                    f"(full space 2^n_vars); got n_vars={n_vars} (n_cities={instance_config.n_cities})."
                )
            cardinal = 1 << n_vars
            if cardinal > cap_q:
                raise ValueError(
                    f"brute_force QUBO would enumerate {cardinal} assignments; "
                    f"exceeds brute_force_max_assignments_qubo={cap_q}. "
                    "Raise the cap only if intended "
                    f"(n_cities={instance_config.n_cities})."
                )
