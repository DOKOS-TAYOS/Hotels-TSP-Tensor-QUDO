"""QAOA circuit implementation for Tensor QUDO problems using Cirq.

Cost layer uses multi-controlled phase gates for Etab and Ettprimeab.
Mirrors the CUDA-Q implementation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import sympy
from scipy.optimize import minimize

import cirq

from instance_gen_process.models import ProblemTQUDO
from utils.costs import calculate_tqudo_cost

if TYPE_CHECKING:
    from collections.abc import Sequence


def _multi_controlled_phase(
    controls: Sequence[cirq.Qid],
    target: cirq.Qid,
    phase: float | sympy.Expr,
) -> cirq.Operation:
    """Apply phase e^(i*phase) to target when all controls are |1⟩.

    Uses X gates to encode the desired basis state, then multi-controlled Z.
    ZPowGate(exponent) gives exp(i*π*exponent) on |1⟩, so exponent = phase/π.
    """
    exponent = phase / np.pi
    if not controls:
        return cirq.ZPowGate(exponent=exponent).on(target)
    gate = cirq.ZPowGate(exponent=exponent).controlled(len(controls))
    return gate.on(*controls, target)


def create_qaoa_circuit(
    depth: int,
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
) -> tuple[cirq.Circuit, dict[str, sympy.Symbol], list[cirq.Qid], int, int]:
    """Create the parametrized QAOA circuit for Tensor QUDO.

    Cost layer: multi-controlled phase for Etab[t,x_0,x_1] and Ettprimeab[t,t',x_t,x_t'].
    Mixer: rx(2*beta) on all qubits.

    Returns:
        Tuple (circuit, symbols, qubits, n_qudits, qubits_per_qudit).
    """
    n_qudits = Etab.shape[0]
    dimension_qudits = Etab.shape[1]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(dimension_qudits))))
    n_qubits_total = n_qudits * qubits_per_qudit

    qubits = list(cirq.LineQubit.range(n_qubits_total))
    q = [qubits[i * qubits_per_qudit : (i + 1) * qubits_per_qudit] for i in range(n_qudits)]

    symbols: dict[str, sympy.Symbol] = {}
    for k in range(depth):
        symbols[f"gamma_{k}"] = sympy.Symbol(f"gamma_{k}")
        symbols[f"beta_{k}"] = sympy.Symbol(f"beta_{k}")

    moments: list[cirq.OP_TREE] = []

    # Initial state |+>^n
    for i in range(n_qudits):
        moments.append(cirq.H.on_each(*q[i]))

    for k in range(depth):
        # Cost layer: Etab
        for t in range(n_qudits - 1):
            for x_0 in range(dimension_qudits):
                for x_1 in range(dimension_qudits):
                    e_val = Etab[t, x_0, x_1]
                    if abs(e_val) < 1e-14:
                        continue
                    phase = -symbols[f"gamma_{k}"] * e_val
                    target = q[t + 1][qubits_per_qudit - 1]
                    last_bit = (x_1 >> (qubits_per_qudit - 1)) & 1

                    # X on qubits where bit is 0 (prepare "all 1" for control)
                    for j in range(qubits_per_qudit):
                        if ((x_0 >> j) & 1) == 0:
                            moments.append(cirq.X(q[t][j]))
                    for j in range(qubits_per_qudit - 1):
                        if ((x_1 >> j) & 1) == 0:
                            moments.append(cirq.X(q[t + 1][j]))
                    if last_bit == 0:
                        moments.append(cirq.X(target))

                    controls = list(q[t]) + list(q[t + 1][:-1])
                    moments.append(_multi_controlled_phase(controls, target, phase))

                    # Undo X
                    for j in range(qubits_per_qudit):
                        if ((x_0 >> j) & 1) == 0:
                            moments.append(cirq.X(q[t][j]))
                    for j in range(qubits_per_qudit - 1):
                        if ((x_1 >> j) & 1) == 0:
                            moments.append(cirq.X(q[t + 1][j]))
                    if last_bit == 0:
                        moments.append(cirq.X(target))

        # Cost layer: Ettprimeab
        for t in range(n_qudits - 1):
            for t_prime in range(t + 1, n_qudits):
                for x_t in range(dimension_qudits):
                    for x_tp in range(dimension_qudits):
                        e_val = Ettprimeab[t, t_prime, x_t, x_tp]
                        if abs(e_val) < 1e-14:
                            continue
                        phase = -symbols[f"gamma_{k}"] * e_val
                        target = q[t_prime][qubits_per_qudit - 1]
                        last_bit = (x_tp >> (qubits_per_qudit - 1)) & 1

                        for j in range(qubits_per_qudit):
                            if ((x_t >> j) & 1) == 0:
                                moments.append(cirq.X(q[t][j]))
                        for j in range(qubits_per_qudit - 1):
                            if ((x_tp >> j) & 1) == 0:
                                moments.append(cirq.X(q[t_prime][j]))
                        if last_bit == 0:
                            moments.append(cirq.X(target))

                        controls = list(q[t]) + list(q[t_prime][:-1])
                        moments.append(_multi_controlled_phase(controls, target, phase))

                        for j in range(qubits_per_qudit):
                            if ((x_t >> j) & 1) == 0:
                                moments.append(cirq.X(q[t][j]))
                        for j in range(qubits_per_qudit - 1):
                            if ((x_tp >> j) & 1) == 0:
                                moments.append(cirq.X(q[t_prime][j]))
                        if last_bit == 0:
                            moments.append(cirq.X(target))

        # Mixer layer
        for i in range(n_qudits):
            for j in range(qubits_per_qudit):
                moments.append(cirq.rx(2.0 * symbols[f"beta_{k}"]).on(q[i][j]))

    circuit = cirq.Circuit(moments)
    return circuit, symbols, qubits, n_qudits, qubits_per_qudit


def _param_resolver(
    params: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
) -> cirq.ParamResolver:
    """Build ParamResolver from params array."""
    resolver_dict: dict[sympy.Symbol, float] = {}
    for k in range(depth):
        resolver_dict[symbols[f"gamma_{k}"]] = float(params[k])
        resolver_dict[symbols[f"beta_{k}"]] = float(params[depth + k])
    return cirq.ParamResolver(resolver_dict)


def bitstring_to_qudit_sequence(
    bitstring: str,
    n_qudits: int,
    qubits_per_qudit: int,
) -> np.ndarray:
    """Convert a measurement bitstring to qudit sequence (route)."""
    seq = np.zeros(n_qudits, dtype=np.int64)
    for i in range(n_qudits):
        start = i * qubits_per_qudit
        for j in range(qubits_per_qudit):
            if start + j < len(bitstring) and bitstring[start + j] == "1":
                seq[i] += 1 << j
    return seq


def evaluate_cost(
    params: np.ndarray,
    circuit: cirq.Circuit,
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    qubits: list[cirq.Qid],
    n_qudits: int,
    qubits_per_qudit: int,
    n_shots: int = 1000,
    seed: int | None = None,
) -> float:
    """Evaluate the QAOA cost by sampling and averaging TQUDO cost."""
    resolver = _param_resolver(params, symbols, depth)
    circuit_with_measure = circuit + cirq.measure(*qubits, key="m")
    simulator = cirq.Simulator(seed=seed)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)
    total = 0.0
    for row in result.measurements["m"]:
        bitstring = "".join(str(int(b)) for b in row)
        seq = bitstring_to_qudit_sequence(bitstring, n_qudits, qubits_per_qudit)
        total += calculate_tqudo_cost(problem, seq)
    return total / n_shots


def sample_solution(
    circuit: cirq.Circuit,
    params: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    qubits: list[cirq.Qid],
    n_shots: int = 1000,
    seed: int | None = None,
) -> dict[str, int]:
    """Sample bitstrings from the QAOA state. Returns dict of bitstring -> count."""
    resolver = _param_resolver(params, symbols, depth)
    circuit_with_measure = circuit + cirq.measure(*qubits, key="m")
    simulator = cirq.Simulator(seed=seed)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    counts: dict[str, int] = {}
    for row in result.measurements["m"]:
        bitstring = "".join(str(int(b)) for b in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def _most_probable(counts: dict[str, int], n_qubits: int) -> str:
    """Return the bitstring with highest count."""
    if not counts:
        return "0" * n_qubits
    return max(counts, key=lambda k: counts[k])


def optimize_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int | None = None,
    seed: int | None = None,
) -> tuple[float, np.ndarray, dict[str, int] | None]:
    """Optimize QAOA parameters to minimize the TQUDO cost."""
    if seed is not None:
        np.random.seed(seed)

    circuit, symbols, qubits, n_qudits, qubits_per_qudit = create_qaoa_circuit(
        depth, Etab, Ettprimeab
    )

    init_params = np.concatenate([
        np.random.uniform(0, 2 * np.pi, depth),
        np.random.uniform(0, np.pi, depth),
    ])

    def cost_fn(x: np.ndarray) -> float:
        return evaluate_cost(
            x, circuit, Etab, Ettprimeab, symbols, depth,
            qubits, n_qudits, qubits_per_qudit,
            n_shots=n_shots, seed=seed,
        )

    opt_result = minimize(
        cost_fn,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    samples: dict[str, int] | None = None
    if sample_shots is not None:
        samples = sample_solution(
            circuit, best_params, symbols, depth, qubits,
            n_shots=sample_shots, seed=seed,
        )
    return best_energy, best_params, samples


def run_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int = 1000,
    seed: int | None = None,
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution."""
    n_qudits = Etab.shape[0]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(Etab.shape[1]))))
    n_qubits_total = n_qudits * qubits_per_qudit

    best_energy, best_params, samples = optimize_qaoa(
        Etab,
        Ettprimeab,
        depth=depth,
        max_iter=max_iter,
        n_shots=n_shots,
        sample_shots=sample_shots,
        seed=seed,
    )

    best_bitstring = _most_probable(samples, n_qubits_total) if samples else "0" * n_qubits_total
    best_sequence = bitstring_to_qudit_sequence(best_bitstring, n_qudits, qubits_per_qudit)

    return {
        "energy": best_energy,
        "params": best_params,
        "samples": samples,
        "best_bitstring": best_bitstring,
        "best_sequence": best_sequence,
    }
