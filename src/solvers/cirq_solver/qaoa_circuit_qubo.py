"""QAOA circuit implementation for generic QUBO problems using Cirq.

Converts QUBO to Ising Hamiltonian, builds the QAOA ansatz, optimizes parameters,
and samples solutions. Mirrors the CUDA-Q implementation.
"""

from __future__ import annotations

import numpy as np
import sympy
from scipy.optimize import minimize

import cirq

from math_utils.qubo_ising import qubo_to_ising
from utils.optimizer import minimize_options


def _coerce_real_expectation(values: list[complex] | np.ndarray, imag_tol: float = 1e-9) -> float:
    """Convert expectation values to a real scalar with an explicit imag-part tolerance."""
    total = complex(np.sum(np.asarray(values, dtype=np.complex128)))
    if abs(total.imag) > imag_tol:
        raise ValueError(
            "Cirq returned an expectation value with a non-negligible imaginary component: "
            f"{total.imag}."
        )
    return float(total.real)


def ising_to_pauli_sum(
    h: np.ndarray,
    j_matrix: np.ndarray,
    qubits: list[cirq.Qid],
) -> cirq.PauliSum:
    """Build a Cirq PauliSum from Ising coefficients (h, J).

    H_C = sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j.

    Args:
        h: 1D array of linear coefficients.
        j_matrix: 2D array, only J[i,j] for i < j used.
        qubits: List of Cirq qubits.

    Returns:
        cirq.PauliSum representing the cost Hamiltonian.
    """
    n = len(h)
    if j_matrix.shape != (n, n) or len(qubits) != n:
        raise ValueError("Shape mismatch: h, j_matrix, qubits must be consistent")

    terms: list[cirq.PauliString] = []
    for i in range(n):
        if abs(h[i]) > 1e-14:
            terms.append(cirq.PauliString(h[i], cirq.Z(qubits[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(j_matrix[i, j]) > 1e-14:
                terms.append(
                    cirq.PauliString(
                        j_matrix[i, j],
                        cirq.Z(qubits[i]),
                        cirq.Z(qubits[j]),
                    )
                )

    if not terms:
        return cirq.PauliSum.from_pauli_strings([cirq.PauliString(0, cirq.Z(qubits[0]))])
    result = terms[0]
    for t in terms[1:]:
        result = result + t
    return result


def create_qaoa_circuit(
    depth: int,
    h_arr: np.ndarray,
    j_matrix: np.ndarray,
    qubits: list[cirq.Qid],
) -> tuple[cirq.Circuit, dict[str, sympy.Symbol]]:
    """Create the parametrized QAOA circuit for given (h, J).

    Cost layer: exp(-i gamma H_C) via rz and ZZ. Mixer: rx(2*beta) on each qubit.
    cirq.rz(rads) applies exp(-i*Z*rads/2), so rz(2*gamma*h) gives exp(-i*gamma*h*Z).
    cirq.ZZ**exponent applies exp(-i*pi*exponent*Z⊗Z/2), so ZZ**(2*gamma*J/pi).

    Args:
        depth: Number of QAOA layers.
        h_arr: Linear Ising coefficients.
        j_matrix: Coupling matrix.
        qubits: Cirq qubits.

    Returns:
        Tuple (circuit, param_map) where param_map has 'gamma_0'..'beta_{p-1}' -> Symbol.
    """
    n_qubits = len(h_arr)
    symbols: dict[str, sympy.Symbol] = {}
    for k in range(depth):
        symbols[f"gamma_{k}"] = sympy.Symbol(f"gamma_{k}")
        symbols[f"beta_{k}"] = sympy.Symbol(f"beta_{k}")

    moments: list[cirq.OP_TREE] = []

    # Initial state |+>^n
    moments.append(cirq.H.on_each(*qubits))

    for k in range(depth):
        # Cost layer: linear terms
        for i in range(n_qubits):
            if abs(h_arr[i]) > 1e-14:
                moments.append(
                    cirq.rz(2.0 * symbols[f"gamma_{k}"] * h_arr[i]).on(qubits[i])
                )
        # Cost layer: quadratic terms (CNOT-Rz-CNOT or ZZ)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                j_val = j_matrix[i, j]
                if abs(j_val) > 1e-14:
                    # exp(-i*gamma*J*Z_i*Z_j): cirq.ZZ**t gives exp(-i*pi*t*ZZ/2)
                    # need pi*t/2 = gamma*J => t = 2*gamma*J/pi
                    exponent = 2.0 * symbols[f"gamma_{k}"] * j_val / np.pi
                    moments.append(cirq.ZZ(qubits[i], qubits[j]) ** exponent)
        # Mixer layer
        for i in range(n_qubits):
            moments.append(cirq.rx(2.0 * symbols[f"beta_{k}"]).on(qubits[i]))

    circuit = cirq.Circuit(moments)
    return circuit, symbols


def _param_resolver(params: np.ndarray, symbols: dict[str, sympy.Symbol], depth: int) -> cirq.ParamResolver:
    """Build ParamResolver from params array."""
    resolver_dict: dict[sympy.Symbol, float] = {}
    for k in range(depth):
        resolver_dict[symbols[f"gamma_{k}"]] = float(params[k])
        resolver_dict[symbols[f"beta_{k}"]] = float(params[depth + k])
    return cirq.ParamResolver(resolver_dict)


def evaluate_cost(
    params: np.ndarray,
    circuit: cirq.Circuit,
    hamiltonian: cirq.PauliSum,
    symbols: dict[str, sympy.Symbol],
    depth: int,
) -> float:
    """Evaluate the QAOA cost (expectation of H_C) at given parameters."""
    resolver = _param_resolver(params, symbols, depth)
    simulator = cirq.Simulator()
    values = simulator.simulate_expectation_values(circuit, hamiltonian, param_resolver=resolver)
    return _coerce_real_expectation(values)


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

    # result.measurements['m'] is (n_shots, n_qubits)
    counts: dict[str, int] = {}
    for row in result.measurements["m"]:
        bitstring = "".join(str(int(b)) for b in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def optimize_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    sample_shots: int | None = None,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55, # se usa valor por defecto recomendado para grafo aleatorios probabilisticos en la referencia
) -> tuple[float, np.ndarray, dict[str, int] | None, float, list[float]]:
    """Optimize QAOA parameters to minimize the cost Hamiltonian.

    QUBO expectations are evaluated exactly with the simulator, so only
    ``sample_shots`` is used for the final state sampling step.
    """
    rng = np.random.default_rng(seed)

    h, j_matrix, offset = qubo_to_ising(qubo_matrix)
    n = len(h)
    qubits = list(cirq.LineQubit.range(n))
    hamiltonian = ising_to_pauli_sum(h, j_matrix, qubits)
    circuit, symbols = create_qaoa_circuit(depth, h, j_matrix, qubits)

    # TQA (Trotterized Quantum Annealing) initialization:
    # gamma_i = (i / p) * delta_t,  beta_i = (1 - i / p) * delta_t
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    init_params = np.concatenate([gamma_init, beta_init])

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        val = evaluate_cost(x, circuit, hamiltonian, symbols, depth)
        energy_history.append(val)
        return val

    initial_energy = evaluate_cost(init_params, circuit, hamiltonian, symbols, depth)

    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=minimize_options(optimizer, max_iter),
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun) + offset
    initial_energy = initial_energy + offset
    energy_history = [energy + offset for energy in energy_history]
    samples: dict[str, int] | None = None
    if sample_shots is not None:
        samples = sample_solution(
            circuit, best_params, symbols, depth, qubits, sample_shots, seed
        )
    return best_energy, best_params, samples, initial_energy, energy_history


def bitstring_to_binary(bitstring: str) -> np.ndarray:
    """Convert a measurement bitstring to a binary solution vector."""
    return np.array([int(b) for b in bitstring], dtype=np.int64)


def _most_probable(counts: dict[str, int], n_qubits: int) -> str:
    """Return the bitstring with highest count, or '0'*n if empty."""
    if not counts:
        return "0" * n_qubits
    return max(counts, key=lambda k: counts[k])


def run_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    sample_shots: int = 1000,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55, # se usa valor por defecto recomendado para grafo aleatorios probabilisticos en la referencia
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution.

    QUBO cost evaluation is exact in this backend, so ``sample_shots`` only
    affects the final solution sampling step.
    """
    best_energy, best_params, samples, initial_energy, energy_history = optimize_qaoa(
        qubo_matrix,
        depth=depth,
        max_iter=max_iter,
        sample_shots=sample_shots,
        seed=seed,
        optimizer=optimizer,
        delta_t=delta_t,
    )
    n = qubo_matrix.shape[0]
    best_bitstring = _most_probable(samples, n) if samples else "0" * n

    return {
        "energy": best_energy,
        "params": best_params,
        "samples": samples,
        "best_bitstring": best_bitstring,
        "best_binary": bitstring_to_binary(best_bitstring),
        "initial_energy": initial_energy,
        "energy_history": energy_history,
    }
