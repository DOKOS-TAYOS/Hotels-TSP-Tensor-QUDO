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
from solvers.cirq_solver.noise_model import get_simulator
from solvers.noise import NoiseConfig
from utils.cooperative_stop import raise_if_solver_stop_requested
from utils.optimizer import minimize_options
from solvers.base import OptimizerType
from utils.progress import reporter
from utils.qaoa_helpers import bitstring_to_binary, most_probable_key, tqa_init_params


def _coerce_real_expectation(values: list[complex] | np.ndarray, imag_tol: float = 1e-9) -> float:
    """Sum *values* and return the real part if the imaginary part is negligible.

    Args:
        values: Complex expectation contributions.
        imag_tol: Maximum allowed magnitude of the imaginary sum.

    Returns:
        Real part of the total.

    Raises:
        ValueError: If the imaginary part exceeds *imag_tol*.
    """
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
    """Map flat ``[gamma…, beta…]`` to SymPy symbols for Cirq.

    Args:
        params: Length ``2 * depth`` parameter vector.
        symbols: ``gamma_k`` / ``beta_k`` symbol map from :func:`create_qaoa_circuit`.
        depth: QAOA depth p.

    Returns:
        ``cirq.ParamResolver`` for the circuit.
    """
    resolver_dict: dict[sympy.Symbol, float] = {}
    for k in range(depth):
        resolver_dict[symbols[f"gamma_{k}"]] = float(params[k])
        resolver_dict[symbols[f"beta_{k}"]] = float(params[depth + k])
    return cirq.ParamResolver(resolver_dict)


def evaluate_cost(
    params: np.ndarray,
    circuit_with_measure: cirq.Circuit,
    qubo_matrix: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    n_shots: int,
    simulator: cirq.SimulatesSamples,
) -> float:
    """Estimate mean ``xᵀ Q x`` by sampling the QAOA circuit.

    Args:
        params: Flat QAOA angles.
        circuit_with_measure: Parametrised circuit with terminal measurements.
        qubo_matrix: Symmetric QUBO matrix (same units as stored problem).
        symbols: Symbol map from :func:`create_qaoa_circuit`.
        depth: QAOA depth p.
        n_shots: Samples per evaluation.
        simulator: Cirq sampler (noise may be on the circuit).

    Returns:
        Sample-averaged QUBO objective.
    """
    resolver = _param_resolver(params, symbols, depth)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    total = 0.0
    for row in result.measurements["m"]:
        x = row.astype(np.float64)
        total += float(x @ qubo_matrix @ x)
    return total / n_shots


def sample_solution(
    circuit_with_measure: cirq.Circuit,
    params: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    n_shots: int,
    simulator: cirq.SimulatesSamples,
) -> dict[str, int]:
    """Draw bitstring samples from the QAOA state at *params*.

    Args:
        circuit_with_measure: Parametrised circuit with measurements.
        params: Flat QAOA angles.
        symbols: Symbol map from :func:`create_qaoa_circuit`.
        depth: QAOA depth p.
        n_shots: Number of samples.
        simulator: Cirq sampler.

    Returns:
        Histogram mapping bitstrings to counts.
    """
    resolver = _param_resolver(params, symbols, depth)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    counts: dict[str, int] = {}
    for row in result.measurements["m"]:
        bitstring = "".join(str(int(b)) for b in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def optimize_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int | None = None,
    seed: int | None = None,
    optimizer: OptimizerType = "COBYLA",
    delta_t: float = 0.55,
    optimizer_tol: float = 1e-6,
    noise_config: NoiseConfig | None = None,
) -> tuple[float, np.ndarray, dict[str, int] | None, dict[str, int] | None, float, list[float]]:
    """Optimize QAOA parameters to minimize sampled QUBO cost.

    Cost is evaluated by drawing ``n_shots`` bitstrings and averaging
    ``xᵀ Q x``, consistent with the TQUDO sampling backend.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        depth: QAOA layers p.
        max_iter: Classical optimizer budget.
        n_shots: Shots per objective evaluation.
        sample_shots: If set, record bitstring histograms at init and best
            parameters; if None, skip extra sampling.
        seed: Optional simulator seed.
        optimizer: SciPy ``minimize`` method.
        delta_t: TQA initial parameter scale.
        optimizer_tol: SciPy ``minimize`` stopping tolerance.
        noise_config: Optional noise; None or disabled uses state vector.

    Returns:
        ``(best_energy, best_params, initial_samples, final_samples,
        initial_energy, energy_history)``; sample dicts are None when
        ``sample_shots`` is None.
    """
    h, j_matrix, offset = qubo_to_ising(qubo_matrix)
    n = len(h)
    qubits = list(cirq.LineQubit.range(n))
    circuit, symbols = create_qaoa_circuit(depth, h, j_matrix, qubits)

    simulator, noise_model = get_simulator(noise_config, seed=seed)
    circuit_with_measure = circuit + cirq.measure(*qubits, key="m")
    if noise_model is not None:
        circuit_with_measure = circuit_with_measure.with_noise(noise_model)

    init_params = tqa_init_params(depth, delta_t)

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        raise_if_solver_stop_requested()
        val = evaluate_cost(
            x, circuit_with_measure, qubo_matrix, symbols, depth,
            n_shots, simulator,
        )
        energy_history.append(val)
        reporter.opt_step(len(energy_history), max_iter, val)
        return val

    raise_if_solver_stop_requested()
    initial_energy = evaluate_cost(
        init_params, circuit_with_measure, qubo_matrix, symbols, depth,
        n_shots, simulator,
    )

    initial_samples: dict[str, int] | None = None
    if sample_shots is not None:
        raise_if_solver_stop_requested()
        initial_samples = sample_solution(
            circuit_with_measure, init_params, symbols, depth,
            sample_shots, simulator,
        )

    raise_if_solver_stop_requested()
    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=minimize_options(optimizer, max_iter, optimizer_tol),
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    final_samples: dict[str, int] | None = None
    if sample_shots is not None:
        raise_if_solver_stop_requested()
        final_samples = sample_solution(
            circuit_with_measure, best_params, symbols, depth,
            sample_shots, simulator,
        )
    return best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history


def run_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int = 1000,
    seed: int | None = None,
    optimizer: OptimizerType = "COBYLA",
    delta_t: float = 0.55,
    optimizer_tol: float = 1e-6,
    noise_config: NoiseConfig | None = None,
) -> dict:
    """Run QUBO QAOA end-to-end and return energies, angles, and best bitstring.

    Args:
        qubo_matrix: Symmetric QUBO ``Q``.
        depth: QAOA layers p.
        max_iter: Classical optimizer budget.
        n_shots: Shots per objective evaluation.
        sample_shots: Shots for final (and initial) histograms.
        seed: Optional RNG seed.
        optimizer: SciPy method name.
        delta_t: TQA initialization scale.
        optimizer_tol: SciPy ``minimize`` stopping tolerance.
        noise_config: Optional noise configuration.

    Returns:
        Dict with ``energy``, ``params``, ``initial_samples``, ``final_samples``,
        ``best_bitstring``, ``best_binary``, ``initial_energy``, ``energy_history``.
    """
    best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history = (
        optimize_qaoa(
            qubo_matrix,
            depth=depth,
            max_iter=max_iter,
            n_shots=n_shots,
            sample_shots=sample_shots,
            seed=seed,
            optimizer=optimizer,
            delta_t=delta_t,
            optimizer_tol=optimizer_tol,
            noise_config=noise_config,
        )
    )
    n = qubo_matrix.shape[0]
    best_bitstring = most_probable_key(final_samples, "0" * n) if final_samples else "0" * n

    return {
        "energy": best_energy,
        "params": best_params,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "best_bitstring": best_bitstring,
        "best_binary": bitstring_to_binary(best_bitstring),
        "initial_energy": initial_energy,
        "energy_history": energy_history,
    }
