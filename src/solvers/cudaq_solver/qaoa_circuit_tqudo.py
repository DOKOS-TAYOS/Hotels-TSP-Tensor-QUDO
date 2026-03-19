"""QAOA circuit implementation for specific Tensor QUDO problems using CUDA-Q.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize

import cudaq

from instance_gen_process.models import ProblemTQUDO
from utils.costs import calculate_tqudo_cost


def create_qaoa_ansatz(
    depth: int,
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
) -> "cudaq.Kernel":
    """Create the QAOA ansatz kernel for Tensor QUDO (Etab, Ettprimeab).

    Uses n_qudits registers of qubits_per_qudit qubits each. Cost layer: multi-controlled
    phase gates for Etab[t,x_0,x_1] (consecutive qudits t, t+1) and Ettprimeab[t,t',x_t,x_t']
    (pairs t < t'). Mixer: rx(2*beta) on all qubits.

    Args:
        depth: Number of QAOA layers (p).
        Etab: 3D tensor (t, origin, destination) for travel/hotel costs.
        Ettprimeab: 4D tensor (t, t_prime, origin, destination) for penalties.

    Returns:
        Kernel with signature (gamma, beta).
    """
    n_qudits = Etab.shape[0]
    dimension_qudits = Etab.shape[1]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(dimension_qudits))))
    n_qubits_total = n_qudits * qubits_per_qudit

    @cudaq.kernel
    def qaoa_kernel(gamma: list[float], beta: list[float]):
        # n_qudits registers, each with qubits_per_qudit qubits. Access: q[i][j] = qubit j of qudit i
        q_full = cudaq.qvector(n_qubits_total)
        q = [q_full.slice(i * qubits_per_qudit, qubits_per_qudit) for i in range(n_qudits)]
        # Initial state |+>^n (h = Hadamard gate)
        for i in range(n_qudits):
            h(q[i])  # noqa: F821
        for k in range(depth):
            # Cost layer: Etab - phase gamma[k]*Etab[t,x_0,x_1] for each (t, x_0, x_1)
            # Multi-controlled P on last qubit of x_1; X before/after if that bit is 0 in x_1
            for t in range(n_qudits - 1):
                for x_0 in range(dimension_qudits):
                    for x_1 in range(dimension_qudits):
                        e_val = Etab[t, x_0, x_1]
                        if abs(e_val) < 1e-14:
                            continue
                        phase = -gamma[k] * e_val
                        target = q[t + 1][qubits_per_qudit - 1]
                        last_bit = (x_1 >> (qubits_per_qudit - 1)) & 1
                        # Controls: all qubits of qudit t and qudit t+1 except target
                        controls = []
                        for j in range(qubits_per_qudit):
                            bit_val = (x_0 >> j) & 1
                            if bit_val == 0:
                                x(q[t][j])  # noqa: F821
                            controls.append(q[t][j])
                        for j in range(qubits_per_qudit - 1):
                            bit_val = (x_1 >> j) & 1
                            if bit_val == 0:
                                x(q[t + 1][j])  # noqa: F821
                            controls.append(q[t + 1][j])
                        if last_bit == 0:
                            x(target)  # noqa: F821
                        r1.ctrl(controls, target, phase)  # noqa: F821
                        for j in range(qubits_per_qudit):
                            if ((x_0 >> j) & 1) == 0:
                                x(q[t][j])  # noqa: F821
                        for j in range(qubits_per_qudit - 1):
                            if ((x_1 >> j) & 1) == 0:
                                x(q[t + 1][j])  # noqa: F821
                        if last_bit == 0:
                            x(target)  # noqa: F821

            # Cost layer: Ettprimeab - phase gamma[k]*Ettprimeab[t,t',x_t,x_t'] for t < t'
            for t in range(n_qudits - 1):
                for t_prime in range(t + 1, n_qudits):
                    for x_t in range(dimension_qudits):
                        for x_tp in range(dimension_qudits):
                            e_val = Ettprimeab[t, t_prime, x_t, x_tp]
                            if abs(e_val) < 1e-14:
                                continue
                            phase = -gamma[k] * e_val
                            target = q[t_prime][qubits_per_qudit - 1]
                            last_bit = (x_tp >> (qubits_per_qudit - 1)) & 1
                            controls = []
                            for j in range(qubits_per_qudit):
                                if ((x_t >> j) & 1) == 0:
                                    x(q[t][j])  # noqa: F821
                                controls.append(q[t][j])
                            for j in range(qubits_per_qudit - 1):
                                if ((x_tp >> j) & 1) == 0:
                                    x(q[t_prime][j])  # noqa: F821
                                controls.append(q[t_prime][j])
                            if last_bit == 0:
                                x(target)  # noqa: F821
                            r1.ctrl(controls, target, phase)  # noqa: F821
                            for j in range(qubits_per_qudit):
                                if ((x_t >> j) & 1) == 0:
                                    x(q[t][j])  # noqa: F821
                            for j in range(qubits_per_qudit - 1):
                                if ((x_tp >> j) & 1) == 0:
                                    x(q[t_prime][j])  # noqa: F821
                            if last_bit == 0:
                                x(target)  # noqa: F821

            # Mixer layer: rx on all qubits of all qudits
            for i in range(n_qudits):
                for j in range(qubits_per_qudit):
                    rx(2.0 * beta[k], q[i][j])  # noqa: F821

    return qaoa_kernel


def evaluate_cost(
    params: np.ndarray,
    kernel: "cudaq.Kernel",
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int,
    n_shots: int = 1000,
) -> float:
    """Evaluate the QAOA cost by sampling and averaging TQUDO cost.

    Args:
        params: [gamma_1...gamma_p, beta_1...beta_p].
        kernel: QAOA kernel from create_qaoa_ansatz.
        Etab: 3D cost tensor.
        Ettprimeab: 4D penalty tensor.
        depth: QAOA depth.
        n_shots: Shots for cost estimation.

    Returns:
        Average TQUDO cost over samples.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    samples = cudaq.sample(kernel, gamma, beta, shots_count=n_shots)
    n_qudits = Etab.shape[0]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(Etab.shape[1]))))
    total = 0.0
    count = 0
    problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)
    for bitstring, cnt in samples.items():
        seq = bitstring_to_qudit_sequence(bitstring, n_qudits, qubits_per_qudit)
        total += calculate_tqudo_cost(problem, seq) * cnt
        count += cnt
    return total / count if count > 0 else 0.0


def sample_solution(
    kernel: "cudaq.Kernel",
    params: np.ndarray,
    depth: int,
    n_shots: int = 1000,
) -> "cudaq.SampleResult":
    """Sample bitstrings from the QAOA state at the given parameters.

    Args:
        kernel: QAOA kernel from create_qaoa_ansatz.
        params: [gamma_1...gamma_p, beta_1...beta_p].
        depth: Number of QAOA layers.
        n_shots: Number of measurement shots.

    Returns:
        cudaq.SampleResult: Dict-like object with bitstring counts.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    return cudaq.sample(kernel, gamma, beta, shots_count=n_shots)


def _minimize_options(method: str, max_iter: int) -> dict:
    """Build scipy minimize options dict for the given method."""
    opts: dict = {"maxiter": max_iter, "disp": False}
    if method in ("Nelder-Mead", "Powell"):
        opts["maxfev"] = max_iter
    if method == "L-BFGS-B":
        opts["maxfun"] = max_iter
    return opts


def optimize_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int | None = None,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55, # se
) -> tuple[float, np.ndarray, "cudaq.SampleResult | None", float, list[float]]:
    """Optimize QAOA parameters to minimize the TQUDO cost.

    Args:
        Etab: 3D cost tensor.
        Ettprimeab: 4D penalty tensor.
        depth: QAOA depth (number of layers).
        max_iter: Maximum optimizer iterations.
        n_shots: Shots per cost evaluation during optimization.
        sample_shots: If set, also sample the solution state (None = no sampling).
        seed: Random seed for initial parameters (None = no seed).
        optimizer: scipy optimizer method (COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead).
        delta_t: Time step for TQA initialization (default 0.55).

    Returns:
        Tuple of (best_energy, best_params, samples, initial_energy, energy_history).
        best_params: [gamma_1..gamma_p, beta_1..beta_p].
        samples: SampleResult when sample_shots is set, else None.
        initial_energy: Energy at init_params before optimization.
        energy_history: List of energies at each optimizer evaluation.
    """
    if seed is not None:
        np.random.seed(seed)
    kernel = create_qaoa_ansatz(depth, Etab, Ettprimeab)

    # TQA (Trotterized Quantum Annealing) initialization:
    # gamma_i = (i / p) * delta_t,  beta_i = (1 - i / p) * delta_t
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    init_params = np.concatenate([gamma_init, beta_init])

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        val = evaluate_cost(x, kernel, Etab, Ettprimeab, depth, n_shots=n_shots)
        energy_history.append(val)
        return val

    initial_energy = evaluate_cost(
        init_params, kernel, Etab, Ettprimeab, depth, n_shots=n_shots
    )

    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=_minimize_options(optimizer, max_iter),
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    samples: "cudaq.SampleResult | None" = None
    if sample_shots is not None:
        samples = sample_solution(kernel, best_params, depth, n_shots=sample_shots)
    return best_energy, best_params, samples, initial_energy, energy_history


def bitstring_to_qudit_sequence(
    bitstring: str,
    n_qudits: int,
    qubits_per_qudit: int,
) -> np.ndarray:
    """Convert a measurement bitstring to qudit sequence (route).

    Args:
        bitstring: String of '0' and '1', length n_qudits * qubits_per_qudit.
        n_qudits: Number of qudit registers.
        qubits_per_qudit: Qubits per qudit.

    Returns:
        1D array of qudit values (city per timestep), shape (n_qudits,).
    """
    seq = np.zeros(n_qudits, dtype=np.int64)
    for i in range(n_qudits):
        start = i * qubits_per_qudit
        for j in range(qubits_per_qudit):
            if start + j < len(bitstring) and bitstring[start + j] == "1":
                seq[i] += 1 << j
    return seq


def run_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int = 1000,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55, # se usa valor por defecto recomendado para grafo aleatorios probabilisticos en la referencia
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution.

    Args:
        Etab: 3D cost tensor.
        Ettprimeab: 4D penalty tensor.
        depth: QAOA depth.
        max_iter: Optimizer iterations.
        n_shots: Shots per cost evaluation during optimization.
        sample_shots: Shots for sampling the final state.
        seed: Random seed.
        optimizer: scipy optimizer method (COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead).
        delta_t: Time step for TQA initialization (default 0.55).

    Returns:
        Dict with keys: energy, params, samples, best_bitstring, best_sequence,
        initial_energy, energy_history.
        best_bitstring: Most frequent bitstring from sampling.
        best_sequence: Qudit sequence (route) from best_bitstring.
    """
    n_qudits = Etab.shape[0]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(Etab.shape[1]))))
    n_qubits_total = n_qudits * qubits_per_qudit

    best_energy, best_params, samples, initial_energy, energy_history = optimize_qaoa(
        Etab,
        Ettprimeab,
        depth=depth,
        max_iter=max_iter,
        n_shots=n_shots,
        sample_shots=sample_shots,
        seed=seed,
        optimizer=optimizer,
        delta_t=delta_t,
    )

    best_bitstring = (
        samples.most_probable() if samples else "0" * n_qubits_total
    )
    best_sequence = bitstring_to_qudit_sequence(
        best_bitstring, n_qudits, qubits_per_qudit
    )

    return {
        "energy": best_energy,
        "params": best_params,
        "samples": samples,
        "best_bitstring": best_bitstring,
        "best_sequence": best_sequence,
        "initial_energy": initial_energy,
        "energy_history": energy_history,
    }
