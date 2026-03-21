"""QAOA circuit implementation for generic QUBO problems using CUDA-Q.

Converts QUBO to Ising Hamiltonian, builds the QAOA ansatz, optimizes parameters,
and samples solutions. Designed so the base Hamiltonian (h, J) can be modified
before building the spin_op.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import cudaq
from cudaq import spin

from math_utils.qubo_ising import qubo_to_ising
from solvers.cudaq_solver.cudaq_target import ensure_cudaq_target
from solvers.cudaq_solver.noise_model import get_noise_model
from solvers.noise import NoiseConfig
from utils.optimizer import minimize_options


def ising_to_spin_op(h: np.ndarray, j_matrix: np.ndarray) -> "cudaq.SpinOperator":
    """Build a CUDA-Q spin_op from Ising coefficients (h, J).

    Maps s_i to Pauli Z_i (eigenvalue +1 for |0>, -1 for |1>).
    H_C = sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j.

    Args:
        h: 1D array of shape (n,) with linear coefficients.
        j_matrix: 2D array of shape (n, n). Only J[i,j] for i < j is used.

    Returns:
        cudaq.SpinOperator representing the cost Hamiltonian H_C.
        Can be passed to cudaq.observe() for expectation value computation.
    """
    n = len(h)
    if j_matrix.shape != (n, n):
        raise ValueError("j_matrix must be (n, n) with n = len(h)")

    # Start with zero operator; spin_op supports algebraic sum
    ham = 0.0
    for i in range(n):
        if abs(h[i]) > 1e-14:
            ham = ham + h[i] * spin.z(i)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(j_matrix[i, j]) > 1e-14:
                ham = ham + j_matrix[i, j] * spin.z(i) * spin.z(j)
    return ham


def create_qaoa_ansatz(
    depth: int,
    h_arr: np.ndarray,
    j_matrix: np.ndarray,
) -> "cudaq.Kernel":
    """Create the QAOA ansatz kernel for given (h, J).

    The kernel prepares |psi(gamma, beta)> = prod_k exp(-i beta_k H_M) exp(-i gamma_k H_C) |+>^n.
    Parameters: gamma, beta each of length depth. All Ising coefficients are
    pre-computed as plain Python lists before kernel definition to avoid
    capturing NumPy arrays in the CUDA-Q JIT closure.

    Cost layer: exp(-i gamma h_i Z_i) = rz(2*gamma*h_i); exp(-i gamma J_ij Z_i Z_j)
    uses CNOT-Rz-CNOT. Mixer: exp(-i beta H_M) = rx(2*beta) on each qubit.

    Args:
        depth: Number of QAOA layers (p).
        h_arr: Linear Ising coefficients, shape (n,). n_qubits = len(h_arr).
        j_matrix: Coupling matrix, shape (n, n). Only i<j used.

    Returns:
        Kernel with signature (gamma, beta). Use observe(kernel, hamiltonian, gamma, beta).
    """
    n_qubits = len(h_arr)

    # Pre-compute non-zero linear terms as plain Python lists
    # (cudaq.kernel JIT cannot capture NumPy arrays from closures)
    h_nonzero_idx: list[int] = []
    h_nonzero_vals: list[float] = []
    for i in range(n_qubits):
        if abs(float(h_arr[i])) > 1e-14:
            h_nonzero_idx.append(i)
            h_nonzero_vals.append(float(h_arr[i]))
    n_h = len(h_nonzero_idx)

    # Pre-compute non-zero coupling terms as plain Python lists
    jj_pairs_i: list[int] = []
    jj_pairs_j: list[int] = []
    jj_vals: list[float] = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            val = float(j_matrix[i, j])
            if abs(val) > 1e-14:
                jj_pairs_i.append(i)
                jj_pairs_j.append(j)
                jj_vals.append(val)
    n_jj = len(jj_vals)

    @cudaq.kernel
    def qaoa_kernel(gamma: list[float], beta: list[float]):
        q = cudaq.qvector(n_qubits)
        # Initial state |+>^n (h = Hadamard gate)
        h(q)  # noqa: F821
        for k in range(depth):
            # Cost layer: linear terms rz(2*gamma*h_i)
            for idx in range(n_h):
                rz(2.0 * gamma[k] * h_nonzero_vals[idx], q[h_nonzero_idx[idx]])  # noqa: F821
            # Cost layer: quadratic terms CNOT-Rz-CNOT
            for idx in range(n_jj):
                qi = jj_pairs_i[idx]
                qj = jj_pairs_j[idx]
                x.ctrl(q[qi], q[qj])  # noqa: F821
                rz(2.0 * gamma[k] * jj_vals[idx], q[qj])  # noqa: F821
                x.ctrl(q[qi], q[qj])  # noqa: F821
            # Mixer layer
            for i in range(n_qubits):
                rx(2.0 * beta[k], q[i])  # noqa: F821

    return qaoa_kernel


def evaluate_cost(
    params: np.ndarray,
    kernel: "cudaq.Kernel",
    qubo_matrix: np.ndarray,
    depth: int,
    n_shots: int = 500,
    noise_config: NoiseConfig | None = None,
) -> float:
    """Evaluate the QAOA cost by sampling and averaging QUBO cost.

    Uses the same sampling-based approach as the TQUDO backend so that
    both formulations are compared on equal footing.

    Args:
        params: Concatenated [gamma_1...gamma_p, beta_1...beta_p], length 2*depth.
        kernel: QAOA kernel from create_qaoa_ansatz (captures h, J; takes gamma, beta).
        qubo_matrix: The original QUBO matrix for direct cost evaluation.
        depth: Number of QAOA layers.
        n_shots: Shots for cost estimation.
        noise_config: Optional noise parameters.

    Returns:
        float: Average x^T Q x over sampled bitstrings.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    noise_model = get_noise_model(noise_config)
    sample_kwargs: dict = {"shots_count": n_shots}
    if noise_model is not None:
        sample_kwargs["noise_model"] = noise_model
    samples = cudaq.sample(kernel, gamma, beta, **sample_kwargs)
    total = 0.0
    count = 0
    for bitstring, cnt in samples.items():
        x = bitstring_to_binary(bitstring).astype(np.float64)
        total += float(x @ qubo_matrix @ x) * cnt
        count += cnt
    return total / count if count > 0 else 0.0


def sample_solution(
    kernel: "cudaq.Kernel",
    params: np.ndarray,
    depth: int,
    n_shots: int = 1000,
    noise_config: NoiseConfig | None = None,
) -> "cudaq.SampleResult":
    """Sample bitstrings from the QAOA state at the given parameters.

    Args:
        kernel: QAOA kernel from create_qaoa_ansatz (captures h, J).
        params: [gamma_1..gamma_p, beta_1..beta_p].
        depth: Number of QAOA layers.
        n_shots: Number of measurement shots.
        noise_config: Optional noise parameters.

    Returns:
        cudaq.SampleResult: Dict-like object with bitstring counts.
        e.g. result["001"] = 42. Use .items() or iterate for (bitstring, count).
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    noise_model = get_noise_model(noise_config)
    sample_kwargs: dict = {"shots_count": n_shots}
    if noise_model is not None:
        sample_kwargs["noise_model"] = noise_model
    return cudaq.sample(kernel, gamma, beta, **sample_kwargs)


def optimize_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int | None = None,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55,
    noise_config: NoiseConfig | None = None,
) -> tuple[
    float,
    np.ndarray,
    "cudaq.SampleResult | None",
    "cudaq.SampleResult | None",
    float,
    list[float],
]:
    """Optimize QAOA parameters to minimize the cost Hamiltonian.

    Cost is evaluated by sampling ``n_shots`` bitstrings and averaging
    x^T Q x, consistent with the TQUDO backend.  ``sample_shots`` controls
    the final solution-sampling step.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        depth: QAOA depth (number of layers).
        max_iter: Maximum optimizer iterations.
        n_shots: Shots per cost evaluation during optimization.
        sample_shots: If set, also sample the solution state (None = no sampling).
        seed: Random seed for initial parameters (None = no seed).
        optimizer: scipy optimizer method (COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead).
        delta_t: Time step for TQA initialization (default 0.55).

    Returns:
        Tuple of (best_energy, best_params, initial_samples, final_samples,
        initial_energy, energy_history).
        best_params: [gamma_1..gamma_p, beta_1..beta_p].
        initial_samples: SampleResult at TQA init params when sample_shots is set, else None.
        final_samples: SampleResult at best_params when sample_shots is set, else None.
        initial_energy: Energy at init_params before optimization.
        energy_history: List of energies at each optimizer evaluation.
    """
    ensure_cudaq_target(noise_config)
    if seed is not None:
        cudaq.set_random_seed(seed)

    h, j_matrix, offset = qubo_to_ising(qubo_matrix)
    kernel = create_qaoa_ansatz(depth, h, j_matrix)

    # TQA (Trotterized Quantum Annealing) initialization:
    # gamma_i = (i / p) * delta_t,  beta_i = (1 - i / p) * delta_t
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    init_params = np.concatenate([gamma_init, beta_init])

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        val = evaluate_cost(
            x, kernel, qubo_matrix, depth, n_shots=n_shots,
            noise_config=noise_config,
        )
        energy_history.append(val)
        return val

    initial_energy = evaluate_cost(
        init_params, kernel, qubo_matrix, depth, n_shots=n_shots,
        noise_config=noise_config,
    )

    initial_samples: "cudaq.SampleResult | None" = None
    if sample_shots is not None:
        initial_samples = sample_solution(
            kernel, init_params, depth, sample_shots,
            noise_config=noise_config,
        )

    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=minimize_options(optimizer, max_iter),
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    final_samples: "cudaq.SampleResult | None" = None
    if sample_shots is not None:
        final_samples = sample_solution(
            kernel, best_params, depth, sample_shots,
            noise_config=noise_config,
        )
    return best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history


def bitstring_to_binary(bitstring: str) -> np.ndarray:
    """Convert a measurement bitstring to a binary solution vector.

    Convention: qubit i in |0> -> x_i=0, qubit i in |1> -> x_i=1.

    Args:
        bitstring: String of '0' and '1', e.g. "1010".

    Returns:
        1D array of 0s and 1s, shape (len(bitstring),).
    """
    return np.array([int(b) for b in bitstring], dtype=np.int64)


def run_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int = 1000,
    seed: int | None = None,
    optimizer: str = "COBYLA",
    delta_t: float = 0.55,
    noise_config: NoiseConfig | None = None,
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution.

    Cost evaluation uses ``n_shots`` samples per optimizer step (sampling-based,
    consistent with the TQUDO backend).  ``sample_shots`` controls the final
    solution-sampling step.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        depth: QAOA depth.
        max_iter: Optimizer iterations.
        n_shots: Shots per cost evaluation during optimization.
        sample_shots: Shots for sampling the final state.
        seed: Random seed.
        optimizer: scipy optimizer method (COBYLA, Powell, L-BFGS-B, SLSQP, Nelder-Mead).
        delta_t: Time step for TQA initialization (default0.55).

    Returns:
        Dict with keys: energy, params, initial_samples, final_samples, best_bitstring,
        best_binary, initial_energy, energy_history.
        initial_samples: SampleResult at TQA init params (before optimization).
        final_samples: SampleResult at best params (after optimization).
        best_bitstring: Most frequent bitstring from final_samples.
        best_binary: bitstring_to_binary(best_bitstring).
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
            noise_config=noise_config,
        )
    )
    n = qubo_matrix.shape[0]

    # SampleResult has most_probable() for the highest-count bitstring
    best_bitstring = final_samples.most_probable() if final_samples else "0" * n

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
