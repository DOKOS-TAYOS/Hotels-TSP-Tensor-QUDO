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

from solvers.cudaq_solver.cudaq_target import ensure_cudaq_target

ensure_cudaq_target()


def qubo_to_ising(qubo_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a symmetric QUBO matrix to Ising form (h, J).

    Uses the standard transformation x_i = (1 - s_i) / 2 with s_i in {-1, +1}.
    For symmetric QUBO: x^T Q x = sum_i Q_ii x_i + sum_{i<j} 2 Q_ij x_i x_j.

    Args:
        qubo_matrix: Symmetric matrix of shape (n, n). Represents the QUBO
            objective min x^T Q x over binary x.

    Returns:
        Tuple (h, J):
        - h: 1D array of shape (n,) with linear Ising coefficients h_i.
              E_ising = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j.
        - J: 2D array of shape (n, n) with coupling coefficients.
              Only entries with i < j are used; J is upper-triangular by convention.
              J_ij = Q_ij / 4 for i < j.

    Output format:
        h[i] = -Q_ii/2 - sum_{j != i} Q_ij/2
        J[i,j] = Q_ij/4 for i < j (rest can be 0)
    """
    n = qubo_matrix.shape[0]
    if qubo_matrix.shape[1] != n:
        raise ValueError("qubo_matrix must be square")
    
    # Ensure symmetry of QUBO matrix
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        raise ValueError("qubo_matrix must be symmetric")
    
    # Linear: h_i = -sum_j Q_ij/2 (from x_i = (1-s_i)/2 substitution)
    h = -0.5 * np.sum(qubo_matrix, axis=1)
    
    # Quadratic: J_ij = Q_ij/4 for i < j
    # Use vectorized upper triangular extraction for efficiency
    j_full = np.triu(qubo_matrix, k=1) / 4.0
    
    return h, j_full


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
    Parameters: gamma, beta each of length depth. h_arr and j_matrix are captured
    from the closure (no need to pass them to observe/sample).

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

    @cudaq.kernel
    def qaoa_kernel(gamma: list[float], beta: list[float]):
        # rz, rx, x, h are injected by cudaq.kernel; h_arr, j_matrix from closure
        q = cudaq.qvector(n_qubits)
        # Initial state |+>^n (h = Hadamard gate)
        h(q)  # noqa: F821
        for k in range(depth):
            # Cost layer: linear terms rz(2*gamma*h_i)
            for i in range(n_qubits):
                if abs(h_arr[i]) > 1e-14:
                    rz(2.0 * gamma[k] * h_arr[i], q[i])  # noqa: F821
            # Cost layer: quadratic terms CNOT-Rz-CNOT (no-op when J_ij=0)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    j_val = j_matrix[i, j]
                    if abs(j_val) > 1e-14:
                        x.ctrl(q[i], q[j])  # noqa: F821
                        rz(2.0 * gamma[k] * j_val, q[j])  # noqa: F821
                        x.ctrl(q[i], q[j])  # noqa: F821
            # Mixer layer
            for i in range(n_qubits):
                rx(2.0 * beta[k], q[i])  # noqa: F821

    return qaoa_kernel


def evaluate_cost(
    params: np.ndarray,
    kernel: "cudaq.Kernel",
    hamiltonian: "cudaq.SpinOperator",
    depth: int,
) -> float:
    """Evaluate the QAOA cost (expectation of H_C) at given parameters.

    Args:
        params: Concatenated [gamma_1...gamma_p, beta_1...beta_p], length 2*depth.
        kernel: QAOA kernel from create_qaoa_ansatz (captures h, J; takes gamma, beta).
        hamiltonian: Cost Hamiltonian H_C (spin_op).
        depth: Number of QAOA layers.

    Returns:
        float: Expectation value <psi(params)|H_C|psi(params)>.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    result = cudaq.observe(kernel, hamiltonian, gamma, beta)
    return float(result.expectation())


def sample_solution(
    kernel: "cudaq.Kernel",
    params: np.ndarray,
    depth: int,
    n_shots: int = 1000,
) -> "cudaq.SampleResult":
    """Sample bitstrings from the QAOA state at the given parameters.

    Args:
        kernel: QAOA kernel from create_qaoa_ansatz (captures h, J).
        params: [gamma_1..gamma_p, beta_1..beta_p].
        depth: Number of QAOA layers.
        n_shots: Number of measurement shots.

    Returns:
        cudaq.SampleResult: Dict-like object with bitstring counts.
        e.g. result["001"] = 42. Use .items() or iterate for (bitstring, count).
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    return cudaq.sample(kernel, gamma, beta, shots_count=n_shots)

def optimize_qaoa(
    qubo_matrix: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int | None = None,
    seed: int | None = None,
) -> tuple[float, np.ndarray, "cudaq.SampleResult | None"]:
    """Optimize QAOA parameters to minimize the cost Hamiltonian.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        depth: QAOA depth (number of layers).
        max_iter: Maximum optimizer iterations.
        n_shots: If set, also sample the solution state (None = no sampling).
        seed: Random seed for initial parameters (None = no seed).

    Returns:
        Tuple of (best_energy, best_params, samples).
        best_params: [gamma_1..gamma_p, beta_1..beta_p].
        samples: SampleResult when n_shots is set, else None.
    """
    if seed is not None:
        np.random.seed(seed)
    h, j_matrix = qubo_to_ising(qubo_matrix)
    hamiltonian = ising_to_spin_op(h, j_matrix)
    kernel = create_qaoa_ansatz(depth, h, j_matrix)

    init_params = np.concatenate([
        np.random.uniform(0, 2 * np.pi, depth),
        np.random.uniform(0, np.pi, depth),
    ])

    def cost_fn(x: np.ndarray) -> float:
        return evaluate_cost(x, kernel, hamiltonian, depth)

    opt_result = minimize(
        cost_fn,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    samples: "cudaq.SampleResult | None" = None
    if n_shots is not None:
        samples = sample_solution(kernel, best_params, depth, n_shots)
    return best_energy, best_params, samples


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
    n_shots: int = 1000,
    seed: int | None = None,
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution.

    Args:
        qubo_matrix: Symmetric QUBO matrix.
        depth: QAOA depth.
        max_iter: Optimizer iterations.
        n_shots: Shots for sampling the final state.
        seed: Random seed.

    Returns:
        Dict with keys: energy, params, samples, best_bitstring, best_binary.
        best_bitstring: Most frequent bitstring from sampling.
        best_binary: bitstring_to_binary(best_bitstring).
    """
    best_energy, best_params, samples = optimize_qaoa(
        qubo_matrix,
        depth=depth,
        max_iter=max_iter,
        n_shots=n_shots,
        seed=seed,
    )
    n = qubo_matrix.shape[0]

    # SampleResult has most_probable() for the highest-count bitstring
    best_bitstring = samples.most_probable() if samples else "0" * n

    return {
        "energy": best_energy,
        "params": best_params,
        "samples": samples,
        "best_bitstring": best_bitstring,
        "best_binary": bitstring_to_binary(best_bitstring),
    }

