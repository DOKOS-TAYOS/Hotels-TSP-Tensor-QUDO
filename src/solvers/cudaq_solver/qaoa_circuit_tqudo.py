"""QAOA circuit implementation for Tensor-QUDO problems using CUDA-Q."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import minimize

import cudaq

from instance_gen_process.models import ProblemTQUDO
from solvers.cudaq_solver.cudaq_target import ensure_cudaq_target
from solvers.cudaq_solver.noise_model import get_noise_model
from solvers.noise import NoiseConfig
from utils.costs import calculate_tqudo_cost
from utils.optimizer import minimize_options
from utils.progress import reporter

logger = logging.getLogger(__name__)


def _is_power_of_two(value: int) -> bool:
    """Return True when *value* is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _validate_tqudo_shapes(Etab: np.ndarray, Ettprimeab: np.ndarray) -> tuple[int, int]:
    """Validate Tensor-QUDO tensor shapes and return basic dimensions."""
    if Etab.ndim != 3:
        raise ValueError(f"Etab must be a rank-3 tensor, got shape {Etab.shape}.")
    if Ettprimeab.ndim != 4:
        raise ValueError(
            f"Ettprimeab must be a rank-4 tensor, got shape {Ettprimeab.shape}."
        )

    n_qudits = Etab.shape[0]
    dimension_qudits = Etab.shape[1]
    expected_ett_shape = (n_qudits, n_qudits, dimension_qudits, dimension_qudits)
    if Etab.shape[2] != dimension_qudits:
        raise ValueError(
            "Etab must be square in its state dimensions, "
            f"got shape {Etab.shape}."
        )
    if Ettprimeab.shape != expected_ett_shape:
        raise ValueError(
            "Ettprimeab shape does not match Etab. "
            f"Expected {expected_ett_shape}, got {Ettprimeab.shape}."
        )
    if not _is_power_of_two(dimension_qudits):
        raise ValueError(
            f"CUDA-Q qubit-emulation TQUDO requires the qudit dimension "
            f"(n_cities - 1 = {dimension_qudits}) to be a power of two. "
            f"Use the native-qudit Cirq backend for arbitrary dimensions."
        )

    return n_qudits, dimension_qudits


def _nonzero_etab_terms(Etab: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Return sparse adjacent-qudit terms as (left, right, x_left, x_right, coeff).

    Only adjacent timesteps (t, t+1) are included, so the first axis is
    sliced to ``Etab[:n_qudits-1]`` before extracting non-zeros.
    """
    n_qudits = Etab.shape[0]
    adjacent = Etab[:n_qudits - 1]
    idx = np.argwhere(np.abs(adjacent) > 1e-14)
    return [
        (int(t), int(t) + 1, int(xl), int(xr), float(adjacent[t, xl, xr]))
        for t, xl, xr in idx
    ]


def _nonzero_ett_terms(Ettprimeab: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Return sparse long-range terms as (left, right, x_left, x_right, coeff).

    Only upper-triangular timestep pairs (t < t') are included.
    """
    n_qudits = Ettprimeab.shape[0]
    mask = np.abs(Ettprimeab) > 1e-14
    triu_mask = np.triu(np.ones((n_qudits, n_qudits), dtype=bool), k=1)
    mask &= triu_mask[:, :, np.newaxis, np.newaxis]
    idx = np.argwhere(mask)
    return [
        (int(t), int(tp), int(xl), int(xr), float(Ettprimeab[t, tp, xl, xr]))
        for t, tp, xl, xr in idx
    ]


def _apply_state_conditioned_phase(
    kernel: "cudaq.Kernel",
    q_full: "cudaq.QuakeValue",
    qubits_per_qudit: int,
    left_qudit: int,
    right_qudit: int,
    left_state: int,
    right_state: int,
    angle: "float | cudaq.QuakeValue",
) -> None:
    """Apply a multi-controlled phase conditioned on two encoded qudit states."""
    left_base = left_qudit * qubits_per_qudit
    right_base = right_qudit * qubits_per_qudit
    target_bit = qubits_per_qudit - 1
    controls = []

    for bit in range(qubits_per_qudit):
        qubit = q_full[left_base + bit]
        if ((left_state >> bit) & 1) == 0:
            kernel.x(qubit)
        controls.append(qubit)

    for bit in range(target_bit):
        qubit = q_full[right_base + bit]
        if ((right_state >> bit) & 1) == 0:
            kernel.x(qubit)
        controls.append(qubit)

    target = q_full[right_base + target_bit]
    if ((right_state >> target_bit) & 1) == 0:
        kernel.x(target)

    kernel.cr1(angle, controls, target)

    if ((right_state >> target_bit) & 1) == 0:
        kernel.x(target)
    for bit in range(target_bit - 1, -1, -1):
        if ((right_state >> bit) & 1) == 0:
            kernel.x(q_full[right_base + bit])
    for bit in range(qubits_per_qudit - 1, -1, -1):
        if ((left_state >> bit) & 1) == 0:
            kernel.x(q_full[left_base + bit])


def create_qaoa_ansatz(
    depth: int,
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
) -> "cudaq.Kernel":
    """Create a generic CUDA-Q builder kernel for Tensor-QUDO QAOA."""
    n_qudits, dimension_qudits = _validate_tqudo_shapes(Etab, Ettprimeab)
    qubits_per_qudit = max(1, int(math.ceil(math.log2(dimension_qudits))))
    n_qubits_total = n_qudits * qubits_per_qudit
    etab_terms = _nonzero_etab_terms(Etab)
    ett_terms = _nonzero_ett_terms(Ettprimeab)

    kernel, gamma, beta = cudaq.make_kernel(list[float], list[float])
    q_full = kernel.qalloc(n_qubits_total)

    for qubit_idx in range(n_qubits_total):
        kernel.h(q_full[qubit_idx])

    for layer in range(depth):
        for left_qudit, right_qudit, left_state, right_state, coeff in etab_terms:
            _apply_state_conditioned_phase(
                kernel,
                q_full,
                qubits_per_qudit,
                left_qudit,
                right_qudit,
                left_state,
                right_state,
                -gamma[layer] * coeff,
            )

        for left_qudit, right_qudit, left_state, right_state, coeff in ett_terms:
            _apply_state_conditioned_phase(
                kernel,
                q_full,
                qubits_per_qudit,
                left_qudit,
                right_qudit,
                left_state,
                right_state,
                -gamma[layer] * coeff,
            )

        for qubit_idx in range(n_qubits_total):
            kernel.rx(2.0 * beta[layer], q_full[qubit_idx])

    return kernel


def evaluate_cost(
    params: np.ndarray,
    kernel: "cudaq.Kernel",
    problem: ProblemTQUDO,
    depth: int,
    n_shots: int = 1000,
    noise_model: "cudaq.NoiseModel | None" = None,
) -> float:
    """Evaluate the QAOA cost by sampling and averaging TQUDO cost.

    Args:
        params: [gamma_1...gamma_p, beta_1...beta_p].
        kernel: QAOA kernel from create_qaoa_ansatz.
        problem: Pre-built ProblemTQUDO (reused across calls to avoid re-allocation).
        depth: QAOA depth.
        n_shots: Shots for cost estimation.
        noise_model: Pre-built cudaq.NoiseModel, or None for noiseless.

    Returns:
        Average TQUDO cost over samples.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    sample_kwargs: dict = {"shots_count": n_shots}
    if noise_model is not None:
        sample_kwargs["noise_model"] = noise_model
    samples = cudaq.sample(kernel, gamma, beta, **sample_kwargs)
    n_qudits = problem.Etab.shape[0]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(problem.Etab.shape[1]))))
    total = 0.0
    count = 0
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
    noise_model: "cudaq.NoiseModel | None" = None,
) -> "cudaq.SampleResult":
    """Sample bitstrings from the QAOA state at the given parameters.

    Args:
        kernel: QAOA kernel from create_qaoa_ansatz.
        params: [gamma_1...gamma_p, beta_1...beta_p].
        depth: Number of QAOA layers.
        n_shots: Number of measurement shots.
        noise_model: Pre-built cudaq.NoiseModel, or None for noiseless.

    Returns:
        cudaq.SampleResult: Dict-like object with bitstring counts.
    """
    gamma = params[:depth].tolist()
    beta = params[depth:].tolist()
    sample_kwargs: dict = {"shots_count": n_shots}
    if noise_model is not None:
        sample_kwargs["noise_model"] = noise_model
    return cudaq.sample(kernel, gamma, beta, **sample_kwargs)


def optimize_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
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

    kernel = create_qaoa_ansatz(depth, Etab, Ettprimeab)
    problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)
    noise_model = get_noise_model(noise_config)

    # TQA (Trotterized Quantum Annealing) initialization:
    # gamma_i = (i / p) * delta_t,  beta_i = (1 - i / p) * delta_t
    indices = np.arange(1, depth + 1)
    gamma_init = (indices / depth) * delta_t
    beta_init = (1 - indices / depth) * delta_t
    init_params = np.concatenate([gamma_init, beta_init])

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        val = evaluate_cost(
            x, kernel, problem, depth, n_shots=n_shots,
            noise_model=noise_model,
        )
        energy_history.append(val)
        reporter.opt_step(len(energy_history), max_iter, val)
        return val

    initial_energy = evaluate_cost(
        init_params, kernel, problem, depth, n_shots=n_shots,
        noise_model=noise_model,
    )

    initial_samples: "cudaq.SampleResult | None" = None
    if sample_shots is not None:
        initial_samples = sample_solution(
            kernel, init_params, depth, n_shots=sample_shots,
            noise_model=noise_model,
        )

    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=minimize_options(optimizer, max_iter),
    )
    if not opt_result.success:
        logger.warning(
            "TQUDO QAOA optimizer (%s) did not converge: %s",
            optimizer, opt_result.message,
        )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    final_samples: "cudaq.SampleResult | None" = None
    if sample_shots is not None:
        final_samples = sample_solution(
            kernel, best_params, depth, n_shots=sample_shots,
            noise_model=noise_model,
        )
    return best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history


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
    delta_t: float = 0.55,
    noise_config: NoiseConfig | None = None,
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
        Dict with keys: energy, params, initial_samples, final_samples, best_bitstring,
        best_sequence, initial_energy, energy_history.
        initial_samples: SampleResult at TQA init params (before optimization).
        final_samples: SampleResult at best params (after optimization).
        best_bitstring: Most frequent bitstring from final_samples.
        best_sequence: Qudit sequence (route) from best_bitstring.
    """
    n_qudits = Etab.shape[0]
    qubits_per_qudit = max(1, int(math.ceil(math.log2(Etab.shape[1]))))
    n_qubits_total = n_qudits * qubits_per_qudit

    best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history = (
        optimize_qaoa(
            Etab,
            Ettprimeab,
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

    best_bitstring = (
        final_samples.most_probable() if final_samples else "0" * n_qubits_total
    )
    best_sequence = bitstring_to_qudit_sequence(
        best_bitstring, n_qudits, qubits_per_qudit
    )

    return {
        "energy": best_energy,
        "params": best_params,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "best_bitstring": best_bitstring,
        "best_sequence": best_sequence,
        "initial_energy": initial_energy,
        "energy_history": energy_history,
    }
