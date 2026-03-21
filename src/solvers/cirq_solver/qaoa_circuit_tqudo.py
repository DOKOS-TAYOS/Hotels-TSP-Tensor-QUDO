"""QAOA circuit implementation for Tensor QUDO problems using **native Cirq qudits**.

Each logical qudit of dimension *d* is represented by a single ``cirq.LineQid``
with ``dimension=d`` — *not* as ⌈log₂ d⌉ qubits.  This removes all binary
encoding/decoding overhead, eliminates spurious basis states when *d* is not a
power of two, and makes the circuit structure mirror the Tensor-QUDO
Hamiltonian directly.

Key differences from the previous qubit-emulation approach
(preserved in ``qaoa_circuit_tqudo_qubit_emulation.py``):

1. **Initial state**: A custom ``QuditHadamardGate`` (d-dim DFT) creates the
   uniform superposition |+_d⟩ = (1/√d) Σ |k⟩  on each qudit.

2. **Cost layer**: A single diagonal two-qudit gate ``QuditDiagonalCostGate``
   of size d²×d² replaces the d² iterations of X-flip → multi-controlled-Z →
   X-unflip.  The unitary is diag(exp(−iγ E[x₀,x₁])).

3. **Mixer layer**: A ``QuditRingMixerGate`` implements the ring mixer
   exp(−i·2β·X_d) where X_d is the cyclic-shift (generalized Pauli-X) on d
   levels.  This is the direct qudit analog of rx(2β) on a qubit.

4. **Measurement**: Cirq returns integers 0…d−1 per qudit; no bitstring→qudit
   decoding is needed.

Note on mixer equivalence
-------------------------
The ring mixer exp(−iβX_d) is *not* identical to the old per-qubit rx(2β).
The qubit-emulation mixer mixed individual bits independently, whereas the
ring mixer rotates in the d-dimensional cyclic-shift basis.  For d = 2 the two
coincide exactly.  For d > 2 the optimization landscape changes; this is
inherent to the native-qudit formulation and should be evaluated experimentally.
"""

from __future__ import annotations

import numpy as np
import sympy
from scipy.optimize import minimize
from scipy.linalg import expm

import cirq

from instance_gen_process.models import ProblemTQUDO
from solvers.cirq_solver.noise_model import get_simulator
from solvers.noise import NoiseConfig
from utils.costs import calculate_tqudo_cost
from utils.optimizer import minimize_options
from solvers.base import OptimizerType
from utils.progress import reporter
from utils.qaoa_helpers import most_probable_key, tqa_init_params


# ---------------------------------------------------------------------------
# Custom qudit gates
# ---------------------------------------------------------------------------


class QuditHadamardGate(cirq.Gate):
    """d-dimensional Hadamard: maps |0⟩ → (1/√d) Σ_k |k⟩.

    Unitary is the d×d Discrete Fourier Transform matrix
    F_d[j,k] = e^{−2πi jk/d} / √d  (np.fft.fft convention).
    For d = 2 this is the standard Hadamard H (up to global phase).
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._dimension,)

    def _unitary_(self) -> np.ndarray:
        d = self._dimension
        return np.fft.fft(np.eye(d), norm="ortho")

    def _circuit_diagram_info_(self, _args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"H_d({self._dimension})"

    def __repr__(self) -> str:
        return f"QuditHadamardGate(dimension={self._dimension})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuditHadamardGate) and self._dimension == other._dimension

    def __hash__(self) -> int:
        return hash((QuditHadamardGate, self._dimension))


class QuditDiagonalCostGate(cirq.Gate):
    """Diagonal two-qudit cost gate for a pair of d-dimensional qudits.

    Unitary: diag(exp(−i·γ·cost_matrix.flatten()))  of size d²×d².

    The basis ordering follows Cirq's convention for multi-qudit systems:
    |x₀⟩⊗|x₁⟩  →  index = x₀·d + x₁.

    Parameters
    ----------
    dimension : int
        Qudit dimension d.
    gamma : float | sympy.Expr
        QAOA variational parameter γ.
    cost_matrix : np.ndarray
        d×d real matrix of cost coefficients (e.g. Etab[t] or Ettprimeab[t,t']).
    """

    def __init__(
        self,
        dimension: int,
        gamma: float | sympy.Expr,
        cost_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self._dimension = dimension
        self._gamma = gamma
        self._cost_matrix = np.asarray(cost_matrix, dtype=float)

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._dimension, self._dimension)

    # -- Sympy parameterization support --
    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self._gamma)

    def _parameter_names_(self) -> frozenset[str]:
        return cirq.parameter_names(self._gamma)

    def _resolve_parameters_(
        self,
        resolver: cirq.ParamResolver,
        recursive: bool,
    ) -> QuditDiagonalCostGate:
        new_gamma = cirq.resolve_parameters(self._gamma, resolver, recursive)
        return QuditDiagonalCostGate(self._dimension, new_gamma, self._cost_matrix)

    def _unitary_(self) -> np.ndarray | None:
        if self._is_parameterized_():
            return None  # Cirq will resolve before calling _unitary_
        gamma_val = float(self._gamma)
        phases = -gamma_val * self._cost_matrix.flatten()
        return np.diag(np.exp(1j * phases))

    def _circuit_diagram_info_(self, _args: cirq.CircuitDiagramInfoArgs) -> list[str]:
        return [f"Cost_d({self._dimension})", f"Cost_d({self._dimension})"]

    def __repr__(self) -> str:
        return (
            f"QuditDiagonalCostGate(dimension={self._dimension}, "
            f"gamma={self._gamma!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuditDiagonalCostGate):
            return NotImplemented
        return (
            self._dimension == other._dimension
            and self._gamma == other._gamma
            and np.array_equal(self._cost_matrix, other._cost_matrix)
        )

    def __hash__(self) -> int:
        return hash((QuditDiagonalCostGate, self._dimension, str(self._gamma)))


class QuditRingMixerGate(cirq.Gate):
    """Ring mixer on a single d-dimensional qudit: exp(−i·angle·M_d).

    M_d is the Hermitian ring-exchange operator:
        M_d = (X_d + X_d†) / 2

    where X_d = Σ_k |k+1 mod d⟩⟨k| is the cyclic-shift operator.

    For d = 2, M_d = X (Pauli-X) and this gate reduces to
    exp(−i·angle·X) = Rx(2·angle).

    Note: X_d itself is **not** Hermitian for d > 2 (its adjoint is the
    backward shift).  Using the Hermitised form (X_d + X_d†)/2 guarantees
    the resulting matrix exponential is always unitary.

    Parameters
    ----------
    dimension : int
        Qudit dimension d.
    angle : float | sympy.Expr
        Rotation angle β (the QAOA variational parameter).
        The unitary is exp(−i·β·M_d).
    """

    def __init__(self, dimension: int, angle: float | sympy.Expr) -> None:
        super().__init__()
        self._dimension = dimension
        self._angle = angle

    def _qid_shape_(self) -> tuple[int, ...]:
        return (self._dimension,)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self._angle)

    def _parameter_names_(self) -> frozenset[str]:
        return cirq.parameter_names(self._angle)

    def _resolve_parameters_(
        self,
        resolver: cirq.ParamResolver,
        recursive: bool,
    ) -> QuditRingMixerGate:
        new_angle = cirq.resolve_parameters(self._angle, resolver, recursive)
        return QuditRingMixerGate(self._dimension, new_angle)

    def _unitary_(self) -> np.ndarray | None:
        if self._is_parameterized_():
            return None
        d = self._dimension
        angle_val = float(self._angle)
        # Build X_d (cyclic shift) and its adjoint (backward shift)
        x_d = np.zeros((d, d), dtype=complex)
        for k in range(d):
            x_d[(k + 1) % d, k] = 1.0
        # Hermitian generator: M_d = (X_d + X_d†) / 2
        m_d = (x_d + x_d.conj().T) / 2.0
        return expm(-1j * angle_val * m_d)

    def _circuit_diagram_info_(self, _args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"Rx_d({self._dimension})"

    def __repr__(self) -> str:
        return f"QuditRingMixerGate(dimension={self._dimension}, angle={self._angle!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuditRingMixerGate):
            return NotImplemented
        return self._dimension == other._dimension and self._angle == other._angle

    def __hash__(self) -> int:
        return hash((QuditRingMixerGate, self._dimension, str(self._angle)))


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------


def create_qaoa_circuit(
    depth: int,
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
) -> tuple[cirq.Circuit, dict[str, sympy.Symbol], list[cirq.Qid], int, int]:
    """Create the parametrized QAOA circuit for Tensor QUDO on native qudits.

    Each qudit is a single ``cirq.LineQid(i, dimension=d)`` — **not** a group
    of qubits.

    Cost layer
    ----------
    For every adjacent pair (t, t+1), a single diagonal d²×d² unitary encodes
    all Etab[t, x₀, x₁] phases.  Likewise for all long-range pairs (t, t')
    with Ettprimeab[t, t', x_t, x_t'].

    Mixer layer
    -----------
    Ring mixer exp(−i·2β·X_d) on each qudit, where X_d is the cyclic-shift.

    Returns
    -------
    circuit : cirq.Circuit
    symbols : dict mapping "gamma_k"/"beta_k" to sympy.Symbol
    qudits  : list of ``cirq.LineQid`` (length n_qudits)
    n_qudits : int
    dimension : int (the qudit dimension d — replaces the old ``qubits_per_qudit``)
    """
    n_qudits = Etab.shape[0]
    dimension = Etab.shape[1]  # d

    qudits = list(cirq.LineQid.range(n_qudits, dimension=dimension))

    symbols: dict[str, sympy.Symbol] = {}
    for k in range(depth):
        symbols[f"gamma_{k}"] = sympy.Symbol(f"gamma_{k}")
        symbols[f"beta_{k}"] = sympy.Symbol(f"beta_{k}")

    moments: list[cirq.OP_TREE] = []

    # Initial state: |+_d⟩ on each qudit  (uniform superposition over d levels)
    h_gate = QuditHadamardGate(dimension)
    for q in qudits:
        moments.append(h_gate.on(q))

    for k in range(depth):
        gamma_k = symbols[f"gamma_{k}"]
        beta_k = symbols[f"beta_{k}"]

        # Cost layer — Etab: adjacent pairs (t, t+1)
        for t in range(n_qudits - 1):
            cost_mat = Etab[t]  # shape (d, d)
            if np.allclose(cost_mat, 0.0):
                continue
            gate = QuditDiagonalCostGate(dimension, gamma_k, cost_mat)
            moments.append(gate.on(qudits[t], qudits[t + 1]))

        # Cost layer — Ettprimeab: all pairs (t, t') with t' > t
        for t in range(n_qudits - 1):
            for t_prime in range(t + 1, n_qudits):
                cost_mat = Ettprimeab[t, t_prime]  # shape (d, d)
                if np.allclose(cost_mat, 0.0):
                    continue
                gate = QuditDiagonalCostGate(dimension, gamma_k, cost_mat)
                moments.append(gate.on(qudits[t], qudits[t_prime]))

        # Mixer layer: ring mixer on each qudit
        # exp(-i·β·M_d) where M_d = (X_d + X_d†)/2.
        # For d=2: M_d = X, so exp(-iβX) = Rx(2β), matching standard QAOA.
        for q in qudits:
            mixer = QuditRingMixerGate(dimension, beta_k)
            moments.append(mixer.on(q))

    circuit = cirq.Circuit(moments)
    return circuit, symbols, qudits, n_qudits, dimension


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def _param_resolver(
    params: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
) -> cirq.ParamResolver:
    """Build ParamResolver from params array [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]."""
    resolver_dict: dict[sympy.Symbol, float] = {}
    for k in range(depth):
        resolver_dict[symbols[f"gamma_{k}"]] = float(params[k])
        resolver_dict[symbols[f"beta_{k}"]] = float(params[depth + k])
    return cirq.ParamResolver(resolver_dict)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def measurement_to_qudit_sequence(
    row: np.ndarray,
    n_qudits: int,
) -> np.ndarray:
    """Convert a native qudit measurement row to a qudit sequence (route).

    With native qudits, each element of *row* is already an integer in
    {0, …, d−1}, so this is a trivial reshape.
    """
    return np.asarray(row[:n_qudits], dtype=np.int64)


def qudit_sequence_to_key(seq: np.ndarray) -> str:
    """Encode a qudit-value sequence as a dash-separated string for counting.

    Example: [0, 3, 1] → "0-3-1".
    """
    return "-".join(str(int(v)) for v in seq)


def key_to_qudit_sequence(key: str) -> np.ndarray:
    """Inverse of ``qudit_sequence_to_key``."""
    return np.array([int(v) for v in key.split("-")], dtype=np.int64)


# Backwards-compatible alias kept for solver.py (signature unchanged).
def bitstring_to_qudit_sequence(
    bitstring: str,
    n_qudits: int,
    qubits_per_qudit: int,
) -> np.ndarray:
    """Decode a measurement key back to qudit values.

    Accepts **both** the new dash-separated format ("0-3-1") and the legacy
    binary-encoded bitstring ("0110") so that callers migrating gradually
    continue to work.
    """
    if "-" in bitstring:
        return key_to_qudit_sequence(bitstring)
    # Legacy binary format (for any remaining callers)
    seq = np.zeros(n_qudits, dtype=np.int64)
    for i in range(n_qudits):
        start = i * qubits_per_qudit
        for j in range(qubits_per_qudit):
            if start + j < len(bitstring) and bitstring[start + j] == "1":
                seq[i] += 1 << j
    return seq


# ---------------------------------------------------------------------------
# Cost evaluation
# ---------------------------------------------------------------------------


def evaluate_cost(
    params: np.ndarray,
    circuit_with_measure: cirq.Circuit,
    problem: ProblemTQUDO,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    n_qudits: int,
    n_shots: int,
    simulator: cirq.SimulatesSamples,
) -> float:
    """Evaluate the QAOA cost by sampling and averaging TQUDO cost.

    The circuit (with measurement gates already appended and noise applied)
    and the simulator are created once and reused across all optimizer steps.
    """
    resolver = _param_resolver(params, symbols, depth)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    total = 0.0
    for row in result.measurements["m"]:
        seq = measurement_to_qudit_sequence(row, n_qudits)
        total += calculate_tqudo_cost(problem, seq)
    return total / n_shots


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_solution(
    circuit_with_measure: cirq.Circuit,
    params: np.ndarray,
    symbols: dict[str, sympy.Symbol],
    depth: int,
    n_qudits: int,
    n_shots: int,
    simulator: cirq.SimulatesSamples,
) -> dict[str, int]:
    """Sample qudit sequences from the QAOA state.

    Returns dict  { "0-3-1": count, ... }  using dash-separated qudit-value keys.
    """
    resolver = _param_resolver(params, symbols, depth)
    result = simulator.run(circuit_with_measure, resolver, repetitions=n_shots)

    counts: dict[str, int] = {}
    for row in result.measurements["m"]:
        seq = measurement_to_qudit_sequence(row, n_qudits)
        key = qudit_sequence_to_key(seq)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


def optimize_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int | None = None,
    seed: int | None = None,
    optimizer: OptimizerType = "COBYLA",
    delta_t: float = 0.55,
    noise_config: NoiseConfig | None = None,
) -> tuple[float, np.ndarray, dict[str, int] | None, dict[str, int] | None, float, list[float]]:
    """Optimize QAOA parameters to minimize the TQUDO cost.

    Returns:
        Tuple of (best_energy, best_params, initial_samples, final_samples,
        initial_energy, energy_history).
        initial_samples: Qudit-sequence counts at TQA init params when sample_shots is set.
        final_samples: Qudit-sequence counts at best_params when sample_shots is set.
    """
    circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
        depth, Etab, Ettprimeab
    )

    problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)
    simulator, noise_model = get_simulator(
        noise_config, qudit_dimension=dimension, seed=seed,
    )
    circuit_with_measure = circuit + cirq.measure(*qudits, key="m")
    if noise_model is not None:
        circuit_with_measure = circuit_with_measure.with_noise(noise_model)

    init_params = tqa_init_params(depth, delta_t)

    energy_history: list[float] = []

    def cost_fn(x: np.ndarray) -> float:
        val = evaluate_cost(
            x, circuit_with_measure, problem, symbols, depth,
            n_qudits, n_shots, simulator,
        )
        energy_history.append(val)
        reporter.opt_step(len(energy_history), max_iter, val)
        return val

    initial_energy = evaluate_cost(
        init_params, circuit_with_measure, problem, symbols, depth,
        n_qudits, n_shots, simulator,
    )

    initial_samples: dict[str, int] | None = None
    if sample_shots is not None:
        initial_samples = sample_solution(
            circuit_with_measure, init_params, symbols, depth,
            n_qudits, sample_shots, simulator,
        )

    opt_result = minimize(
        cost_fn,
        init_params,
        method=optimizer,
        options=minimize_options(optimizer, max_iter),
    )
    best_params = opt_result.x
    best_energy = float(opt_result.fun)
    final_samples: dict[str, int] | None = None
    if sample_shots is not None:
        final_samples = sample_solution(
            circuit_with_measure, best_params, symbols, depth,
            n_qudits, sample_shots, simulator,
        )
    return best_energy, best_params, initial_samples, final_samples, initial_energy, energy_history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_qaoa(
    Etab: np.ndarray,
    Ettprimeab: np.ndarray,
    depth: int = 1,
    max_iter: int = 100,
    n_shots: int = 500,
    sample_shots: int = 1000,
    seed: int | None = None,
    optimizer: OptimizerType = "COBYLA",
    delta_t: float = 0.55,
    noise_config: NoiseConfig | None = None,
) -> dict:
    """Run full QAOA: optimize, sample, and return best solution."""
    n_qudits = Etab.shape[0]

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

    fallback_key = "-".join(["0"] * n_qudits)
    best_key = most_probable_key(final_samples, fallback_key) if final_samples else fallback_key
    best_sequence = key_to_qudit_sequence(best_key)

    return {
        "energy": best_energy,
        "params": best_params,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "best_bitstring": best_key,
        "best_sequence": best_sequence,
        "initial_energy": initial_energy,
        "energy_history": energy_history,
    }
