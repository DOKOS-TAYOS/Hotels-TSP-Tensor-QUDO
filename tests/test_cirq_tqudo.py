"""Tests for the native-qudit Cirq Tensor-QUDO QAOA circuit.

Verifies that the custom qudit gates (QuditHadamardGate, QuditDiagonalCostGate,
QuditRingMixerGate) are well-formed, that create_qaoa_circuit produces runnable
circuits, and that the end-to-end run_qaoa returns sensible results.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

# Import the qaoa_circuit_tqudo module directly from its file path to avoid the
# circular-import issue in solvers.__init__ → instance_gen_process → solvers.base.
_src = Path(__file__).resolve().parent.parent / "src"
_file = _src / "solvers" / "cirq_solver" / "qaoa_circuit_tqudo.py"
_spec = importlib.util.spec_from_file_location(
    "solvers.cirq_solver.qaoa_circuit_tqudo",
    _file,
    submodule_search_locations=[],
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

QuditHadamardGate = _mod.QuditHadamardGate
QuditDiagonalCostGate = _mod.QuditDiagonalCostGate
QuditRingMixerGate = _mod.QuditRingMixerGate
bitstring_to_qudit_sequence = _mod.bitstring_to_qudit_sequence
create_qaoa_circuit = _mod.create_qaoa_circuit
evaluate_cost = _mod.evaluate_cost
key_to_qudit_sequence = _mod.key_to_qudit_sequence
measurement_to_qudit_sequence = _mod.measurement_to_qudit_sequence
qudit_sequence_to_key = _mod.qudit_sequence_to_key
run_qaoa = _mod.run_qaoa
sample_solution = _mod.sample_solution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_tqudo_tensors(
    n_qudits: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build small sparse tensors that exercise both local and long-range phases."""
    Etab = np.zeros((n_qudits, dimension, dimension), dtype=float)
    Ettprimeab = np.zeros(
        (n_qudits, n_qudits, dimension, dimension),
        dtype=float,
    )
    Etab[0, 0, dimension - 1] = 1.0
    if n_qudits > 2:
        Etab[1, min(1, dimension - 1), max(0, dimension - 2)] = 0.5
        Ettprimeab[0, n_qudits - 1, dimension - 1, 0] = 0.75
    return Etab, Ettprimeab


# ---------------------------------------------------------------------------
# QuditHadamardGate
# ---------------------------------------------------------------------------

class TestQuditHadamardGate:
    """Tests for the d-dim Hadamard (DFT) gate."""

    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    def test_unitary_is_unitary(self, d: int) -> None:
        gate = QuditHadamardGate(d)
        u = cirq.unitary(gate)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(d), atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_creates_uniform_superposition(self, d: int) -> None:
        """Applying H_d to |0⟩ should give (1/√d) Σ|k⟩."""
        gate = QuditHadamardGate(d)
        u = cirq.unitary(gate)
        state = u[:, 0]  # first column = action on |0⟩
        expected = np.ones(d) / np.sqrt(d)
        np.testing.assert_allclose(np.abs(state), expected, atol=1e-12)

    def test_qid_shape(self) -> None:
        assert cirq.qid_shape(QuditHadamardGate(5)) == (5,)

    def test_d2_matches_qubit_hadamard(self) -> None:
        """For d=2 the qudit Hadamard should match the standard H up to global phase."""
        u_qudit = cirq.unitary(QuditHadamardGate(2))
        u_qubit = cirq.unitary(cirq.H)
        # They differ at most by a global phase
        ratio = u_qudit / u_qubit
        np.testing.assert_allclose(np.abs(ratio), np.ones((2, 2)), atol=1e-12)


# ---------------------------------------------------------------------------
# QuditDiagonalCostGate
# ---------------------------------------------------------------------------

class TestQuditDiagonalCostGate:
    """Tests for the two-qudit diagonal cost gate."""

    def test_unitary_is_diagonal(self) -> None:
        d = 3
        cost = np.array([[1.0, 0.5, 0.0], [0.0, 2.0, 0.3], [0.1, 0.0, 1.5]])
        gate = QuditDiagonalCostGate(d, gamma=0.5, cost_matrix=cost)
        u = cirq.unitary(gate)
        # Off-diagonal should be zero
        np.testing.assert_allclose(u - np.diag(np.diag(u)), 0.0, atol=1e-12)

    def test_unitary_is_unitary(self) -> None:
        d = 3
        cost = np.random.default_rng(42).random((d, d))
        gate = QuditDiagonalCostGate(d, gamma=1.2, cost_matrix=cost)
        u = cirq.unitary(gate)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(d * d), atol=1e-12)

    def test_diagonal_values_match_formula(self) -> None:
        d = 2
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])
        gamma = 0.7
        gate = QuditDiagonalCostGate(d, gamma=gamma, cost_matrix=cost)
        u = cirq.unitary(gate)
        expected_diag = np.exp(-1j * gamma * cost.flatten())
        np.testing.assert_allclose(np.diag(u), expected_diag, atol=1e-12)

    def test_qid_shape(self) -> None:
        d = 4
        cost = np.zeros((d, d))
        gate = QuditDiagonalCostGate(d, gamma=0.1, cost_matrix=cost)
        assert cirq.qid_shape(gate) == (d, d)

    def test_parameterized_returns_none_unitary(self) -> None:
        import sympy
        d = 3
        cost = np.zeros((d, d))
        gate = QuditDiagonalCostGate(d, gamma=sympy.Symbol("g"), cost_matrix=cost)
        assert gate._is_parameterized_()
        assert gate._unitary_() is None

    def test_resolve_parameters(self) -> None:
        import sympy
        d = 2
        cost = np.array([[1.0, 0.0], [0.0, 1.0]])
        g = sympy.Symbol("g")
        gate = QuditDiagonalCostGate(d, gamma=g, cost_matrix=cost)
        resolved = cirq.resolve_parameters(gate, {"g": 0.5})
        u = cirq.unitary(resolved)
        assert u is not None
        assert u.shape == (d * d, d * d)


# ---------------------------------------------------------------------------
# QuditRingMixerGate
# ---------------------------------------------------------------------------

class TestQuditRingMixerGate:
    """Tests for the ring mixer exp(-i·angle·X_d)."""

    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    def test_unitary_is_unitary(self, d: int) -> None:
        gate = QuditRingMixerGate(d, angle=0.8)
        u = cirq.unitary(gate)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(d), atol=1e-12)

    def test_zero_angle_is_identity(self) -> None:
        d = 4
        gate = QuditRingMixerGate(d, angle=0.0)
        u = cirq.unitary(gate)
        np.testing.assert_allclose(u, np.eye(d), atol=1e-12)

    def test_d2_matches_rx(self) -> None:
        """For d=2, exp(-i·angle·M_2) = exp(-i·angle·X) = Rx(2·angle)."""
        angle = 1.3
        u_qudit = cirq.unitary(QuditRingMixerGate(2, angle=angle))
        u_rx = cirq.unitary(cirq.rx(2.0 * angle))
        # Should be equal up to global phase
        ratio = u_qudit.flatten() / u_rx.flatten()
        phases = np.angle(ratio)
        np.testing.assert_allclose(phases - phases[0], np.zeros(4), atol=1e-12)

    def test_qid_shape(self) -> None:
        assert cirq.qid_shape(QuditRingMixerGate(5, angle=0.1)) == (5,)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

class TestCreateQaoaCircuit:
    """Tests for the QAOA circuit builder with native qudits."""

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_circuit_builds_without_error(self, d: int) -> None:
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension=d)
        circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
            depth=1, Etab=Etab, Ettprimeab=Ettprimeab
        )
        assert isinstance(circuit, cirq.Circuit)
        assert len(qudits) == 3
        assert n_qudits == 3
        assert dimension == d

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_qudits_have_correct_dimension(self, d: int) -> None:
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension=d)
        _, _, qudits, _, _ = create_qaoa_circuit(depth=1, Etab=Etab, Ettprimeab=Ettprimeab)
        for q in qudits:
            assert q.dimension == d

    def test_symbols_created_for_each_layer(self) -> None:
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=2, dimension=2)
        _, symbols, _, _, _ = create_qaoa_circuit(depth=3, Etab=Etab, Ettprimeab=Ettprimeab)
        for k in range(3):
            assert f"gamma_{k}" in symbols
            assert f"beta_{k}" in symbols

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_circuit_simulates_without_error(self, d: int) -> None:
        """The circuit should be simulable with concrete parameters."""
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension=d)
        circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
            depth=1, Etab=Etab, Ettprimeab=Ettprimeab
        )
        resolver = cirq.ParamResolver({
            symbols["gamma_0"]: 0.3,
            symbols["beta_0"]: 0.2,
        })
        circuit_m = circuit + cirq.measure(*qudits, key="m")
        sim = cirq.Simulator(seed=42)
        result = sim.run(circuit_m, resolver, repetitions=10)
        assert result.measurements["m"].shape == (10, n_qudits)
        # All values should be in [0, d)
        assert np.all(result.measurements["m"] >= 0)
        assert np.all(result.measurements["m"] < d)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

class TestMeasurementHelpers:
    """Tests for qudit-key encoding/decoding."""

    def test_qudit_sequence_to_key_and_back(self) -> None:
        seq = np.array([0, 3, 1, 2], dtype=np.int64)
        key = qudit_sequence_to_key(seq)
        assert key == "0-3-1-2"
        recovered = key_to_qudit_sequence(key)
        np.testing.assert_array_equal(recovered, seq)

    def test_measurement_to_qudit_sequence(self) -> None:
        row = np.array([2, 0, 3], dtype=np.int64)
        seq = measurement_to_qudit_sequence(row, n_qudits=3)
        np.testing.assert_array_equal(seq, row)

    def test_bitstring_to_qudit_sequence_dash_format(self) -> None:
        """New dash-separated format should be decoded correctly."""
        result = bitstring_to_qudit_sequence("1-3-0", n_qudits=3, qubits_per_qudit=2)
        np.testing.assert_array_equal(result, [1, 3, 0])

    def test_bitstring_to_qudit_sequence_legacy_format(self) -> None:
        """Legacy binary format should still work for backwards compat."""
        result = bitstring_to_qudit_sequence("0110", n_qudits=2, qubits_per_qudit=2)
        expected = np.array([2, 1], dtype=np.int64)  # "01" → 2 (little-endian), "10" → 1
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Evaluate cost
# ---------------------------------------------------------------------------

class TestEvaluateCost:
    """Smoke test for cost evaluation."""

    @pytest.mark.parametrize("d", [2, 4])
    def test_evaluate_cost_returns_float(self, d: int) -> None:
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension=d)
        circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
            depth=1, Etab=Etab, Ettprimeab=Ettprimeab
        )
        params = np.array([0.3, 0.2])
        val = evaluate_cost(
            params, circuit, Etab, Ettprimeab, symbols, 1,
            qudits, n_qudits, dimension,
            n_shots=20, seed=42,
        )
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# End-to-end run_qaoa
# ---------------------------------------------------------------------------

class TestRunQaoa:
    """Integration tests for the full QAOA pipeline."""

    @pytest.mark.parametrize("d", [2, 3, 4])
    def test_run_qaoa_returns_expected_keys(self, d: int) -> None:
        Etab, Ettprimeab = _synthetic_tqudo_tensors(n_qudits=3, dimension=d)
        result = run_qaoa(
            Etab, Ettprimeab,
            depth=1,
            max_iter=4,
            n_shots=16,
            sample_shots=16,
            seed=42,
        )
        assert isinstance(result["energy"], float)
        assert isinstance(result["best_bitstring"], str)
        assert result["best_sequence"].shape == (3,)
        assert isinstance(result["initial_energy"], float)
        assert isinstance(result["energy_history"], list)
        # Best sequence values must be in [0, d)
        assert np.all(result["best_sequence"] >= 0)
        assert np.all(result["best_sequence"] < d)

    def test_run_qaoa_with_real_instance(self) -> None:
        """Use generate_TQUDO_from_problem for a realistic smoke test."""
        import random
        from instance_gen_process import InstanceConfig, generate_random_instance, generate_TQUDO_from_problem
        from instance_gen_process.models import RestrictionConfig

        config = InstanceConfig(
            n_cities=5,
            n_precedences_range=(0, 0),
            prices_range_hotels=(1.0, 2.0),
            prices_range_travels=(1.0, 2.0),
            seed=17,
        )
        rng = random.Random(config.seed)
        instance = generate_random_instance(config, rng)
        problem = generate_TQUDO_from_problem(
            instance,
            RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0),
        )
        raw = run_qaoa(
            problem.Etab,
            problem.Ettprimeab,
            depth=1,
            max_iter=4,
            n_shots=16,
            sample_shots=16,
            seed=17,
        )
        assert isinstance(raw["energy"], float)
        assert raw["best_sequence"].shape == (instance.n_cities - 1,)
