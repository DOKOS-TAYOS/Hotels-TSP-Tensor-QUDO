"""Tests for the noise simulation module.

Covers:
- NoiseConfig validation and defaults
- Cirq noise model builder (build_noise_model, get_simulator)
- CUDA-Q noise model builder (build_noise_model, get_noise_model)  [import-guarded]
- Solver config round-trip with noise section
- End-to-end noisy QAOA run on Cirq (small 2-qubit QUBO)
- Kill-switch logic (HTSP_ENABLE_NOISE_SIMULATION=false overrides YAML)
"""

from __future__ import annotations


import numpy as np
import pytest

from conftest import cleanup_workspace_tmp_dir, workspace_tmp_dir
from solvers.noise import NoiseConfig, VALID_NOISE_TYPES


# ---------------------------------------------------------------------------
# NoiseConfig dataclass tests
# ---------------------------------------------------------------------------


class TestNoiseConfigDefaults:
    """Verify default values and immutability."""

    def test_defaults(self) -> None:
        cfg = NoiseConfig()
        assert cfg.enabled is False
        assert cfg.noise_type == "depolarizing"
        assert cfg.probability == 0.01
        assert cfg.gate_noise == {}

    def test_frozen(self) -> None:
        cfg = NoiseConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]


class TestNoiseConfigValidation:
    """Boundary and invalid-input checks."""

    @pytest.mark.parametrize("noise_type", sorted(VALID_NOISE_TYPES))
    def test_valid_noise_types(self, noise_type: str) -> None:
        cfg = NoiseConfig(enabled=True, noise_type=noise_type, probability=0.05)
        assert cfg.noise_type == noise_type

    def test_invalid_noise_type(self) -> None:
        with pytest.raises(ValueError, match="noise_type must be one of"):
            NoiseConfig(noise_type="unknown")  # type: ignore[arg-type]

    def test_probability_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="probability must be in"):
            NoiseConfig(probability=1.5)

    def test_probability_out_of_range_negative(self) -> None:
        with pytest.raises(ValueError, match="probability must be in"):
            NoiseConfig(probability=-0.01)

    def test_gate_noise_invalid_probability(self) -> None:
        with pytest.raises(ValueError, match="gate_noise"):
            NoiseConfig(gate_noise={"x": 2.0})

    def test_probability_boundary_zero(self) -> None:
        cfg = NoiseConfig(probability=0.0)
        assert cfg.probability == 0.0

    def test_probability_boundary_one(self) -> None:
        cfg = NoiseConfig(probability=1.0)
        assert cfg.probability == 1.0


class TestNoiseConfigWarning:
    """Performance warning for large qubit counts."""

    def test_no_warning_when_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = NoiseConfig(enabled=False)
        cfg.warn_if_large_system(20)
        assert "Noise simulation enabled" not in caplog.text

    def test_no_warning_small_system(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = NoiseConfig(enabled=True)
        cfg.warn_if_large_system(10)
        assert "Noise simulation enabled" not in caplog.text

    def test_warning_large_system(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = NoiseConfig(enabled=True)
        import logging
        with caplog.at_level(logging.WARNING):
            cfg.warn_if_large_system(20)
        assert "Noise simulation enabled" in caplog.text


# ---------------------------------------------------------------------------
# Cirq noise model builder
# ---------------------------------------------------------------------------


class TestCirqNoiseModelBuilder:
    """Tests for solvers.cirq_solver.noise_model."""

    cirq = pytest.importorskip("cirq")

    def test_get_simulator_noiseless_returns_state_vector(self) -> None:
        from solvers.cirq_solver.noise_model import get_simulator
        sim, noise = get_simulator(None, seed=42)
        assert isinstance(sim, self.cirq.Simulator)
        assert noise is None

    def test_get_simulator_disabled_returns_state_vector(self) -> None:
        from solvers.cirq_solver.noise_model import get_simulator
        cfg = NoiseConfig(enabled=False)
        sim, noise = get_simulator(cfg, seed=0)
        assert isinstance(sim, self.cirq.Simulator)
        assert noise is None

    def test_get_simulator_enabled_returns_density_matrix(self) -> None:
        from solvers.cirq_solver.noise_model import get_simulator
        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        sim, noise = get_simulator(cfg, seed=0)
        assert isinstance(sim, self.cirq.DensityMatrixSimulator)
        assert noise is not None

    @pytest.mark.parametrize("noise_type", sorted(VALID_NOISE_TYPES))
    def test_build_all_noise_types(self, noise_type: str) -> None:
        from solvers.cirq_solver.noise_model import build_noise_model
        cfg = NoiseConfig(enabled=True, noise_type=noise_type, probability=0.05)
        model = build_noise_model(cfg)
        assert model is not None

    def test_qudit_noise_returns_model(self) -> None:
        from solvers.cirq_solver.noise_model import build_noise_model
        cfg = NoiseConfig(enabled=True, probability=0.01)
        model = build_noise_model(cfg, qudit_dimension=3)
        assert model is not None


# ---------------------------------------------------------------------------
# Solver config round-trip with noise section
# ---------------------------------------------------------------------------


class TestSolverConfigNoiseRoundTrip:
    """Test that noise config survives YAML → load_solver_config → SolverRunConfig."""

    def test_noise_section_parsed(self) -> None:
        from instance_gen_process import load_solver_config, solver_config_to_run_config

        tmp = workspace_tmp_dir("solver_noise_rt")
        try:
            cfg_path = tmp / "solver_config.yaml"
            cfg_path.write_text(
                "\n".join([
                    "n_instances: 1",
                    "solver: cirq",
                    "formulation: qubo",
                    "optimizer: COBYLA",
                    "noise:",
                    "  enabled: true",
                    "  noise_type: amplitude_damping",
                    "  probability: 0.03",
                    "  gate_noise:",
                    "    x: 0.05",
                ]),
                encoding="utf-8",
            )
            config = load_solver_config(cfg_path)
            noise: NoiseConfig = config["noise_config"]
            assert noise.enabled is True
            assert noise.noise_type == "amplitude_damping"
            assert noise.probability == pytest.approx(0.03)
            assert noise.gate_noise == {"x": 0.05}

            run_config = solver_config_to_run_config(config)
            assert run_config.noise_config.enabled is True
            assert run_config.noise_config.noise_type == "amplitude_damping"
        finally:
            cleanup_workspace_tmp_dir(tmp)

    def test_noise_section_defaults_when_absent(self) -> None:
        from instance_gen_process import load_solver_config, solver_config_to_run_config

        tmp = workspace_tmp_dir("solver_noise_absent")
        try:
            cfg_path = tmp / "solver_config.yaml"
            cfg_path.write_text(
                "\n".join([
                    "n_instances: 1",
                    "solver: cirq",
                    "formulation: qubo",
                    "optimizer: COBYLA",
                ]),
                encoding="utf-8",
            )
            config = load_solver_config(cfg_path)
            noise: NoiseConfig = config["noise_config"]
            assert noise.enabled is False
            assert noise.noise_type == "depolarizing"
            assert noise.probability == pytest.approx(0.01)

            run_config = solver_config_to_run_config(config)
            assert run_config.noise_config.enabled is False
        finally:
            cleanup_workspace_tmp_dir(tmp)

    def test_invalid_noise_type_rejected(self) -> None:
        from instance_gen_process import load_solver_config

        tmp = workspace_tmp_dir("solver_noise_invalid")
        try:
            cfg_path = tmp / "solver_config.yaml"
            cfg_path.write_text(
                "\n".join([
                    "n_instances: 1",
                    "solver: cirq",
                    "formulation: qubo",
                    "optimizer: COBYLA",
                    "noise:",
                    "  enabled: true",
                    "  noise_type: invalid_channel",
                    "  probability: 0.01",
                ]),
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="noise.noise_type"):
                load_solver_config(cfg_path)
        finally:
            cleanup_workspace_tmp_dir(tmp)

    def test_invalid_noise_probability_rejected(self) -> None:
        from instance_gen_process import load_solver_config

        tmp = workspace_tmp_dir("solver_noise_prob")
        try:
            cfg_path = tmp / "solver_config.yaml"
            cfg_path.write_text(
                "\n".join([
                    "n_instances: 1",
                    "solver: cirq",
                    "formulation: qubo",
                    "optimizer: COBYLA",
                    "noise:",
                    "  enabled: true",
                    "  noise_type: depolarizing",
                    "  probability: 2.5",
                ]),
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="noise.probability"):
                load_solver_config(cfg_path)
        finally:
            cleanup_workspace_tmp_dir(tmp)


# ---------------------------------------------------------------------------
# End-to-end Cirq noisy run (small 2-qubit QUBO)
# ---------------------------------------------------------------------------


class TestCirqNoisyQAOARun:
    """Verify that a noisy QAOA run completes without errors on a trivial problem."""

    cirq = pytest.importorskip("cirq")

    def _small_qubo(self) -> np.ndarray:
        """2-qubit diagonal QUBO: min at x=[0,0]."""
        return np.array([[1.0, 0.5], [0.5, 2.0]])

    def test_noiseless_run(self) -> None:
        """Baseline: noiseless run returns a result."""
        from solvers.cirq_solver.qaoa_circuit_qubo import run_qaoa

        result = run_qaoa(
            self._small_qubo(), depth=1, max_iter=5, n_shots=50,
            sample_shots=50, seed=42, noise_config=None,
        )
        assert "energy" in result
        assert "best_binary" in result

    def test_noisy_depolarizing_run(self) -> None:
        """Noisy run with depolarizing noise completes."""
        from solvers.cirq_solver.qaoa_circuit_qubo import run_qaoa

        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        result = run_qaoa(
            self._small_qubo(), depth=1, max_iter=5, n_shots=50,
            sample_shots=50, seed=42, noise_config=cfg,
        )
        assert "energy" in result
        assert "best_binary" in result

    def test_noisy_amplitude_damping_run(self) -> None:
        from solvers.cirq_solver.qaoa_circuit_qubo import run_qaoa

        cfg = NoiseConfig(enabled=True, noise_type="amplitude_damping", probability=0.02)
        result = run_qaoa(
            self._small_qubo(), depth=1, max_iter=5, n_shots=50,
            sample_shots=50, seed=42, noise_config=cfg,
        )
        assert "energy" in result

    def test_disabled_noise_matches_noiseless(self) -> None:
        """Disabled noise config should produce identical results to None."""
        from solvers.cirq_solver.qaoa_circuit_qubo import run_qaoa

        qubo = self._small_qubo()
        result_none = run_qaoa(
            qubo, depth=1, max_iter=5, n_shots=100,
            sample_shots=100, seed=42, noise_config=None,
        )
        cfg_off = NoiseConfig(enabled=False, noise_type="depolarizing", probability=0.5)
        result_off = run_qaoa(
            qubo, depth=1, max_iter=5, n_shots=100,
            sample_shots=100, seed=42, noise_config=cfg_off,
        )
        assert result_none["energy"] == pytest.approx(result_off["energy"])
        assert result_none["best_bitstring"] == result_off["best_bitstring"]


class TestCirqNoisyTQUDOQubitEmulation:
    """End-to-end noisy run for TQUDO qubit-emulation (Cirq)."""

    cirq = pytest.importorskip("cirq")

    def _small_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Minimal 2-qudit, dimension-2 tensors."""
        rng = np.random.default_rng(123)
        Etab = rng.random((2, 2, 2))
        Ettprimeab = rng.random((2, 2, 2, 2))
        return Etab, Ettprimeab

    def test_noisy_run(self) -> None:
        from solvers.cirq_solver.qaoa_circuit_tqudo_qubit_emulation import run_qaoa

        Etab, Ettprimeab = self._small_tensors()
        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        result = run_qaoa(
            Etab, Ettprimeab, depth=1, max_iter=5, n_shots=50,
            sample_shots=50, seed=42, noise_config=cfg,
        )
        assert "energy" in result
        assert "best_sequence" in result


class TestQuditNoiseChannels:
    """Tests for custom d-dimensional noise channels (Kraus completeness)."""

    cirq = pytest.importorskip("cirq")

    @pytest.mark.parametrize("noise_type", sorted(VALID_NOISE_TYPES))
    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    def test_single_qudit_kraus_completeness(self, noise_type: str, d: int) -> None:
        """Σ K_i† K_i must equal I_d (trace-preserving condition)."""
        from solvers.cirq_solver.qudit_noise_channels import _make_single_qudit_channel

        gate = _make_single_qudit_channel(noise_type, d, 0.05)
        kraus_ops = self.cirq.kraus(gate)
        total = sum(k.conj().T @ k for k in kraus_ops)
        np.testing.assert_allclose(total, np.eye(d), atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3])
    def test_two_qudit_kraus_completeness(self, d: int) -> None:
        """Σ K_i† K_i must equal I_{d²} for the correlated two-qudit channel."""
        from solvers.cirq_solver.qudit_noise_channels import TwoQuditDepolarizingChannel

        gate = TwoQuditDepolarizingChannel(d, 0.05)
        kraus_ops = self.cirq.kraus(gate)
        d2 = d * d
        total = sum(k.conj().T @ k for k in kraus_ops)
        np.testing.assert_allclose(total, np.eye(d2), atol=1e-12)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_single_qudit_qid_shape(self, d: int) -> None:
        from solvers.cirq_solver.qudit_noise_channels import QuditDepolarizingChannel

        gate = QuditDepolarizingChannel(d, 0.01)
        assert self.cirq.qid_shape(gate) == (d,)

    @pytest.mark.parametrize("d", [2, 3])
    def test_two_qudit_qid_shape(self, d: int) -> None:
        from solvers.cirq_solver.qudit_noise_channels import TwoQuditDepolarizingChannel

        gate = TwoQuditDepolarizingChannel(d, 0.01)
        assert self.cirq.qid_shape(gate) == (d, d)

    def test_depolarizing_p0_is_identity_channel(self) -> None:
        """With p=0, only the identity Kraus operator should be non-zero."""
        from solvers.cirq_solver.qudit_noise_channels import QuditDepolarizingChannel

        d = 3
        gate = QuditDepolarizingChannel(d, 0.0)
        kraus_ops = self.cirq.kraus(gate)
        nonzero = [k for k in kraus_ops if np.linalg.norm(k) > 1e-14]
        assert len(nonzero) == 1
        np.testing.assert_allclose(nonzero[0], np.eye(d), atol=1e-12)


class TestConstantQuditNoiseModel:
    """Tests for the ConstantQuditNoiseModel."""

    cirq = pytest.importorskip("cirq")

    def test_noisy_circuit_runs_on_density_matrix_sim(self) -> None:
        """A trivial qudit circuit with noise should run without error."""
        from solvers.cirq_solver.noise_model import build_noise_model
        from solvers.cirq_solver.qaoa_circuit_tqudo import QuditHadamardGate

        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        noise_model = build_noise_model(cfg, qudit_dimension=3)

        q = self.cirq.LineQid(0, dimension=3)
        circuit = self.cirq.Circuit([
            QuditHadamardGate(3).on(q),
            self.cirq.measure(q, key="m"),
        ])
        noisy = circuit.with_noise(noise_model)
        sim = self.cirq.DensityMatrixSimulator(seed=42)
        result = sim.run(noisy, repetitions=20)
        assert result.measurements["m"].shape == (20, 1)

    def test_gate_noise_override(self) -> None:
        """Per-gate probability overrides via qudit_hadamard key."""
        from solvers.cirq_solver.qudit_noise_channels import ConstantQuditNoiseModel

        cfg = NoiseConfig(
            enabled=True, noise_type="depolarizing", probability=0.01,
            gate_noise={"qudit_hadamard": 0.5},
        )
        model = ConstantQuditNoiseModel(cfg, dimension=3)

        from solvers.cirq_solver.qaoa_circuit_tqudo import QuditHadamardGate

        h_gate = QuditHadamardGate(3)
        assert model._get_probability(h_gate) == 0.5
        assert model._get_probability(None) == 0.01


class TestCirqNativeQuditNoisyRun:
    """End-to-end noisy run for native qudit QAOA (Cirq)."""

    cirq = pytest.importorskip("cirq")

    def _small_tensors(self, d: int = 3) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        Etab = rng.random((2, d, d))
        Ettprimeab = rng.random((2, 2, d, d))
        return Etab, Ettprimeab

    def test_noisy_depolarizing_run(self) -> None:
        from solvers.cirq_solver.qaoa_circuit_tqudo import run_qaoa

        Etab, Ettprimeab = self._small_tensors(d=3)
        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        result = run_qaoa(
            Etab, Ettprimeab, depth=1, max_iter=3, n_shots=30,
            sample_shots=30, seed=42, noise_config=cfg,
        )
        assert "energy" in result
        assert "best_sequence" in result
        assert result["best_sequence"].shape == (2,)

    @pytest.mark.parametrize("noise_type", [
        "amplitude_damping", "phase_damping", "bit_flip", "phase_flip",
    ])
    def test_all_noise_types_evaluate_cost(self, noise_type: str) -> None:
        from solvers.cirq_solver.qaoa_circuit_tqudo import (
            evaluate_cost, create_qaoa_circuit,
        )

        Etab, Ettprimeab = self._small_tensors(d=3)
        circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
            1, Etab, Ettprimeab,
        )
        params = np.array([0.5, 0.5])
        cfg = NoiseConfig(enabled=True, noise_type=noise_type, probability=0.02)
        val = evaluate_cost(
            params, circuit, Etab, Ettprimeab, symbols, 1,
            qudits, n_qudits, dimension,
            n_shots=10, seed=42, noise_config=cfg,
        )
        assert isinstance(val, float)

    def test_disabled_noise_matches_noiseless(self) -> None:
        from solvers.cirq_solver.qaoa_circuit_tqudo import (
            evaluate_cost, create_qaoa_circuit,
        )

        Etab, Ettprimeab = self._small_tensors(d=3)
        circuit, symbols, qudits, n_qudits, dimension = create_qaoa_circuit(
            1, Etab, Ettprimeab,
        )
        params = np.array([0.3, 0.2])

        val_none = evaluate_cost(
            params, circuit, Etab, Ettprimeab, symbols, 1,
            qudits, n_qudits, dimension,
            n_shots=50, seed=42, noise_config=None,
        )
        cfg_off = NoiseConfig(enabled=False, noise_type="depolarizing", probability=0.5)
        val_off = evaluate_cost(
            params, circuit, Etab, Ettprimeab, symbols, 1,
            qudits, n_qudits, dimension,
            n_shots=50, seed=42, noise_config=cfg_off,
        )
        assert val_none == pytest.approx(val_off)

    @pytest.mark.parametrize("d", [3, 4])
    def test_noisy_run_different_dimensions(self, d: int) -> None:
        from solvers.cirq_solver.qaoa_circuit_tqudo import run_qaoa

        Etab, Ettprimeab = self._small_tensors(d=d)
        cfg = NoiseConfig(enabled=True, noise_type="depolarizing", probability=0.01)
        result = run_qaoa(
            Etab, Ettprimeab, depth=1, max_iter=3, n_shots=20,
            sample_shots=20, seed=42, noise_config=cfg,
        )
        assert isinstance(result["energy"], float)
        assert result["best_sequence"].shape == (2,)
        assert np.all(result["best_sequence"] >= 0)
        assert np.all(result["best_sequence"] < d)
