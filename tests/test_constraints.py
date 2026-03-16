"""Tests for solution constraint validators (T-QUDO and QUBO)."""

import numpy as np

from instance_gen_process.models import ProblemInstance
from utils.constraints import (
    idx,
    qubo_binary_to_sequence,
    sequence_to_qubo_binary,
    validate_solution_constraints_qubo,
    validate_solution_constraints_tqudo,
)


def _minimal_instance(
    n_cities: int = 5,
    precedences: list[tuple[int, int]] | None = None,
) -> ProblemInstance:
    """Create a minimal ProblemInstance for constraint testing."""
    n_available = n_cities - 1
    return ProblemInstance(
        n_cities=n_cities,
        precedences=precedences or [],
        prices_hotels=np.zeros((n_available, n_available)),
        prices_travels=np.zeros((n_cities, n_cities, n_cities)),
    )


class TestValidateSolutionConstraintsTqudo:
    """Tests for validate_solution_constraints_tqudo."""

    def test_valid_solution_no_precedences(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        solution = [0, 1, 2, 3]  # Valid permutation
        assert validate_solution_constraints_tqudo(instance, solution) is True

    def test_valid_solution_with_precedence(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[(1, 3)])  # 1 before 3
        solution = [0, 1, 2, 3]  # 1 at pos 1, 3 at pos 3
        assert validate_solution_constraints_tqudo(instance, solution) is True

    def test_invalid_precedence(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[(3, 1)])  # 3 before 1
        solution = [0, 1, 2, 3]  # 1 at pos 1, 3 at pos 3 -> 1 before 3, violates 3 before 1
        assert validate_solution_constraints_tqudo(instance, solution) is False

    def test_valid_precedence_reordered(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[(3, 1)])
        solution = [0, 2, 3, 1]  # 3 at pos 2, 1 at pos 3
        assert validate_solution_constraints_tqudo(instance, solution) is True

    def test_duplicate_node_rejected(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        solution = [0, 1, 1, 3]  # Duplicate 1
        assert validate_solution_constraints_tqudo(instance, solution) is False

    def test_wrong_length_rejected(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        assert validate_solution_constraints_tqudo(instance, [0, 1, 2]) is False
        assert validate_solution_constraints_tqudo(instance, [0, 1, 2, 3, 4]) is False

    def test_out_of_range_node_rejected(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        solution = [0, 1, 2, 10]  # 10 out of range
        assert validate_solution_constraints_tqudo(instance, solution) is False

    def test_accepts_numpy_array(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        solution = np.array([0, 1, 2, 3])
        assert validate_solution_constraints_tqudo(instance, solution) is True


class TestValidateSolutionConstraintsQubo:
    """Tests for validate_solution_constraints_qubo."""

    def _solution_from_sequence(self, sequence: list[int], n_available: int) -> np.ndarray:
        """Build QUBO binary vector from a sequence (t -> city)."""
        x = np.zeros(n_available * n_available)
        for t, city in enumerate(sequence):
            x[idx(t, city, n_available)] = 1
        return x

    def test_valid_solution_no_precedences(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        n_available = 4
        solution = self._solution_from_sequence([0, 1, 2, 3], n_available)
        assert validate_solution_constraints_qubo(instance, solution) is True

    def test_valid_solution_with_precedence(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[(1, 3)])
        n_available = 4
        solution = self._solution_from_sequence([0, 1, 2, 3], n_available)
        assert validate_solution_constraints_qubo(instance, solution) is True

    def test_invalid_precedence(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[(3, 1)])
        n_available = 4
        solution = self._solution_from_sequence([0, 1, 2, 3], n_available)
        assert validate_solution_constraints_qubo(instance, solution) is False

    def test_invalid_binary_two_per_timestep(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        n_available = 4
        solution = self._solution_from_sequence([0, 1, 2, 3], n_available)
        solution[idx(0, 0, n_available)] = 1
        solution[idx(0, 1, n_available)] = 1  # Two cities at t=0
        assert validate_solution_constraints_qubo(instance, solution) is False

    def test_invalid_binary_duplicate_city(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        n_available = 4
        # City 1 at t=0 and t=1
        solution = np.zeros(n_available * n_available)
        solution[idx(0, 1, n_available)] = 1
        solution[idx(1, 1, n_available)] = 1
        solution[idx(2, 0, n_available)] = 1
        solution[idx(3, 2, n_available)] = 1
        assert validate_solution_constraints_qubo(instance, solution) is False

    def test_invalid_binary_wrong_length(self) -> None:
        instance = _minimal_instance(n_cities=5, precedences=[])
        solution = np.zeros(10)  # Wrong length
        assert validate_solution_constraints_qubo(instance, solution) is False


class TestQuboBinaryToSequence:
    """Tests for qubo_binary_to_sequence helper."""

    def test_decode_valid(self) -> None:
        n_available = 4
        seq = [2, 0, 3, 1]
        x = np.zeros(n_available * n_available)
        for t, i in enumerate(seq):
            x[idx(t, i, n_available)] = 1
        result = qubo_binary_to_sequence(x, n_available)
        assert result is not None
        assert list(result) == seq

    def test_decode_invalid_length_returns_none(self) -> None:
        assert qubo_binary_to_sequence(np.zeros(5), 4) is None


class TestSequenceToQuboBinary:
    """Tests for sequence_to_qubo_binary helper."""

    def test_encode_decode_roundtrip(self) -> None:
        n_available = 4
        seq = [2, 0, 3, 1]
        binary = sequence_to_qubo_binary(seq, n_available)
        decoded = qubo_binary_to_sequence(binary, n_available)
        assert decoded is not None
        assert list(decoded) == seq

    def test_one_hot_per_timestep(self) -> None:
        n_available = 3
        seq = [0, 1, 2]
        binary = sequence_to_qubo_binary(seq, n_available)
        assert binary.shape == (9,)
        assert binary.sum() == 3
        for t in range(n_available):
            assert binary[idx(t, seq[t], n_available)] == 1.0
