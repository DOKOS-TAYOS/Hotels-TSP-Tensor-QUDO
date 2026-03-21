"""Tests for cost calculation functions."""

import numpy as np
import pytest

from conftest import make_problem_instance as _minimal_instance
from instance_gen_process.models import ProblemInstance, ProblemQUBO, ProblemTQUDO
from utils.costs import calculate_qubo_cost, calculate_real_cost, calculate_tqudo_cost


class TestCalculateQuboCost:
    """Tests for calculate_qubo_cost."""

    def test_simple_quadratic(self) -> None:
        """x^T Q x for known Q and x."""
        Q = np.array([[1, 2], [2, 3]], dtype=float)
        problem = ProblemQUBO(qubo_matrix=Q)
        x = np.array([1, 0])
        cost = calculate_qubo_cost(problem, x)
        assert cost == 1.0  # x^T Q x = [1,0] @ [[1,2],[2,3]] @ [1,0]^T = 1

    def test_symmetric_matrix(self) -> None:
        """Cost for full solution vector."""
        Q = np.array([[2, 1], [1, 2]], dtype=float)
        problem = ProblemQUBO(qubo_matrix=Q)
        x = np.array([1, 1])
        cost = calculate_qubo_cost(problem, x)
        assert cost == 6.0  # 2+1+1+2 = 6

    def test_accepts_2d_array(self) -> None:
        """Solution can be (n, 1) and gets flattened."""
        Q = np.eye(2)
        problem = ProblemQUBO(qubo_matrix=Q)
        x = np.array([[1], [0]])
        cost = calculate_qubo_cost(problem, x)
        assert cost == 1.0


class TestCalculateTqudoCost:
    """Tests for calculate_tqudo_cost."""

    def test_sequence_format(self) -> None:
        """TQUDO cost with qudit sequence (route)."""
        n_available = 3
        Etab = np.zeros((n_available, n_available, n_available))
        Etab[0, 0, 1] = 5.0
        Etab[0, 1, 2] = 3.0
        Etab[1, 0, 1] = 2.0
        Ettprimeab = np.zeros((n_available, n_available, n_available, n_available))
        problem = ProblemTQUDO(Etab=Etab, Ettprimeab=Ettprimeab)
        seq = np.array([0, 1, 2])  # route: t0=0, t1=1, t2=2
        cost = calculate_tqudo_cost(problem, seq)
        # Etab[0,0,1] + Etab[1,1,2] = 5 + 0 (no Etab[1,1,2] set)
        # Actually Etab[1,1,2] = 0, so cost = Etab[0,0,1] + Etab[1,1,2]
        # Etab indices: t, origin, destination. seq[t]=origin, seq[t+1]=destination
        # t=0: origin=0, dest=1 -> Etab[0,0,1]=5
        # t=1: origin=1, dest=2 -> Etab[1,1,2]=0
        assert cost == 5.0


class TestCalculateRealCost:
    """Tests for calculate_real_cost."""

    def test_hotel_and_travel_cost(self) -> None:
        """Sum of hotel and travel costs with known prices."""
        n_cities = 4
        n_available = 3
        prices_hotels = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        prices_travels = np.zeros((n_cities, n_cities, n_cities))
        # start->first: prices_travels[0, n_available, seq[0]]
        prices_travels[0, n_available, 0] = 10.0
        # segment 1: prices_travels[1, 0, 1]
        prices_travels[1, 0, 1] = 20.0
        # segment 2: prices_travels[2, 1, 2]
        prices_travels[2, 1, 2] = 30.0
        # last->start: prices_travels[n_available, 2, n_available]
        prices_travels[n_available, 2, n_available] = 40.0

        instance = ProblemInstance(
            n_cities=n_cities,
            precedences=(),
            prices_hotels=prices_hotels,
            prices_travels=prices_travels,
        )
        sequence = [0, 1, 2]
        cost = calculate_real_cost(instance, sequence)
        # hotel: prices_hotels[0,0] + prices_hotels[1,1] + prices_hotels[2,2] = 1+5+9 = 15
        # travel: 10 + 20 + 30 + 40 = 100
        expected = 15.0 + 100.0
        assert cost == expected

    def test_wrong_length_raises(self) -> None:
        """Sequence length must equal n_available."""
        instance = _minimal_instance(n_cities=5)
        with pytest.raises(ValueError):
            calculate_real_cost(instance, [0, 1, 2])  # length 3, need 4

    def test_accepts_list(self) -> None:
        """Sequence can be list of int."""
        instance = _minimal_instance(n_cities=4)
        cost = calculate_real_cost(instance, [0, 1, 2])
        # 3 hotels (1 each) + 4 travel segments (1 each) = 7
        assert cost == 7.0
