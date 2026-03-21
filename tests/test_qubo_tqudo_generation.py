"""Regression tests for QUBO and TQUDO generation correctness.

These tests verify that:
1. Constraint penalties make infeasible solutions more expensive than feasible ones.
2. QUBO and TQUDO costs are consistent for the same feasible route.
3. The lambda_0/lambda_1 off-diagonal coefficients correctly penalize constraint violations.
"""

from __future__ import annotations

import numpy as np
import pytest

from instance_gen_process.generator import (
    generate_QUBO_from_problem,
    generate_random_instance,
    generate_TQUDO_from_problem,
)
from instance_gen_process.models import (
    InstanceConfig,
    ProblemInstance,
    RestrictionConfig,
)
from utils.constraints import (
    idx,
    sequence_to_qubo_binary,
    validate_solution_constraints_tqudo,
)
from utils.costs import calculate_qubo_cost, calculate_real_cost, calculate_tqudo_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_instance(
    n_cities: int = 4,
    precedences: list[tuple[int, int]] | None = None,
) -> ProblemInstance:
    """Deterministic small instance with known costs for testing."""
    n_available = n_cities - 1
    # Simple hotel costs: hotel[t, city] = 1 for all
    prices_hotels = np.ones((n_available, n_available), dtype=float)
    # Simple travel costs: travel[t, i, j] = 1 for i!=j, 0 for i==j
    prices_travels = np.ones((n_cities, n_cities, n_cities), dtype=float)
    for k in range(n_cities):
        prices_travels[:, k, k] = 0.0
    return ProblemInstance(
        n_cities=n_cities,
        precedences=precedences or [],
        prices_hotels=prices_hotels,
        prices_travels=prices_travels,
    )


def _high_penalty_restriction() -> RestrictionConfig:
    """Penalties much larger than any feasible cost to test constraint enforcement."""
    return RestrictionConfig(lambda_0=1000.0, lambda_1=1000.0, lambda_2=1000.0)


def _zero_penalty_restriction() -> RestrictionConfig:
    """Zero penalties to isolate cost terms from constraint penalties."""
    return RestrictionConfig(lambda_0=0.0, lambda_1=0.0, lambda_2=0.0)


# ---------------------------------------------------------------------------
# QUBO constraint penalty tests (regression for the lambda_0/2 bug)
# ---------------------------------------------------------------------------

class TestQUBOConstraintPenalties:
    """Verify that QUBO penalties correctly distinguish feasible from infeasible."""

    def test_feasible_cheaper_than_two_cities_per_timestep(self) -> None:
        """A valid one-hot solution must be cheaper than one with two cities at t=0.

        This is the direct regression test for the lambda_0/2 bug: with the bug,
        both solutions had IDENTICAL penalty cost, making the constraint ineffective.
        """
        instance = _small_instance(n_cities=4)
        restriction = _high_penalty_restriction()
        qubo = generate_QUBO_from_problem(instance, restriction)

        # Feasible: each timestep has exactly one city
        feasible_seq = [0, 1, 2]
        x_feasible = sequence_to_qubo_binary(feasible_seq, 3)
        cost_feasible = calculate_qubo_cost(qubo, x_feasible)

        # Infeasible: two cities at timestep 0
        x_infeasible = x_feasible.copy()
        x_infeasible[idx(0, 0, 3)] = 1
        x_infeasible[idx(0, 1, 3)] = 1  # city 0 AND city 1 at t=0
        cost_infeasible = calculate_qubo_cost(qubo, x_infeasible)

        assert cost_feasible < cost_infeasible, (
            f"Feasible cost ({cost_feasible}) should be strictly less than "
            f"infeasible cost ({cost_infeasible}). "
            "If they are equal, the lambda_0 penalty has no effect."
        )

    def test_feasible_cheaper_than_two_timesteps_per_city(self) -> None:
        """A valid one-hot solution must be cheaper than one with a city used twice.

        Regression test for the lambda_1/2 bug.
        """
        instance = _small_instance(n_cities=4)
        restriction = _high_penalty_restriction()
        qubo = generate_QUBO_from_problem(instance, restriction)

        # Feasible: each city appears exactly once
        feasible_seq = [0, 1, 2]
        x_feasible = sequence_to_qubo_binary(feasible_seq, 3)
        cost_feasible = calculate_qubo_cost(qubo, x_feasible)

        # Infeasible: city 0 at t=0 AND t=1 (city 1 dropped from t=1)
        x_infeasible = np.zeros(9)
        x_infeasible[idx(0, 0, 3)] = 1  # t=0: city 0
        x_infeasible[idx(1, 0, 3)] = 1  # t=1: city 0 again (violation!)
        x_infeasible[idx(2, 2, 3)] = 1  # t=2: city 2
        cost_infeasible = calculate_qubo_cost(qubo, x_infeasible)

        assert cost_feasible < cost_infeasible, (
            f"Feasible cost ({cost_feasible}) should be strictly less than "
            f"infeasible cost ({cost_infeasible}). "
            "If they are equal, the lambda_1 penalty has no effect."
        )

    def test_penalty_scales_with_lambda(self) -> None:
        """Higher lambda should produce a larger gap between feasible/infeasible costs."""
        instance = _small_instance(n_cities=4)

        low = RestrictionConfig(lambda_0=10.0, lambda_1=10.0, lambda_2=10.0)
        high = RestrictionConfig(lambda_0=100.0, lambda_1=100.0, lambda_2=100.0)

        qubo_low = generate_QUBO_from_problem(instance, low)
        qubo_high = generate_QUBO_from_problem(instance, high)

        feasible_seq = [0, 1, 2]
        x_feasible = sequence_to_qubo_binary(feasible_seq, 3)

        # Two cities at t=0 (lambda_0 violation)
        x_bad = x_feasible.copy()
        x_bad[idx(0, 1, 3)] = 1

        gap_low = calculate_qubo_cost(qubo_low, x_bad) - calculate_qubo_cost(qubo_low, x_feasible)
        gap_high = calculate_qubo_cost(qubo_high, x_bad) - calculate_qubo_cost(qubo_high, x_feasible)

        assert gap_high > gap_low > 0, (
            f"Gap should grow with lambda. low={gap_low}, high={gap_high}"
        )

    def test_all_feasible_permutations_satisfy_onehot(self) -> None:
        """Every valid permutation must satisfy both one-hot constraints (lambda_0, lambda_1).

        For a 3-city instance, there are 3! = 6 permutations. All should produce
        the same total penalty contribution (which should be the minimum).
        """
        import itertools

        instance = _small_instance(n_cities=4)
        restriction = _high_penalty_restriction()
        qubo = generate_QUBO_from_problem(instance, restriction)

        n_available = 3
        costs = []
        for perm in itertools.permutations(range(n_available)):
            x = sequence_to_qubo_binary(list(perm), n_available)
            costs.append(calculate_qubo_cost(qubo, x))

        # All feasible solutions should have the same penalty component
        # (differences come only from the route cost, not from penalties)
        # With zero-cost instance, all should be equal
        zero_instance = ProblemInstance(
            n_cities=4,
            precedences=[],
            prices_hotels=np.zeros((3, 3)),
            prices_travels=np.zeros((4, 4, 4)),
        )
        qubo_zero_cost = generate_QUBO_from_problem(zero_instance, restriction)
        penalty_costs = []
        for perm in itertools.permutations(range(n_available)):
            x = sequence_to_qubo_binary(list(perm), n_available)
            penalty_costs.append(calculate_qubo_cost(qubo_zero_cost, x))

        # All feasible permutations should have identical penalty cost
        assert all(
            abs(c - penalty_costs[0]) < 1e-10 for c in penalty_costs
        ), f"Feasible permutations should have equal penalty costs, got: {penalty_costs}"


# ---------------------------------------------------------------------------
# QUBO precedence penalty tests
# ---------------------------------------------------------------------------

class TestQUBOPrecedencePenalty:
    """Verify that precedence penalties correctly penalize violated orderings."""

    def test_precedence_violation_is_penalized(self) -> None:
        """Route violating (0 before 1) must cost more than a valid route."""
        instance = _small_instance(n_cities=4, precedences=[(0, 1)])
        restriction = _high_penalty_restriction()
        qubo = generate_QUBO_from_problem(instance, restriction)

        # Valid: city 0 at t=0, city 1 at t=1 → 0 before 1 ✓
        x_valid = sequence_to_qubo_binary([0, 1, 2], 3)
        cost_valid = calculate_qubo_cost(qubo, x_valid)

        # Invalid: city 1 at t=0, city 0 at t=1 → 0 after 1 ✗
        x_invalid = sequence_to_qubo_binary([1, 0, 2], 3)
        cost_invalid = calculate_qubo_cost(qubo, x_invalid)

        assert cost_valid < cost_invalid, (
            f"Valid precedence cost ({cost_valid}) should be < "
            f"invalid precedence cost ({cost_invalid})"
        )


# ---------------------------------------------------------------------------
# TQUDO generation tests
# ---------------------------------------------------------------------------

class TestTQUDOGeneration:
    """Verify TQUDO tensor construction and cost consistency."""

    def test_tqudo_cost_matches_real_cost_no_penalties(self) -> None:
        """With zero penalties, TQUDO cost should equal real cost for feasible routes."""
        config = InstanceConfig(
            n_cities=5,
            n_precedences_range=(0, 0),
            prices_range_hotels=(10.0, 50.0),
            prices_range_travels=(10.0, 50.0),
            seed=42,
        )
        instance = generate_random_instance(config, 42)
        restriction = _zero_penalty_restriction()
        tqudo = generate_TQUDO_from_problem(instance, restriction)

        # Test all 4! = 24 permutations of 4 available cities
        import itertools
        n_available = instance.n_cities - 1
        for perm in itertools.permutations(range(n_available)):
            seq = list(perm)
            tqudo_cost = calculate_tqudo_cost(tqudo, np.array(seq))
            real_cost = calculate_real_cost(instance, seq)
            assert abs(tqudo_cost - real_cost) < 1e-10, (
                f"TQUDO cost ({tqudo_cost}) != real cost ({real_cost}) for seq={seq}"
            )

    def test_tqudo_precedence_penalty_applied(self) -> None:
        """TQUDO should penalize routes that violate precedence constraints."""
        instance = _small_instance(n_cities=4, precedences=[(0, 1)])
        restriction = _high_penalty_restriction()
        tqudo = generate_TQUDO_from_problem(instance, restriction)

        # Valid: 0 before 1
        cost_valid = calculate_tqudo_cost(tqudo, np.array([0, 1, 2]))
        # Invalid: 1 before 0
        cost_invalid = calculate_tqudo_cost(tqudo, np.array([1, 0, 2]))

        assert cost_valid < cost_invalid

    def test_tqudo_duplicate_city_penalty(self) -> None:
        """TQUDO should penalize solutions where a city appears at multiple timesteps."""
        instance = _small_instance(n_cities=4, precedences=[])
        restriction = _high_penalty_restriction()
        tqudo = generate_TQUDO_from_problem(instance, restriction)

        # Valid: no duplicates
        cost_valid = calculate_tqudo_cost(tqudo, np.array([0, 1, 2]))
        # Invalid: city 0 at both t=0 and t=1
        cost_invalid = calculate_tqudo_cost(tqudo, np.array([0, 0, 2]))

        assert cost_valid < cost_invalid


# ---------------------------------------------------------------------------
# Cross-formulation consistency
# ---------------------------------------------------------------------------

class TestCrossFormulationConsistency:
    """Verify that QUBO and TQUDO agree on the ranking of feasible solutions."""

    def test_qubo_and_tqudo_agree_on_best_route(self) -> None:
        """Both formulations should identify the same optimal feasible route."""
        import itertools

        config = InstanceConfig(
            n_cities=4,
            n_precedences_range=(1, 1),
            prices_range_hotels=(10.0, 100.0),
            prices_range_travels=(10.0, 100.0),
            seed=123,
        )
        instance = generate_random_instance(config, 123)
        restriction = _high_penalty_restriction()

        qubo = generate_QUBO_from_problem(instance, restriction)
        tqudo = generate_TQUDO_from_problem(instance, restriction)

        n_available = instance.n_cities - 1
        best_qubo_seq = None
        best_qubo_cost = float("inf")
        best_tqudo_seq = None
        best_tqudo_cost = float("inf")

        for perm in itertools.permutations(range(n_available)):
            seq = list(perm)

            x = sequence_to_qubo_binary(seq, n_available)
            qcost = calculate_qubo_cost(qubo, x)
            if qcost < best_qubo_cost:
                best_qubo_cost = qcost
                best_qubo_seq = seq

            tcost = calculate_tqudo_cost(tqudo, np.array(seq))
            if tcost < best_tqudo_cost:
                best_tqudo_cost = tcost
                best_tqudo_seq = seq

        assert best_qubo_seq == best_tqudo_seq, (
            f"QUBO optimal route {best_qubo_seq} differs from "
            f"TQUDO optimal route {best_tqudo_seq}"
        )
        # Both optimal routes should also be feasible
        assert validate_solution_constraints_tqudo(instance, best_tqudo_seq)


# ---------------------------------------------------------------------------
# Config validation regression
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Regression tests for config loader validation changes."""

    def test_n_cities_2_rejected(self) -> None:
        """n_cities=2 should be rejected to avoid degenerate TQUDO costs."""
        from instance_gen_process.config_loader import load_instance_config
        from pathlib import Path
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "n_cities": 2,
                "n_precedences_range": [0, 0],
                "prices_range_hotels": [10.0, 50.0],
                "prices_range_travels": [10.0, 50.0],
                "seed": 42,
            }, f)
            f.flush()
            with pytest.raises(ValueError, match="at least 3"):
                load_instance_config(Path(f.name))

    def test_n_cities_3_accepted(self) -> None:
        """n_cities=3 should be accepted."""
        from instance_gen_process.config_loader import load_instance_config
        from pathlib import Path
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "n_cities": 3,
                "n_precedences_range": [0, 1],
                "prices_range_hotels": [10.0, 50.0],
                "prices_range_travels": [10.0, 50.0],
                "seed": 42,
            }, f)
            f.flush()
            config = load_instance_config(Path(f.name))
            assert config.n_cities == 3
