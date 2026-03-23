"""Tests for the SA initial temperature estimation module."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_problem_instance
from instance_gen_process.generator import generate_random_instance
from instance_gen_process.models import InstanceConfig
from solvers.simulated_annealing import T0EstimationResult, estimate_initial_temperature


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CONFIG = InstanceConfig(
    n_cities=5,
    n_precedences_range=(1, 2),
    prices_range_hotels=(30, 150),
    prices_range_travels=(30, 150),
)


@pytest.fixture()
def small_instance():
    """A small random instance suitable for fast T₀ estimation."""
    return generate_random_instance(SMALL_CONFIG, seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formulation", ["tqudo", "qubo"])
def test_estimate_returns_positive_temperature(small_instance, formulation: str):
    result = estimate_initial_temperature(
        small_instance, formulation=formulation, seed=99,
    )
    assert isinstance(result, T0EstimationResult)
    assert result.t0 > 0
    assert result.n_samples > 0


@pytest.mark.parametrize("formulation", ["tqudo", "qubo"])
def test_estimate_achieves_target_chi(small_instance, formulation: str):
    chi_0 = 0.8
    epsilon = 1e-3
    result = estimate_initial_temperature(
        small_instance,
        formulation=formulation,
        chi_0=chi_0,
        epsilon=epsilon,
        seed=99,
    )
    if result.converged:
        assert abs(result.chi_achieved - chi_0) <= epsilon


def test_deterministic_with_seed(small_instance):
    r1 = estimate_initial_temperature(small_instance, seed=123)
    r2 = estimate_initial_temperature(small_instance, seed=123)
    assert r1.t0 == r2.t0
    assert r1.chi_achieved == r2.chi_achieved
    assert r1.iterations == r2.iterations


def test_higher_chi_gives_higher_t0(small_instance):
    low = estimate_initial_temperature(small_instance, chi_0=0.5, seed=7)
    high = estimate_initial_temperature(small_instance, chi_0=0.95, seed=7)
    assert high.t0 > low.t0


def test_flat_landscape_fallback():
    """An instance with uniform costs has no uphill transitions."""
    n_cities = 4
    n_available = n_cities - 1
    instance = make_problem_instance(
        n_cities=n_cities,
        prices_hotels=np.full((n_available, n_available), 50.0),
        prices_travels=np.full((n_cities, n_cities, n_cities), 50.0),
    )
    result = estimate_initial_temperature(instance, seed=0)
    assert result.t0 > 0
    assert result.converged is False


def test_invalid_formulation(small_instance):
    with pytest.raises(ValueError, match="formulation"):
        estimate_initial_temperature(small_instance, formulation="tqudo_virtual")


def test_invalid_chi_0(small_instance):
    with pytest.raises(ValueError, match="chi_0"):
        estimate_initial_temperature(small_instance, chi_0=1.5)
